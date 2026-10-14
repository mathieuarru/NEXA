"""Score benchmark JSONL records: per-arm report, paired comparison or run-to-run variability.

Records from every system are accepted; identity fields a record lacks are
filled in from the file and arm directory names.

Per-arm report (one arm directory -> <dir>/report.txt and report.json): accuracy
with Wilson 95% CI, open-ended set metrics, latency, tokens, errors, results by
answer-set size and a capped list of incorrect questions.
  python -m benchmark.evaluate_benchmark benchmark/results/alzkb/nexa-direct_gpt-5-mini

Paired comparison (per-type accuracy delta, exact McNemar p, regressions):
  python -m benchmark.evaluate_benchmark \\
      --compare benchmark/results/alzkb/nexa-direct_gpt-5-mini \\
                benchmark/results/alzkb/nexa-flex_gpt-5-mini \\
      --label-a direct --label-b flex

Variability over repeated runs (records tagged with --run-tag):
  python -m benchmark.evaluate_benchmark --variability <arm dir>... -o <report.txt>
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from benchmark.bench_utils import (
        METRIC_NAMES,
        answer_sets,
        compute_set_metrics,
        entity_key,
        entity_list_to_set,
        hops_from_name,
        iter_jsonl,
        list_to_set,
        mcnemar_exact_p,
        mean,
        mean_ci95_across_runs,
        percentile,
        question_id,
        rescore_open_ended,
        stdev,
        wilson_ci,
    )
except ImportError:  # executed as a plain script from inside benchmark/
    from bench_utils import (
        METRIC_NAMES,
        answer_sets,
        compute_set_metrics,
        entity_key,
        entity_list_to_set,
        hops_from_name,
        iter_jsonl,
        list_to_set,
        mcnemar_exact_p,
        mean,
        mean_ci95_across_runs,
        percentile,
        question_id,
        rescore_open_ended,
        stdev,
        wilson_ci,
    )

RESULTS_DIR = os.path.join(os.path.dirname(__file__) or ".", "results")

# Open-ended stratification: |expected| buckets.
SIZE_BUCKETS = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 10 ** 9, ">20")]

# Types whose answer is a set of entities: set metrics and |expected| strata.
# Per-type answer parsing lives in bench_utils.answer_sets.
SET_METRIC_TYPES = ("open_ended", "open_ended_list")


def answer_set_of(ans: Any, btype: str) -> set:
    """Case-insensitive entity set from a bare answer payload (no record around it)."""
    parsed = entity_list_to_set(ans or []) if btype == "open_ended_list" else list_to_set(ans)
    return {entity_key(t) for t in parsed}


# ---------------------------------------------------------------------------
# Loading + normalization
# ---------------------------------------------------------------------------

def collect_input_paths(inputs: List[str]) -> List[str]:
    """JSONL paths from files, directories or globs (default benchmark/results/*.jsonl), deduplicated."""
    paths: List[str] = []
    if not inputs:
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.jsonl")))
    else:
        for inp in inputs:
            if os.path.isdir(inp):
                paths.extend(sorted(glob.glob(os.path.join(inp, "*.jsonl"))))
            elif any(ch in inp for ch in "*?["):
                paths.extend(sorted(glob.glob(inp)))
            elif inp.endswith(".jsonl") and os.path.isfile(inp):
                paths.append(inp)
    seen, ordered = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def _filename_fallbacks(path: str) -> Dict[str, Any]:
    """Model and agent mode for records that lack them, from the file and directory names.

    File names follow `dataset__type__model__kgmode__[flex]__...` (`agentic` is
    an alias of flex). An arm directory `<system>_<model>` takes precedence.
    """
    parts = os.path.splitext(os.path.basename(path))[0].split("__")
    out: Dict[str, Any] = {}
    if len(parts) >= 3:
        out["model"] = parts[2]
    rest = parts[3:]
    out["agent_mode"] = "flex" if ("flex" in rest or "agentic" in rest) else "direct"
    # Baseline file names follow no convention and some baseline records carry
    # neither field, so the arm directory decides.
    system, _, rest = os.path.basename(os.path.dirname(os.path.abspath(path))).partition("_")
    model = rest.split("_")[0]  # ablation arms append a suffix: nexa-direct_gpt-4o-mini_bare
    if system in _ARM_MODE and model:
        out["agent_mode"] = _ARM_MODE[system]
        out["model"] = model
    return out


# Arm directory prefix -> the agent_mode its records carry when they carry one.
_ARM_MODE = {"vanilla": "vanilla", "graphcypher": "graphcypher", "nexa-direct": "direct", "nexa-flex": "flex"}


def normalize_record(rec: Dict[str, Any], path: str, fallbacks: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing identity fields and score open-ended answers with the current scorer."""
    rec.setdefault("file", os.path.basename(path))
    rec.setdefault("type", "unknown")
    if not rec.get("qid") and isinstance(rec.get("question"), str):
        rec["qid"] = question_id(rec["question"])
    if rec.get("hops") is None:
        rec["hops"] = hops_from_name(rec.get("file", ""))
    if not rec.get("agent_mode"):
        rec["agent_mode"] = fallbacks.get("agent_mode", "direct")
    # "agentic" is an alias of "flex".
    if rec.get("agent_mode") == "agentic":
        rec["agent_mode"] = "flex"
    if not rec.get("model"):
        rec["model"] = fallbacks.get("model", "unknown")
    if rec.get("type") in SET_METRIC_TYPES:
        if isinstance(rec.get("metrics"), dict):
            # Stored scores reflect the scorer at run time.
            rescore_open_ended(rec)
        else:
            gold, pred = answer_sets(rec)
            rec["metrics"] = compute_set_metrics(gold, pred)
    return rec


def load_records(paths: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for p in paths:
        fallbacks = _filename_fallbacks(p)
        for rec in iter_jsonl(p):
            records.append(normalize_record(rec, p, fallbacks))
    return records


GroupKey = Tuple[str, str, str, str]  # (type, agent_mode, model, dataset file)


def group_records(records: List[Dict[str, Any]]) -> Dict[GroupKey, List[Dict[str, Any]]]:
    groups: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r["type"], r["agent_mode"], r["model"], r["file"])].append(r)
    return groups


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _num_values(records: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(r[key]) for r in records
            if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]


def _usage_values(records: List[Dict[str, Any]], key: str) -> List[float]:
    vals = []
    for r in records:
        u = r.get("usage")
        if isinstance(u, dict) and isinstance(u.get(key), (int, float)):
            vals.append(float(u[key]))
    return vals


def summarize_group(
    records: List[Dict[str, Any]],
    bench_type: str,
    price_input: Optional[float] = None,
    price_output: Optional[float] = None,
) -> Dict[str, Any]:
    """Accuracy with Wilson CI, set metrics (open-ended) and telemetry of a group of records."""
    n = len(records)
    correct = sum(1 for r in records if r.get("correct"))
    lo, hi = wilson_ci(correct, n)
    summary: Dict[str, Any] = {
        "count": n,
        "correct": correct,
        "accuracy": (correct / n) if n else 0.0,
        "accuracy_ci95": [lo, hi],
    }

    if bench_type in SET_METRIC_TYPES:
        metric_rows = [r["metrics"] for r in records if isinstance(r.get("metrics"), dict)]
        summary["avg_metrics"] = {
            k: mean([float(m.get(k, 0.0)) for m in metric_rows]) for k in METRIC_NAMES
        }
        # Stratify by |expected|: large answer sets are where retrieval gets hard.
        buckets: Dict[str, Dict[str, Any]] = {}
        for lo_b, hi_b, label in SIZE_BUCKETS:
            in_bucket = [r for r in records if lo_b <= len(answer_sets(r)[0]) <= hi_b]
            if not in_bucket:
                continue
            b_correct = sum(1 for r in in_bucket if r.get("correct"))
            buckets[label] = {
                "count": len(in_bucket),
                "accuracy": b_correct / len(in_bucket),
                "f1": mean([float((r.get("metrics") or {}).get("f1", 0.0)) for r in in_bucket]),
            }
        summary["by_expected_size"] = buckets

    # --- Telemetry ---------------------------------------------------------
    telemetry: Dict[str, Any] = {}

    fas = [r["first_attempt_success"] for r in records
           if isinstance(r.get("first_attempt_success"), bool)]
    if fas:
        telemetry["first_attempt_success_rate"] = {"value": sum(fas) / len(fas), "n": len(fas)}
    iters = _num_values(records, "n_correction_iterations")
    if iters:
        telemetry["avg_correction_iterations"] = {"value": mean(iters), "n": len(iters)}

    lats = sorted(_num_values(records, "latency_ms"))
    if lats:
        telemetry["latency_ms"] = {
            "mean": mean(lats),
            "p50": percentile(lats, 50),
            "p95": percentile(lats, 95),
            "n": len(lats),
        }

    in_toks = _usage_values(records, "input_tokens")
    out_toks = _usage_values(records, "output_tokens")
    if in_toks or out_toks:
        tokens = {
            "avg_input": mean(in_toks),
            "avg_output": mean(out_toks),
            "total_input": sum(in_toks),
            "total_output": sum(out_toks),
            "n": max(len(in_toks), len(out_toks)),
        }
        if price_input is not None and price_output is not None:
            total_cost = (sum(in_toks) * price_input + sum(out_toks) * price_output) / 1e6
            tokens["total_cost_usd"] = total_cost
            tokens["avg_cost_usd"] = total_cost / tokens["n"] if tokens["n"] else 0.0
        telemetry["tokens"] = tokens

    # Flex-only counters.
    tool_calls = _num_values(records, "n_tool_calls")
    if tool_calls:
        telemetry["avg_tool_calls"] = {"value": mean(tool_calls), "n": len(tool_calls)}
    exec_cypher = _num_values(records, "n_execute_cypher")
    if exec_cypher:
        telemetry["avg_execute_cypher"] = {"value": mean(exec_cypher), "n": len(exec_cypher)}
        no_cypher = sum(1 for v in exec_cypher if v == 0)
        telemetry["no_cypher_rate"] = {"value": no_cypher / len(exec_cypher), "n": len(exec_cypher)}
    recursion = [r["recursion_hit"] for r in records if isinstance(r.get("recursion_hit"), bool)]
    if recursion:
        telemetry["recursion_hit_rate"] = {"value": sum(recursion) / len(recursion), "n": len(recursion)}

    # Completeness guard (only in runs that recorded it). cited_incomplete_rate:
    # the answer relied on a result known to be incomplete.
    trunc = _num_values(records, "n_truncated_cypher")
    if trunc:
        telemetry["truncated_cypher_rate"] = {
            "value": sum(1 for v in trunc if v > 0) / len(trunc), "n": len(trunc),
        }
    cited_inc = _num_values(records, "n_cited_incomplete_results")
    if cited_inc:
        telemetry["cited_incomplete_rate"] = {
            "value": sum(1 for v in cited_inc if v > 0) / len(cited_inc), "n": len(cited_inc),
        }

    # Flex answer rescue (only in runs that recorded it): answers rebuilt after
    # the agent loop failed to deliver one.
    rescued = [r["answer_rescued"] for r in records if isinstance(r.get("answer_rescued"), bool)]
    if rescued:
        telemetry["answer_rescued_rate"] = {
            "value": sum(rescued) / len(rescued), "n": len(rescued),
        }
        salvaged = [r for r in records if r.get("answer_rescued")]
        if salvaged:
            telemetry["answer_rescued_accuracy"] = {
                "value": sum(1 for r in salvaged if r.get("correct")) / len(salvaged),
                "n": len(salvaged),
            }

    n_error = sum(1 for r in records if r.get("error"))
    telemetry["error_rate"] = {"value": (n_error / n) if n else 0.0, "n": n}

    summary["telemetry"] = telemetry
    return summary


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _fmt_ci(summary: Dict[str, Any]) -> str:
    lo, hi = summary["accuracy_ci95"]
    return f"{summary['accuracy']:.3f} [95% CI {lo:.3f}-{hi:.3f}]"


def render_telemetry(lines: List[str], telemetry: Dict[str, Any], indent: str = "  ") -> None:
    def add(label: str, text: str):
        lines.append(f"{indent}- {label}: {text}")

    if "first_attempt_success_rate" in telemetry:
        t = telemetry["first_attempt_success_rate"]
        add("first attempt success rate", f"{t['value']:.3f} (n={t['n']})")
    if "avg_correction_iterations" in telemetry:
        t = telemetry["avg_correction_iterations"]
        add("avg correction iterations", f"{t['value']:.3f} (n={t['n']})")
    if "latency_ms" in telemetry:
        t = telemetry["latency_ms"]
        add("latency ms", f"mean={t['mean']:.0f} p50={t['p50']:.0f} p95={t['p95']:.0f} (n={t['n']})")
    if "tokens" in telemetry:
        t = telemetry["tokens"]
        text = (f"avg in={t['avg_input']:.0f} out={t['avg_output']:.0f} | "
                f"total in={t['total_input']:.0f} out={t['total_output']:.0f}")
        if "total_cost_usd" in t:
            text += f" | cost ${t['total_cost_usd']:.2f} (${t['avg_cost_usd'] * 1000:.2f}/1k q)"
        add("tokens", text)
    if "avg_tool_calls" in telemetry:
        t = telemetry["avg_tool_calls"]
        add("avg tool calls", f"{t['value']:.2f} (n={t['n']})")
    if "avg_execute_cypher" in telemetry:
        t = telemetry["avg_execute_cypher"]
        add("avg execute_cypher", f"{t['value']:.2f} (n={t['n']})")
    if "no_cypher_rate" in telemetry:
        t = telemetry["no_cypher_rate"]
        add("answered without any Cypher (fabrication risk)", f"{t['value']:.3f} (n={t['n']})")
    if "recursion_hit_rate" in telemetry:
        t = telemetry["recursion_hit_rate"]
        add("recursion hit rate", f"{t['value']:.3f} (n={t['n']})")
    if "truncated_cypher_rate" in telemetry:
        t = telemetry["truncated_cypher_rate"]
        add("questions with a LIMIT-truncated Cypher", f"{t['value']:.3f} (n={t['n']})")
    if "cited_incomplete_rate" in telemetry:
        t = telemetry["cited_incomplete_rate"]
        add("answered from a known-incomplete result (unsound)", f"{t['value']:.3f} (n={t['n']})")
    if "answer_rescued_rate" in telemetry:
        t = telemetry["answer_rescued_rate"]
        add("answers rebuilt by the constrained rescue", f"{t['value']:.3f} (n={t['n']})")
    if "answer_rescued_accuracy" in telemetry:
        t = telemetry["answer_rescued_accuracy"]
        add("  of those, correct", f"{t['value']:.3f} (n={t['n']})")
    if "error_rate" in telemetry:
        t = telemetry["error_rate"]
        add("records with error", f"{t['value']:.3f} (n={t['n']})")


def render_wrong_records(
    lines: List[str], records: List[Dict[str, Any]], bench_type: str, max_wrong: int
) -> None:
    wrong = [r for r in records if not r.get("correct")]
    if not wrong:
        return
    shown = wrong if max_wrong < 0 else wrong[:max_wrong]
    lines.append(f"-- Incorrect Questions ({len(wrong)} total, showing {len(shown)}) --")
    for r in shown:
        lines.append(f"Q [{r.get('qid', '?')}]: {r.get('question')}")
        lines.append(f"  Expected: {r.get('expected')}")
        lines.append(f"  Got:      {r.get('model_answer')}")
        if r.get("error"):
            lines.append(f"  Error:    {r.get('error')}")
        lines.append(f"  Thinking: {r.get('thinking', 'N/A')}")
        cypher_first = r.get("cypher_first_attempt")
        cypher_final = r.get("cypher_final")
        if cypher_first:
            lines.append(f"  Cypher (1st attempt): {cypher_first}")
        if cypher_final and cypher_final != cypher_first:
            lines.append(f"  Cypher (final):       {cypher_final}")
        if isinstance(r.get("n_correction_iterations"), int):
            lines.append(f"  Correction iters:     {r['n_correction_iterations']}")

        trace = r.get("trace")
        if isinstance(trace, list) and trace:
            extras = []
            if isinstance(r.get("n_tool_calls"), int):
                extras.append(f"{r['n_tool_calls']} tool calls")
            if r.get("recursion_hit"):
                extras.append("recursion limit hit")
            head = "  Flex trace" + (f" ({', '.join(extras)})" if extras else "")
            lines.append(head + ":")
            for ev in trace:
                rid = ev.get("result_id")
                rid_s = f" [{str(rid)[:8]}]" if rid else ""
                lines.append(f"    - {ev.get('name')}: {ev.get('summary')}{rid_s}")

        if bench_type in SET_METRIC_TYPES:
            m = r.get("metrics") or {}
            lines.append("  Metrics: " + "  ".join(f"{k}={float(m.get(k, 0.0)):.3f}" for k in METRIC_NAMES))
            exp_set, got_set = answer_sets(r)
            missing = sorted(exp_set - got_set)
            extra = sorted(got_set - exp_set)
            lines.append(f"  Missing (in expected, not in got): {', '.join(missing) if missing else '-'}")
            lines.append(f"  Extra (in got, not in expected):   {', '.join(extra) if extra else '-'}")
        lines.append("")


def generate_report(
    paths: List[str],
    output_path: Optional[str] = None,
    max_wrong: int = 15,
    price_input: Optional[float] = None,
    price_output: Optional[float] = None,
) -> Tuple[str, str]:
    """Build the text + JSON reports. Returns (txt_path, json_path)."""
    records = load_records(paths)
    groups = group_records(records)

    lines: List[str] = []
    json_summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "files": paths,
        "groups": [],
        "overall": [],
    }

    lines.append("--- NEXA Benchmark Evaluation ---")
    lines.append(f"Files processed: {len(paths)}")
    for p in paths:
        lines.append(f"  - {p}")

    for key in sorted(groups.keys()):
        btype, agent_mode, model, base_file = key
        recs = groups[key]
        summary = summarize_group(recs, btype, price_input, price_output)

        lines.append("")
        lines.append(f"=== Dataset: {base_file} | Type: {btype} | Mode: {agent_mode} | Model: {model} ===")
        lines.append(
            f"Count: {summary['count']}  |  Correct: {summary['correct']}  |  Accuracy: {_fmt_ci(summary)}"
        )
        if btype in SET_METRIC_TYPES:
            avg = summary.get("avg_metrics", {})
            lines.append("Average set metrics: " + "  ".join(f"{k}={avg.get(k, 0.0):.3f}" for k in METRIC_NAMES))
            buckets = summary.get("by_expected_size") or {}
            if buckets:
                lines.append("By |expected| size:")
                for label, b in buckets.items():
                    lines.append(
                        f"  - {label:>4}: n={b['count']:<5} acc={b['accuracy']:.3f} f1={b['f1']:.3f}"
                    )
        lines.append("Telemetry:")
        render_telemetry(lines, summary["telemetry"])
        lines.append("")
        render_wrong_records(lines, recs, btype, max_wrong)

        json_summary["groups"].append({
            "type": btype, "agent_mode": agent_mode, "model": model, "file": base_file,
            **summary,
        })

    # ---- Overall, grouped by (type, agent_mode, model), record-weighted ----
    lines.append("")
    lines.append("==== Overall Summary (record-weighted) ====")
    overall_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        overall_groups[(r["type"], r["agent_mode"], r["model"])].append(r)
    for okey in sorted(overall_groups.keys()):
        btype, agent_mode, model = okey
        recs = overall_groups[okey]
        summary = summarize_group(recs, btype, price_input, price_output)
        lines.append("")
        lines.append(f"Type: {btype} | Mode: {agent_mode} | Model: {model}")
        lines.append(
            f"  Total: {summary['count']}  |  Correct: {summary['correct']}  |  Accuracy: {_fmt_ci(summary)}"
        )
        if btype in SET_METRIC_TYPES:
            avg = summary.get("avg_metrics", {})
            lines.append("  Avg set metrics: " + "  ".join(f"{k}={avg.get(k, 0.0):.3f}" for k in METRIC_NAMES))
        render_telemetry(lines, summary["telemetry"], indent="  ")
        json_summary["overall"].append({
            "type": btype, "agent_mode": agent_mode, "model": model, **summary,
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_DIR, f"summary_{ts}.txt")
    json_path = os.path.splitext(output_path)[0] + ".json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, ensure_ascii=False, indent=2)

    return output_path, json_path


# ---------------------------------------------------------------------------
# Paired comparison (e.g. direct vs flex)
# ---------------------------------------------------------------------------

def _index_by_qid(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Index records by (type, qid). Later records win (latest result kept)."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in records:
        qid = r.get("qid")
        if qid:
            out[(r["type"], qid)] = r
    return out


def compare_runs(
    paths_a: List[str],
    paths_b: List[str],
    label_a: str,
    label_b: str,
    max_regressions: int = 20,
) -> Tuple[List[str], Dict[str, Any]]:
    """Pair two runs by (type, qid): accuracy delta, exact McNemar p and regressions per type."""
    recs_a = _index_by_qid(load_records(paths_a))
    recs_b = _index_by_qid(load_records(paths_b))
    common = sorted(set(recs_a) & set(recs_b))

    lines: List[str] = []
    lines.append(f"==== Paired comparison: A={label_a} vs B={label_b} ====")
    lines.append(f"A records: {len(recs_a)}  |  B records: {len(recs_b)}  |  paired: {len(common)}")
    json_out: Dict[str, Any] = {"label_a": label_a, "label_b": label_b, "paired": len(common), "by_type": {}}

    by_type: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for key in common:
        by_type[key[0]].append((recs_a[key], recs_b[key]))

    for btype in sorted(by_type.keys()):
        pairs = by_type[btype]
        n = len(pairs)
        acc_a = sum(1 for a, _ in pairs if a.get("correct")) / n
        acc_b = sum(1 for _, b in pairs if b.get("correct")) / n
        b_disc = sum(1 for a, b in pairs if a.get("correct") and not b.get("correct"))
        c_disc = sum(1 for a, b in pairs if not a.get("correct") and b.get("correct"))
        p = mcnemar_exact_p(b_disc, c_disc)

        lines.append("")
        lines.append(f"--- Type: {btype} (n={n} paired) ---")
        lines.append(f"  Accuracy {label_a}: {acc_a:.3f}  |  {label_b}: {acc_b:.3f}  |  delta: {acc_b - acc_a:+.3f}")
        lines.append(
            f"  Discordant pairs: {label_a}-only correct={b_disc}, {label_b}-only correct={c_disc}"
            f"  |  McNemar exact p={p:.4g}"
        )
        entry: Dict[str, Any] = {
            "n": n, "acc_a": acc_a, "acc_b": acc_b, "delta": acc_b - acc_a,
            "a_only_correct": b_disc, "b_only_correct": c_disc, "mcnemar_p": p,
        }

        if btype in SET_METRIC_TYPES:
            f1_a = mean([float((a.get("metrics") or {}).get("f1", 0.0)) for a, _ in pairs])
            f1_b = mean([float((b.get("metrics") or {}).get("f1", 0.0)) for _, b in pairs])
            lines.append(f"  Mean F1 {label_a}: {f1_a:.3f}  |  {label_b}: {f1_b:.3f}  |  delta: {f1_b - f1_a:+.3f}")
            entry.update({"f1_a": f1_a, "f1_b": f1_b, "f1_delta": f1_b - f1_a})

        lat_a = mean([float(a.get("latency_ms") or 0) for a, _ in pairs])
        lat_b = mean([float(b.get("latency_ms") or 0) for _, b in pairs])
        lines.append(f"  Mean latency ms {label_a}: {lat_a:.0f}  |  {label_b}: {lat_b:.0f}")
        tok_a = mean([float((a.get("usage") or {}).get("total_tokens") or 0) for a, _ in pairs])
        tok_b = mean([float((b.get("usage") or {}).get("total_tokens") or 0) for _, b in pairs])
        if tok_a or tok_b:
            lines.append(f"  Mean total tokens {label_a}: {tok_a:.0f}  |  {label_b}: {tok_b:.0f}")

        regressions = [(a, b) for a, b in pairs if a.get("correct") and not b.get("correct")]
        if regressions:
            lines.append(f"  Regressions ({label_a} correct, {label_b} wrong), showing "
                         f"{min(len(regressions), max_regressions)}/{len(regressions)}:")
            for a, b in regressions[:max_regressions]:
                lines.append(f"    [{a.get('qid')}] {a.get('question', '')[:90]}")
                lines.append(f"      expected={a.get('expected')} | {label_b} got={b.get('model_answer')}")

        json_out["by_type"][btype] = entry

    return lines, json_out


# ---------------------------------------------------------------------------
# Variability (run-to-run) aggregation
# ---------------------------------------------------------------------------

def _answer_key(ans: Any, btype: str) -> str:
    """Canonical key of an answer: a sorted entity set for open-ended, a normalized string otherwise."""
    if btype in SET_METRIC_TYPES:
        return "|".join(sorted(answer_set_of(ans, btype)))
    return str(ans).strip().lower()


def _mean_pairwise_jaccard(sets: List[set]) -> float:
    """Mean Jaccard similarity over all run pairs (1.0 = identical sets)."""
    if len(sets) < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            total += (len(sets[i] & sets[j]) / len(union)) if union else 1.0
            pairs += 1
    return total / pairs if pairs else 1.0


def variability_report(paths: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """Aggregate repeated runs of the same configuration over a fixed question set.

    Groups by (model, agent_mode, type, dataset file); each `run_tag` is one run.
    Per group: accuracy mean, SD, min-max and a t-based 95% CI over runs (decoding
    variability only, not question sampling); answer consensus (open-ended
    answers compared as sets); correctness consensus and flip rate;
    majority-vote accuracy; and, for open-ended, mean pairwise Jaccard of answers.
    """
    records = load_records(paths)
    # group -> run_tag -> [records]
    groups: Dict[Tuple[str, str, str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        key = (r.get("model", "unknown"), r.get("agent_mode", "direct"),
               r.get("type", "unknown"), r.get("file", "unknown"))
        groups[key][r.get("run_tag") or "run?"].append(r)

    lines: List[str] = ["=" * 78, "RUN-TO-RUN VARIABILITY", "=" * 78]
    json_out: Dict[str, Any] = {"groups": []}

    for key in sorted(groups):
        model, mode, btype, dfile = key
        runs = groups[key]
        run_tags = sorted(runs)
        R = len(run_tags)

        # Per-run accuracy over the questions present in that run.
        run_acc: Dict[str, float] = {}
        for tag in run_tags:
            recs = runs[tag]
            n = len(recs)
            run_acc[tag] = (sum(1 for x in recs if x.get("correct")) / n) if n else 0.0
        accs = [run_acc[t] for t in run_tags]
        mean_acc = mean(accs)
        sd_acc = stdev(accs)

        # Per-question view, restricted to qids present in all runs.
        per_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for tag in run_tags:
            for x in runs[tag]:
                per_q[x.get("qid")].append(x)
        complete = {q: rs for q, rs in per_q.items() if len(rs) == R and q is not None}
        n_q = len(complete)

        ans_consensus = corr_consensus = maj_correct = 0
        jacc_vals: List[float] = []  # per-question mean pairwise Jaccard (open-ended only)
        for q, rs in complete.items():
            keys = [_answer_key(x.get("model_answer"), btype) for x in rs]
            corr_vals = {bool(x.get("correct")) for x in rs}
            if len(set(keys)) == 1:
                ans_consensus += 1
            if len(corr_vals) == 1:
                corr_consensus += 1
            # Majority-vote answer (ties broken by key), scored by its correctness.
            counts = Counter(keys)
            modal = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            modal_correct = next(bool(x.get("correct")) for x, k in zip(rs, keys) if k == modal)
            maj_correct += 1 if modal_correct else 0
            if btype in SET_METRIC_TYPES:
                jacc_vals.append(_mean_pairwise_jaccard(
                    [answer_set_of(x.get("model_answer"), btype) for x in rs]))

        ans_rate = ans_consensus / n_q if n_q else 0.0
        corr_rate = corr_consensus / n_q if n_q else 0.0
        maj_acc = maj_correct / n_q if n_q else 0.0
        mean_jaccard = mean(jacc_vals) if jacc_vals else None

        # t-interval on the mean over runs, not a Wilson interval: the runs replay
        # the same questions.
        mean_ci = mean_ci95_across_runs(accs)
        mean_ci_txt = (
            f"  [95% CI on the {R}-run mean {mean_ci[0]:.3f}-{mean_ci[1]:.3f}]"
            if mean_ci else "  [95% CI on the mean: needs >= 2 runs]"
        )

        lines += [
            "",
            f"[{model} | {mode} | {btype} | {dfile}]",
            f"  runs: {R} ({', '.join(run_tags)})   questions (in all runs): {n_q}",
            f"  accuracy: {mean_acc:.3f} ± {sd_acc:.3f}  "
            f"(min {min(accs):.3f}, max {max(accs):.3f}){mean_ci_txt}",
            f"  per-run: " + "  ".join(f"{t}={run_acc[t]:.3f}" for t in run_tags),
            f"  answer consensus:      {ans_rate:.3f}  "
            f"({ans_consensus}/{n_q} questions with identical "
            f"{'answer set' if btype in SET_METRIC_TYPES else 'answer'} across all runs)",
            f"  correctness consensus: {corr_rate:.3f}  (flip rate {1 - corr_rate:.3f})",
            f"  majority-vote accuracy: {maj_acc:.3f}",
        ]
        if mean_jaccard is not None:
            lines.append(
                f"  content stability:     {mean_jaccard:.3f}  "
                f"(mean pairwise Jaccard of answer sets; 1.0 = identical, order-insensitive)"
            )
        if R < 2:
            lines.append("  warning: only 1 run found; SD and consensus need >= 2 runs "
                         "(one --run-tag value per run).")

        json_out["groups"].append({
            "model": model, "agent_mode": mode, "type": btype, "file": dfile,
            "n_runs": R, "run_tags": run_tags, "n_questions": n_q,
            "accuracy_mean": mean_acc, "accuracy_sd": sd_acc,
            "accuracy_min": min(accs) if accs else 0.0,
            "accuracy_max": max(accs) if accs else 0.0,
            # Distinct from `accuracy_ci95` (binomial over questions).
            "accuracy_mean_ci95_across_runs": list(mean_ci) if mean_ci else None,
            "per_run_accuracy": run_acc,
            "answer_consensus_rate": ans_rate,
            "correctness_consensus_rate": corr_rate,
            "flip_rate": 1 - corr_rate,
            "majority_vote_accuracy": maj_acc,
            "mean_jaccard": mean_jaccard,
        })

    lines.append("")
    return lines, json_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate NEXA benchmark JSONL results.")
    parser.add_argument("inputs", nargs="*",
                        help="JSONL files, directories, or globs. Default: benchmark/results/*.jsonl")
    parser.add_argument("-o", "--output",
                        help="Report path (+ .json beside it). Default: <dir>/report.txt for a single "
                             "directory input, else benchmark/results/summary_<ts>.txt")
    parser.add_argument("--max-wrong", type=int, default=15,
                        help="Max incorrect questions dumped per dataset (-1 = all). Default 15.")
    parser.add_argument("--price-input", type=float,
                        help="USD per 1M input tokens (adds cost estimates).")
    parser.add_argument("--price-output", type=float,
                        help="USD per 1M output tokens (adds cost estimates).")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Two runs (file/dir/glob each) to compare pairwise by question id.")
    parser.add_argument("--label-a", default="A", help="Label for the first compared run.")
    parser.add_argument("--label-b", default="B", help="Label for the second compared run.")
    parser.add_argument("--variability", action="store_true",
                        help="Aggregate repeated runs (by run_tag) of the same config: "
                             "accuracy mean±SD, min-max, answer/correctness consensus, "
                             "majority-vote accuracy.")
    args = parser.parse_args()

    if args.variability:
        paths = collect_input_paths(args.inputs)
        if not paths:
            print("No JSONL inputs found for the variability report.")
            return
        lines, json_out = variability_report(paths)
        text = "\n".join(lines)
        print(text)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = args.output or os.path.join(
            RESULTS_DIR, f"variability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)
        print(f"\nVariability report written to: {out}")
        return

    if args.compare:
        paths_a = collect_input_paths([args.compare[0]])
        paths_b = collect_input_paths([args.compare[1]])
        if not paths_a or not paths_b:
            print("No JSONL inputs found for one of the compared runs.")
            return
        lines, json_out = compare_runs(paths_a, paths_b, args.label_a, args.label_b)
        text = "\n".join(lines)
        print(text)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = args.output or os.path.join(
            RESULTS_DIR, f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(json_out, f, ensure_ascii=False, indent=2)
        print(f"\nComparison written to: {out}")
        return

    paths = collect_input_paths(args.inputs)
    if not paths:
        print("No JSONL inputs found. Provide files/dirs or ensure benchmark/results has JSONL files.")
        return

    # A single arm directory gets its report beside its records (<arm>/report.txt).
    output = args.output
    if not output and len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
        output = os.path.join(args.inputs[0], "report.txt")

    txt_path, json_path = generate_report(
        paths,
        output,
        max_wrong=args.max_wrong,
        price_input=args.price_input,
        price_output=args.price_output,
    )
    print(f"Report written to: {txt_path}")
    print(f"JSON summary written to: {json_path}")


if __name__ == "__main__":
    main()

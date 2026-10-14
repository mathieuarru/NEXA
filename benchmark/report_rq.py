"""Research questions RQ1-RQ3 on AlzKB and BioHopR: paired tests with Holm correction.

Primary metric: accuracy for True/False and MCQ, Jaccard for open-ended (AlzKB)
and BioHopR. Jaccard rather than overlap, which ignores extra entities; EM,
precision, recall, F1 and overlap are descriptive.

Contrasts (Delta = A - B, same backbone or same mode):
  RQ1            NEXA - GraphCypherQAChain
  RQ1 secondary  NEXA - LLM only
  RQ2            Flex - Direct
  RQ3            Qwen3.8-27B - GPT-5-mini (a difference test; no equivalence margin)
Each RQ is one Holm family spanning both benchmarks: RQ1 35 comparisons (AlzKB 30
+ BioHopR 5), RQ1 secondary 35, RQ2 14 (12 + 2), RQ3 14 (12 + 2). Separate
families: the same contrasts on the disambiguated AlzKB True/False 2-hop set
(sensitivity analysis), and NEXA against the published ESCARGOT aggregates
(worst-case McNemar bound, `mcnemar_bounds_from_aggregates`).

A contrast whose arm has not run is "pending" and counts as p = 1 in a family of
the declared size, so adjusted p-values are upper bounds of their final values.
An arm still running is flagged "partial" with its n; BioHopR is ordered by
relation path, so a partial slice is not a random sample.

    poetry run python -m benchmark.report_rq                 # both benchmarks, 100k resamples
    poetry run python -m benchmark.report_rq --n-boot 20000

Writes one report per benchmark to benchmark/results/<benchmark>/stats/rq/.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import namedtuple
from typing import Any, Dict, List, Optional

try:
    from benchmark.paired_stats import (N_BOOT, compare_binary, holm_adjust,
                                        mcnemar_bounds_from_aggregates,
                                        paired_bootstrap_diff)
    from benchmark.published_results import BENCHMARK_N, ESCARGOT_PUBLISHED
    from benchmark.report_paired_stats import PRESETS, graded_score, load_arm, question_id
except ImportError:  # run from inside benchmark/
    from paired_stats import (N_BOOT, compare_binary, holm_adjust,
                              mcnemar_bounds_from_aggregates, paired_bootstrap_diff)
    from published_results import BENCHMARK_N, ESCARGOT_PUBLISHED
    from report_paired_stats import PRESETS, graded_score, load_arm, question_id

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

PRIMARY_SETS = [
    ("True_or_False_1hop.json", "TF 1-hop", "accuracy"),
    ("True_or_False_2hop.json", "TF 2-hop", "accuracy"),
    ("MCQ_1hop.json", "MCQ 1-hop", "accuracy"),
    ("MCQ_2hop.json", "MCQ 2-hop", "accuracy"),
    ("OpenEnded_1hop.json", "OE 1-hop", "jaccard"),
    ("OpenEnded_2hop.json", "OE 2-hop", "jaccard"),
]
# True/False 2-hop without its ambiguous negated conjunction: every RQ1-RQ3
# contrast repeated on it, as one separate Holm family (sensitivity analysis).
SENSITIVITY_SETS = [
    ("True_or_False_2hop_atleastone.json", "TF 2-hop (disambiguated)", "accuracy"),
]

# Per benchmark: question sets, sensitivity sets, ESCARGOT bound, output folder.
BENCHMARKS = {
    "alzkb": dict(title="AlzKB", short="AlzKB", metric_note="accuracy for TF/MCQ, Jaccard for open-ended",
                  primary=PRIMARY_SETS, sensitivity=SENSITIVITY_SETS, escargot=True),
    "biohopr": dict(title="BioHopR / PrimeKG v2.1", short="BioHopR", metric_note="Jaccard index",
                    primary=[("BioHopR_2hop.json", "BioHopR 2-hop", "jaccard")],
                    sensitivity=[], escargot=False),
}


def out_dir(benchmark: str) -> str:
    return os.path.join(_REPO, "benchmark", "results", benchmark, "stats", "rq")


Contrast = namedtuple("Contrast", "a b backbone label")   # Delta = a - b

RQ1 = [
    Contrast("nexa-direct_gpt-4o-mini", "graphcypher_gpt-4o-mini", "GPT-4o-mini", "Direct"),
    Contrast("nexa-direct_gpt-5-mini", "graphcypher_gpt-5-mini", "GPT-5-mini", "Direct"),
    Contrast("nexa-flex_gpt-5-mini", "graphcypher_gpt-5-mini", "GPT-5-mini", "Flex"),
    Contrast("nexa-direct_qwen3.8-27b", "graphcypher_qwen3.8-27b", "Qwen3.8-27B", "Direct"),
    Contrast("nexa-flex_qwen3.8-27b", "graphcypher_qwen3.8-27b", "Qwen3.8-27B", "Flex"),
]
RQ1_SECONDARY = [c._replace(b=c.b.replace("graphcypher_", "vanilla_")) for c in RQ1]
RQ2 = [
    Contrast("nexa-flex_gpt-5-mini", "nexa-direct_gpt-5-mini", "GPT-5-mini", "Flex − Direct"),
    Contrast("nexa-flex_qwen3.8-27b", "nexa-direct_qwen3.8-27b", "Qwen3.8-27B", "Flex − Direct"),
]
RQ3 = [
    Contrast("nexa-direct_qwen3.8-27b", "nexa-direct_gpt-5-mini", "Qwen − GPT-5-mini", "Direct"),
    Contrast("nexa-flex_qwen3.8-27b", "nexa-flex_gpt-5-mini", "Qwen − GPT-5-mini", "Flex"),
]

QUESTIONS = {
    "RQ1": dict(title="With the same backbone, does NEXA outperform GraphCypherQAChain?",
                delta="NEXA − GraphCypherQAChain", contrasts=RQ1, kind="difference"),
    "RQ1_secondary": dict(title="Secondary: with the same backbone, does NEXA outperform the LLM alone?",
                          delta="NEXA − LLM only", contrasts=RQ1_SECONDARY, kind="difference"),
    "RQ2": dict(title="With the same backbone, does Flex improve on Direct?",
                delta="Flex − Direct", contrasts=RQ2, kind="difference"),
    "RQ3": dict(title="With the same mode, does the local model perform differently from GPT-5-mini?",
                delta="Qwen3.8-27B − GPT-5-mini (same mode)", contrasts=RQ3, kind="difference"),
}


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

_ARMS: Dict[str, Optional[dict]] = {}
PRESET = "alzkb"     # set by main() to each benchmark in turn


def arm(label: str) -> Optional[dict]:
    """Records of one arm, or None if it is not registered or has no records."""
    if (PRESET, label) not in _ARMS:
        globs = dict(PRESETS[PRESET]).get(label)
        recs = load_arm(globs, label=label) if globs else {}
        _ARMS[(PRESET, label)] = recs or None
    return _ARMS[(PRESET, label)]


def set_size(dataset: str) -> Optional[int]:
    """Questions in the source question set, to tell a complete arm from one still running."""
    path = os.path.join(_REPO, "benchmark", "datasets", PRESET, dataset)
    try:
        with open(path, encoding="utf-8") as fh:
            items = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(items, dict):
        items = next((v for v in items.values() if isinstance(v, list)), [])
    return len({question_id(it["question"]) for it in items
                if isinstance(it, dict) and isinstance(it.get("question"), str)})


def compare(c: Contrast, dataset: str, metric: str, n_boot: int) -> Dict[str, Any]:
    """One contrast on one question set: the paired test, or why it is missing."""
    row: Dict[str, Any] = {"a": c.a, "b": c.b, "backbone": c.backbone, "label": c.label,
                           "dataset": dataset, "metric": metric}
    A, B = arm(c.a), arm(c.b)
    missing = [x for x, recs in ((c.a, A), (c.b, B)) if recs is None]
    if missing:
        return {**row, "status": "pending", "missing": missing}
    keys = sorted(k for k in A if k[0] == dataset and k in B)
    if not keys:
        absent = [x for x, recs in ((c.a, A), (c.b, B)) if not any(k[0] == dataset for k in recs)]
        return {**row, "status": "not run on this set", "missing": absent}
    if metric == "accuracy":
        r = compare_binary([bool(A[k].get("correct")) for k in keys],
                           [bool(B[k].get("correct")) for k in keys])
        score_a, score_b = r["acc_a"], r["acc_b"]
    else:
        r = paired_bootstrap_diff([graded_score(A[k], metric) for k in keys],
                                  [graded_score(B[k], metric) for k in keys],
                                  n_boot=n_boot)
        score_a, score_b = r["mean_a"], r["mean_b"]
    size = set_size(dataset)
    partial = [x for x, recs in ((c.a, A), (c.b, B))
               if size and sum(1 for k in recs if k[0] == dataset) < size]
    return {**row, "status": "ok", "n": r["n"], "test": r["test"],
            "partial": partial, "set_size": size,
            "score_a": 100 * score_a, "score_b": 100 * score_b,
            "delta": 100 * r["delta"], "ci95": [100 * r["ci_lo"], 100 * r["ci_hi"]],
            "p": r["p"], "p_floor": 1.0 / n_boot
            if r["test"] != "mcnemar_exact" else None,
            "b_only_a": r.get("b_only_a"), "c_only_b": r.get("c_only_b")}


def apply_holm(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Holm-adjust one family in place; return its size and pending count.

    Pending comparisons are members at p = 1, so adjusted values are upper
    bounds. Comparisons never planned (an arm not run on that set) are not members.
    """
    members = [r for r in rows if r["status"] in ("ok", "pending")]
    ps = [r["p"] if r["status"] == "ok" else 1.0 for r in members]
    for r, adj in zip(members, holm_adjust(ps)):
        if r["status"] == "ok":
            r["p_holm"] = adj
            r["significant"] = adj < 0.05
    n_pending = sum(1 for r in members if r["status"] == "pending")
    return {"size": len(members), "pending": n_pending, "provisional": n_pending > 0}


def run_question(key: str, sets, n_boot: int) -> Dict[str, Any]:
    """Every contrast of one research question on the current benchmark's sets, before Holm."""
    q = QUESTIONS[key]
    rows = [compare(c, ds, metric, n_boot) | {"set_label": lbl}
            for c in q["contrasts"] for ds, lbl, metric in sets]
    return {"key": key, **{k: v for k, v in q.items() if k != "contrasts"}, "rows": rows}


def apply_holm_across(results: Dict[str, Dict[str, Any]], key: str) -> None:
    """Holm-adjust one research question as a single family across all benchmarks.

    Each benchmark's report records the family size and its own member count.
    """
    rows = [r for b in results for r in results[b][key]["rows"]]
    fam = apply_holm(rows)
    fam["members"] = {b: sum(1 for r in results[b][key]["rows"] if r["status"] in ("ok", "pending"))
                      for b in results}
    for b in results:
        results[b][key]["family"] = fam


def escargot_bounds() -> List[Dict[str, Any]]:
    """NEXA Direct GPT-4o-mini against the published ESCARGOT accuracies (binary sets only).

    Counted over the full official question sets (MCQ 1-hop: 498), the
    denominator of the published figures.
    """
    rows = []
    for ds, pct in ESCARGOT_PUBLISHED["accuracy"].items():
        recs = [json.loads(line) for f in glob.glob(os.path.join(
                    _REPO, "benchmark/results/alzkb/nexa-direct_gpt-4o-mini", f"{ds[:-5]}__*.jsonl"))
                for line in open(f, encoding="utf-8") if line.strip()]
        n = len(recs)
        if n != BENCHMARK_N[ds]:
            rows.append({"dataset": ds, "status": f"n mismatch ({n} vs {BENCHMARK_N[ds]})"})
            continue
        k_a = sum(1 for r in recs if r.get("correct"))
        b = mcnemar_bounds_from_aggregates(k_a, n, pct)
        rows.append({"dataset": ds, "status": "ok", "n": n, "nexa": 100 * k_a / n,
                     "escargot": pct, "delta": 100 * k_a / n - pct, **b})
    adj = holm_adjust([r["p_worst"] for r in rows if r["status"] == "ok"])
    for r, a in zip([r for r in rows if r["status"] == "ok"], adj):
        r["p_worst_holm"] = a
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _p_text(r: Dict[str, Any]) -> str:
    """Raw p. A Monte-Carlo p at its floor is only known to be below it."""
    floor = r.get("p_floor")
    return f"<{floor:.0e}" if floor and r["p"] <= floor + 1e-15 else f"{r['p']:.1e}"


def _holm_text(r: Dict[str, Any]) -> str:
    v = r.get("p_holm")
    if v is None:
        return "—"
    floor = r.get("p_floor")
    return f"<{v:.0e}" if floor and r["p"] <= floor + 1e-15 else f"{v:.1e}"


METRIC_SHORT = {"accuracy": "Acc", "jaccard": "Jac"}


def render(results: Dict[str, Any], args) -> List[str]:
    """Text report of one benchmark."""
    L: List[str] = []
    W = 108
    L += ["=" * W, f"{results['settings']['title']}: pre-declared comparisons "
          f"(primary metric: {results['settings']['metric_note']})", "=" * W, ""]
    L += [f"  bootstrap resamples: {args.n_boot:,}", ""]

    for key in ("RQ1", "RQ2", "RQ3", "RQ1_secondary"):
        q = results[key]
        fam = q.get("family", {})
        L += ["-" * W, f"{key}  {q['title']}", f"     Delta = {q['delta']}   (percentage points)"]
        if fam:
            split = " + ".join(f"{n} {BENCHMARKS[b]['short']}" for b, n in fam.get("members", {}).items())
            L.append(f"     Holm family: {fam['size']} comparisons across both benchmarks ({split})"
                     + (f" (PROVISIONAL: {fam['pending']} pending counted as p = 1)"
                        if fam["provisional"] else ""))
        lw = max(len(c.label) for c in QUESTIONS[key]["contrasts"])
        bw = max(12, max(len(c.backbone) for c in QUESTIONS[key]["contrasts"]))
        L += ["-" * W,
              f"  {'set':26s} {'backbone':{bw}s} {'contrast':<{lw}} {'':3} {'n':>4} {'A':>6} {'B':>6} "
              f"{'Delta':>7} {'95% CI':>17} {'b/c':>8} {'p':>9} {'p Holm':>9}"]
        for r in q["rows"]:
            if r["status"] != "ok":
                L.append(f"  {r['set_label']:26s} {r['backbone']:{bw}s} {r['label']:<{lw}}  "
                         f"{r['status']}{' (' + ', '.join(r.get('missing', [])) + ')' if r.get('missing') else ''}")
                continue
            bc = f"{r['b_only_a']}/{r['c_only_b']}" if r["b_only_a"] is not None else "—"
            star = "  *" if r.get("significant") else ""
            if r.get("partial"):
                star += f"  PARTIAL ({', '.join(r['partial'])}: {r['n']} of {r['set_size']} questions)"
            L.append(f"  {r['set_label']:26s} {r['backbone']:{bw}s} {r['label']:<{lw}} {METRIC_SHORT[r['metric']]:3} "
                     f"{r['n']:4d} {r['score_a']:6.1f} {r['score_b']:6.1f} {r['delta']:+7.1f} "
                     f"[{r['ci95'][0]:+6.1f},{r['ci95'][1]:+6.1f}] {bc:>8} {_p_text(r):>9} "
                     f"{_holm_text(r):>9}{star}")
        L.append("")

    fam = results.get("sensitivity_family") or {"size": 0}
    if fam["size"]:
        L += _render_sensitivity(results, fam, W)
    if results.get("escargot_published"):
        L += _render_escargot(results, W)
    L += ["", "* significant after Holm correction within its family (alpha = 0.05); the RQ families",
          "  span both benchmarks, the sensitivity and ESCARGOT families are AlzKB only.", ""]
    return L


def _render_sensitivity(results: Dict[str, Any], fam: Dict[str, Any], W: int) -> List[str]:
    L: List[str] = []
    L += ["-" * W, "Sensitivity analysis: the same contrasts on the disambiguated True/False 2-hop set",
          f"     One Holm family across RQ1-RQ3: {fam['size']} comparisons"
          + (f" (PROVISIONAL: {fam['pending']} pending counted as p = 1)" if fam["provisional"] else ""),
          "-" * W]
    for key in ("RQ1", "RQ2", "RQ3"):
        s = results[f"{key}_sensitivity"]
        L += [f"  {key}"]
        for r in s["rows"]:
            if r["status"] != "ok":
                L.append(f"  {r['backbone']:18s} {r['label']:14s} {r['status']}")
                continue
            ci = r["ci95"]
            tail = f"  p={_p_text(r)}  p Holm={_holm_text(r)}{'  *' if r.get('significant') else ''}"
            L.append(f"  {r['backbone']:18s} {r['label']:14s} n={r['n']:4d} {r['score_a']:6.1f} vs "
                     f"{r['score_b']:6.1f}  Delta={r['delta']:+6.1f} [{ci[0]:+6.1f},{ci[1]:+6.1f}]{tail}")
    L.append("")
    return L


def _render_escargot(results: Dict[str, Any], W: int) -> List[str]:
    L: List[str] = []
    L += ["-" * W, "Supplementary: NEXA Direct GPT-4o-mini vs published ESCARGOT (aggregates only)",
          "     No per-question records: the discordant counts are bounded, not observed.",
          "     p_worst = largest McNemar p over every pairing compatible with both accuracies.",
          "-" * W,
          f"  {'set':22s} {'n':>4} {'NEXA':>6} {'ESC.':>6} {'Delta':>7} {'p best':>9} {'p worst':>9} {'Holm':>9}"]
    for r in results["escargot_published"]:
        if r["status"] != "ok":
            L.append(f"  {r['dataset'][:-5]:22s} {r['status']}")
            continue
        L.append(f"  {r['dataset'][:-5]:22s} {r['n']:4d} {r['nexa']:6.1f} {r['escargot']:6.1f} "
                 f"{r['delta']:+7.1f} {r['p_best']:9.1e} {r['p_worst']:9.1e} {r['p_worst_holm']:9.1e}"
                 f"{'  *' if r['p_worst_holm'] < 0.05 else ''}")
    return L


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Flat table for the Supplementary; intervals split into _lo / _hi columns."""
    flat, keys = [], []
    for r in rows:
        f = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
        for k in ("ci95",):
            if isinstance(r.get(k), list):
                f[f"{k}_lo"], f[f"{k}_hi"] = r[k]
        flat.append(f)
        keys += [k for k in f if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)


def main() -> None:
    global PRESET
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help=f"bootstrap / permutation resamples for the Jaccard tests (default {N_BOOT:,})")
    args = ap.parse_args()

    results: Dict[str, Dict[str, Any]] = {}
    for name, bench in BENCHMARKS.items():
        PRESET = name
        res: Dict[str, Any] = {}
        for key in ("RQ1", "RQ1_secondary", "RQ2", "RQ3"):
            res[key] = run_question(key, bench["primary"], args.n_boot)
            if key != "RQ1_secondary" and bench["sensitivity"]:
                res[f"{key}_sensitivity"] = run_question(key, bench["sensitivity"], args.n_boot)
        if bench["sensitivity"]:
            res["sensitivity_family"] = apply_holm(
                [r for key in ("RQ1", "RQ2", "RQ3") for r in res[f"{key}_sensitivity"]["rows"]])
        if bench["escargot"]:
            res["escargot_published"] = escargot_bounds()
        res["settings"] = {"benchmark": name, "title": bench["title"],
                           "metric_note": bench["metric_note"], "n_boot": args.n_boot,
                           "primary_metric": {ds: m for ds, _, m in bench["primary"]},
                           "holm_families": "one per research question, across both benchmarks"}
        results[name] = res
    for key in ("RQ1", "RQ1_secondary", "RQ2", "RQ3"):
        apply_holm_across(results, key)

    for name, res in results.items():
        target = out_dir(name)
        os.makedirs(target, exist_ok=True)
        text = "\n".join(render(res, args))
        with open(os.path.join(target, "report.txt"), "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        with open(os.path.join(target, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False, default=str)
        for key in ("RQ1", "RQ1_secondary", "RQ2", "RQ3"):
            write_csv(os.path.join(target, f"{key.lower()}.csv"),
                      res[key]["rows"] + res.get(f"{key}_sensitivity", {}).get("rows", []))
        if "escargot_published" in res:
            write_csv(os.path.join(target, "escargot_published.csv"), res["escargot_published"])
        print(text)
        print(f"written to {target}/ (report.txt, report.json, rq*.csv"
              + (", escargot_published.csv)" if "escargot_published" in res else ")") + "\n")


if __name__ == "__main__":
    main()

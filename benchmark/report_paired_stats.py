"""Marginal intervals and paired tests between the arms of one experiment.

  marginal   per arm and question set: accuracy with Wilson 95% CI.
  paired     per pair of arms and question set: the difference with its 95% CI
             and discordant counts; exact McNemar p and Newcombe CI for binary
             outcomes, BCa bootstrap CI and sign-flip p for graded ones.

With `--reference`, the Holm family is every comparison against that arm;
otherwise all pairs. Holm is applied within each metric. Records are paired by
(question set, qid); qid is recomputed from the question text when missing, and
the baselines' file suffix ("MCQ_1hop.json__vanilla") is stripped.

Usage
-----
    # the AlzKB experiment, everything against NEXA Direct
    poetry run python -m benchmark.report_paired_stats --preset alzkb \\
        --reference nexa-direct_gpt-5-mini \\
        -o benchmark/results/alzkb/stats/paired

    # any set of arms, ad hoc
    poetry run python -m benchmark.report_paired_stats \\
        "flex=benchmark/results/biohopr/nexa-flex_gpt-5-mini/*.jsonl" \\
        "direct=benchmark/results/biohopr/nexa-direct_gpt-5-mini/*.jsonl" \\
        -o /tmp/biohopr_paired
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from benchmark.bench_utils import question_id, rescore_open_ended, wilson_ci
    from benchmark.paired_stats import N_BOOT, compare_binary, holm_adjust, paired_bootstrap_diff, stars
except ImportError:  # running as a script from inside benchmark/
    from bench_utils import question_id, rescore_open_ended, wilson_ci
    from paired_stats import N_BOOT, compare_binary, holm_adjust, paired_bootstrap_diff, stars


# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------
# One directory per arm: benchmark/results/<experiment>/<system>_<model>/*.jsonl.
# Labels are listed by hand, in display order.

def _arms(experiment: str, labels: Sequence[str]) -> List[Tuple[str, List[str]]]:
    return [(label, [f"benchmark/results/{experiment}/{label}/*.jsonl"]) for label in labels]


PRESETS: Dict[str, List[Tuple[str, List[str]]]] = {
    "alzkb": _arms("alzkb", [
        "vanilla_gpt-4o-mini", "vanilla_gpt-5-mini",
        "graphcypher_gpt-4o-mini", "graphcypher_gpt-5-mini",
        "nexa-direct_gpt-4o-mini", "nexa-direct_gpt-5-mini", "nexa-flex_gpt-5-mini",
        "nexa-direct_qwen3.8-27b", "nexa-flex_qwen3.8-27b",
        "vanilla_qwen3.8-27b", "graphcypher_qwen3.8-27b",
    ]),
    "biohopr": _arms("biohopr", [
        "vanilla_gpt-4o-mini", "vanilla_gpt-5-mini",
        "graphcypher_gpt-4o-mini", "graphcypher_gpt-5-mini",
        "nexa-direct_gpt-4o-mini", "nexa-direct_gpt-5-mini", "nexa-flex_gpt-5-mini",
        "nexa-direct_qwen3.8-27b", "nexa-flex_qwen3.8-27b",
        "vanilla_qwen3.8-27b", "graphcypher_qwen3.8-27b",
    ]),
}

# Set-answer types: `correct` is exact set equality there, so they also get
# graded comparisons.
GRADED_TYPES = {"open_ended", "open_ended_list"}
GRADED_METRICS = ["f1", "recall", "precision", "jaccard"]

# Display order for AlzKB; anything unlisted is appended alphabetically.
DATASET_ORDER = [
    "True_or_False_1hop.json", "True_or_False_2hop.json",
    "True_or_False_2hop_atleastone.json",
    "MCQ_1hop.json", "MCQ_2hop.json",
    "OpenEnded_1hop.json", "OpenEnded_2hop.json",
    "BioHopR_2hop.json",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(pattern: str) -> List[str]:
    """Expand a glob; a relative one that matches nothing is retried from the repository root.

    Registry paths are relative to the root, but some importers (figures/) run
    from elsewhere, where the glob would silently match nothing.
    """
    hits = glob.glob(pattern)
    if hits or os.path.isabs(pattern):
        return hits
    return glob.glob(os.path.join(_REPO_ROOT, pattern))


def normalise_dataset(name: str) -> str:
    """Strip the baselines' suffix ("MCQ_1hop.json__vanilla" -> "MCQ_1hop.json")."""
    return (name or "").split("__")[0]


_DATASET_GLOBS = ("benchmark/datasets/alzkb/*.json", "benchmark/datasets/biohopr/*.json")
_conflicts_cache: Optional[set] = None


def dataset_conflicts() -> set:
    """(question set, qid) pairs listed more than once with different gold answers.

    Read from the source datasets, not from arm records: the baselines skip
    repeated question texts while the NEXA runner keeps them, so a per-arm check
    would exclude different questions in different arms.
    """
    global _conflicts_cache
    if _conflicts_cache is not None:
        return _conflicts_cache
    seen: Dict[Tuple[str, str], set] = defaultdict(set)
    for pattern in _DATASET_GLOBS:
        for path in _resolve(pattern):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            items = data if isinstance(data, list) else next(
                (v for v in data.values() if isinstance(v, list)), []) if isinstance(data, dict) else []
            ds = os.path.basename(path)
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("question"), str):
                    gold = json.dumps(it.get("answer"), sort_keys=True, ensure_ascii=False)
                    seen[(ds, question_id(it["question"]))].add(gold)
    _conflicts_cache = {k for k, golds in seen.items() if len(golds) > 1}
    return _conflicts_cache


def _norm_model(name: Optional[str]) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def load_arm(patterns: Sequence[str], label: Optional[str] = None) -> Dict[Tuple[str, str], dict]:
    """Records of one arm, keyed by (question set, qid).

    Open-ended records are rescored with the current scorer; a missing qid is
    recomputed with `question_id`. Questions listed twice with different gold
    answers are dropped. Raises ValueError when records name a backbone other
    than the label's (`<system>_<model>`), or several backbones without a label.
    """
    out: Dict[Tuple[str, str], dict] = {}
    conflicting: Dict[Tuple[str, str], str] = {}
    file_models: Dict[str, set] = {}
    for pattern in patterns:
        for path in sorted(_resolve(pattern)):
            file_models[path] = set()
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    file_models[path].add(rec.get("model"))
                    question = rec.get("question")
                    if not question:
                        continue
                    # Stored scores reflect the scorer at run time.
                    rescore_open_ended(rec)
                    qid = rec.get("qid") or question_id(question)
                    key = (normalise_dataset(rec.get("file")), qid)
                    prev = out.get(key)
                    if prev is not None and str(prev.get("expected")) != str(rec.get("expected")):
                        conflicting[key] = question
                    out[key] = rec

    # A question listed twice with different golds is unscorable and has no single
    # identity: drop it. The per-arm check covers question sets with no source file.
    drop = set(conflicting) | (dataset_conflicts() & set(out))
    for key in drop:
        out.pop(key, None)

    # One arm, one backbone, read from the label: another backbone's file matching
    # the glob would overwrite answers. Records without a `model` field pass.
    declared = _norm_model(label.split("_", 1)[1]) if label and "_" in label else None
    named = {m for ms in file_models.values() for m in ms if m}
    if declared:
        wrong = sorted(m for m in named if not _norm_model(m).startswith(declared))
        if wrong:
            detail = "; ".join(f"{os.path.basename(p)}: {sorted(map(str, ms))}"
                               for p, ms in file_models.items())
            raise ValueError(f"arm {label!r} holds records from {wrong}: {detail}")
    elif len({_norm_model(m) for m in named}) > 1:
        detail = "; ".join(f"{os.path.basename(p)}: {sorted(map(str, ms))}"
                           for p, ms in file_models.items())
        raise ValueError(f"arm mixes records from different models: {detail}")
    return out


def graded_score(rec: dict, metric: str) -> float:
    """Per-question graded score, 0 when the answer could not be scored.

    Unscorable answers stay in the denominator, so every metric of a comparison
    uses the same n.
    """
    m = rec.get("metrics")
    v = m.get(metric) if isinstance(m, dict) else None
    return float(v) if isinstance(v, (int, float)) else 0.0


def is_graded(recs: Sequence[dict]) -> bool:
    return any((r.get("type") or "") in GRADED_TYPES for r in recs)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def dataset_sort_key(name: str) -> Tuple[int, str]:
    return (DATASET_ORDER.index(name), "") if name in DATASET_ORDER else (len(DATASET_ORDER), name)


def build(arms: List[Tuple[str, Dict[Tuple[str, str], dict]]],
          reference: Optional[str],
          n_boot: int) -> Tuple[List[str], Dict[str, Any]]:
    """Marginal table, paired comparisons (Holm per metric) and the CI-overlap diagnostic."""
    lines: List[str] = []
    payload: Dict[str, Any] = {"arms": [], "marginal": {}, "paired": [], "family": None}

    labels = [lbl for lbl, _ in arms]
    by_label = dict(arms)
    datasets = sorted({ds for _, recs in arms for ds, _ in recs}, key=dataset_sort_key)

    lines.append("=" * 100)
    lines.append("MARGINAL: accuracy with Wilson 95% CI (sampling uncertainty, per arm)")
    lines.append("=" * 100)
    lines.append("")
    for lbl, recs in arms:
        payload["arms"].append({"label": lbl, "n_records": len(recs)})
    header = f"{'question set':34s}" + "".join(f"{l[:22]:>24s}" for l in labels)
    lines.append(header)
    for ds in datasets:
        row = f"{ds[:-5][:34]:34s}"
        for lbl in labels:
            recs = by_label[lbl]
            ks = [k for k in recs if k[0] == ds]
            if not ks:
                row += f"{'—':>24s}"
                continue
            n = len(ks)
            c = sum(1 for k in ks if recs[k].get("correct"))
            lo, hi = wilson_ci(c, n)
            row += f"{c / n:.3f} [{lo:.3f}-{hi:.3f}]".rjust(24)
            payload["marginal"].setdefault(ds, {})[lbl] = {
                "n": n, "correct": c, "accuracy": c / n, "ci_lo": lo, "ci_hi": hi,
            }
        lines.append(row)
    lines.append("")
    lines.append("Reading note: these intervals are marginal. Two of them overlapping does")
    lines.append("not mean the two arms are indistinguishable: all arms answered the same")
    lines.append("questions, so the comparison to read is the paired one below.")
    lines.append("")

    # --- paired ------------------------------------------------------------
    if reference:
        if reference not in by_label:
            raise SystemExit(f"--reference {reference!r} is not one of: {', '.join(labels)}")
        pairs = [(reference, l) for l in labels if l != reference]
        family = f"every arm against {reference}"
    else:
        pairs = list(itertools.combinations(labels, 2))
        family = "all pairs of arms"
    payload["family"] = family

    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        for a_lbl, b_lbl in pairs:
            A, B = by_label[a_lbl], by_label[b_lbl]
            ks = sorted(k for k in A if k[0] == ds and k in B)
            if not ks:
                continue
            a_ok = [bool(A[k].get("correct")) for k in ks]
            b_ok = [bool(B[k].get("correct")) for k in ks]
            res = compare_binary(a_ok, b_ok)
            res.update({"dataset": ds, "a": a_lbl, "b": b_lbl, "metric": "accuracy"})
            rows.append(res)

            if is_graded([A[k] for k in ks[:50]]):
                for metric in GRADED_METRICS:
                    a_s = [graded_score(A[k], metric) for k in ks]
                    b_s = [graded_score(B[k], metric) for k in ks]
                    if len(ks) < 10:
                        continue
                    g = paired_bootstrap_diff(a_s, b_s, n_boot=n_boot)
                    g.update({"dataset": ds, "a": a_lbl, "b": b_lbl, "metric": metric,
                              "acc_a": g.pop("mean_a"), "acc_b": g.pop("mean_b"),
                              "b_only_a": None, "c_only_b": None})
                    rows.append(g)

    # Holm within each metric: each metric is its own family of claims.
    families: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        families[r["metric"]].append(r)
    for metric, group in families.items():
        for r, q in zip(group, holm_adjust([r["p"] for r in group])):
            r["p_holm"] = q
            r["family_size"] = len(group)

    lines.append("=" * 100)
    lines.append(f"PAIRED (family: {family}; {len(rows)} comparisons)")
    lines.append("=" * 100)
    lines.append("")
    lines.append("  Delta = A - B on the same questions. b/c = discordant counts (A only / B only).")
    lines.append("  Binary outcomes: Newcombe 95% CI + exact McNemar. Graded: BCa bootstrap + sign-flip.")
    lines.append("  Holm is applied within each metric; family sizes: "
                 + ", ".join(f"{m} n={len(g)}" for m, g in sorted(families.items())) + ".")
    lines.append(f"  Graded p-values are Monte-Carlo and cannot go below 1/{n_boot:g}; they are")
    lines.append("  printed as '<floor' there, and the CI is the inference to read.")
    lines.append("")
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] == ds]
        if not sub:
            continue
        lines.append(f"--- {ds[:-5]} ---")
        lines.append(f"    {'A':26s} {'B':26s} {'metric':10s} {'n':>5} {'A':>7} {'B':>7} "
                     f"{'delta':>8} {'95% CI':>18} {'b/c':>10} {'p':>9} {'p Holm':>9}")
        for r in sub:
            bc = f"{r['b_only_a']}/{r['c_only_b']}" if r["b_only_a"] is not None else "—"
            floored = r["test"] != "mcnemar_exact" and r["p"] <= 1.0 / n_boot + 1e-12
            praw = f"<{1.0 / n_boot:.0e}" if floored else f"{r['p']:.2e}"
            padj = f"<{r['p_holm']:.0e}" if floored else f"{r['p_holm']:.2e}"
            ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
            lines.append(
                f"    {r['a'][:26]:26s} {r['b'][:26]:26s} {r['metric']:10s} {r['n']:5d} "
                f"{r['acc_a']:7.3f} {r['acc_b']:7.3f} {r['delta']:+8.3f} {ci:>18} "
                f"{bc:>10} {praw:>9} {padj:>9} {stars(r['p_holm'])}"
            )
        lines.append("")
    payload["paired"] = rows

    lines.append("=" * 100)
    lines.append("SUMMARY")
    lines.append("=" * 100)
    for metric, group in sorted(families.items()):
        sig = sum(1 for r in group if r["p_holm"] < 0.05)
        lines.append(f"  {metric:10s} {len(group):4d} comparisons, {sig:4d} significant after Holm")

    # Descriptive diagnostic outside the family: accuracy differences with raw
    # p < 0.05 whose marginal CIs overlap, over every pair of arms.
    hidden: List[Dict[str, Any]] = []
    checked = 0
    for ds in datasets:
        for a_lbl, b_lbl in itertools.combinations(labels, 2):
            A, B = by_label[a_lbl], by_label[b_lbl]
            ks = sorted(k for k in A if k[0] == ds and k in B)
            if not ks:
                continue
            checked += 1
            res = compare_binary([bool(A[k].get("correct")) for k in ks],
                                 [bool(B[k].get("correct")) for k in ks])
            if res["p"] >= 0.05:
                continue
            res.update({"dataset": ds, "a": a_lbl, "b": b_lbl, "metric": "accuracy"})
            if _ci_overlap(payload, res):
                hidden.append(res)
    payload["overlap_diagnostic"] = {"pairs_checked": checked, "hidden": hidden}

    lines.append("")
    lines.append(f"  Diagnostic over all {checked} arm pairs x question sets (raw p, outside the")
    lines.append("  declared family; descriptive, not a claim):")
    lines.append(f"    differences with p < 0.05 whose marginal CIs overlap: {len(hidden)}")
    if hidden:
        lines.append("    i.e. real differences that a 'do the error bars overlap?' reading loses:")
        for r in sorted(hidden, key=lambda x: x["p"])[:12]:
            lines.append(f"      {r['dataset'][:-5][:26]:26s} {r['a'][:24]:24s} vs {r['b'][:24]:24s} "
                         f"{r['delta']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  p={r['p']:.1e}")
        if len(hidden) > 12:
            lines.append(f"      ... and {len(hidden) - 12} more (see the JSON)")
    return lines, payload


def _ci_overlap(payload: Dict[str, Any], row: Dict[str, Any]) -> bool:
    m = payload["marginal"].get(row["dataset"], {})
    a, b = m.get(row["a"]), m.get(row["b"])
    if not a or not b:
        return False
    return a["ci_lo"] <= b["ci_hi"] and b["ci_lo"] <= a["ci_hi"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*", metavar="LABEL=GLOB",
                    help="arm label and a glob of its JSONL files; repeatable")
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="use a registered set of arms instead of listing them")
    ap.add_argument("--reference", help="compare every arm against this one (defines the family)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help=f"bootstrap / permutation resamples (default {N_BOOT:,})")
    ap.add_argument("-o", "--out", help="output base path; writes <out>.txt and <out>.json")
    args = ap.parse_args()

    specs: List[Tuple[str, List[str]]] = []
    if args.preset:
        specs.extend(PRESETS[args.preset])
    for spec in args.arms:
        if "=" not in spec:
            raise SystemExit(f"bad arm spec {spec!r}: expected LABEL=GLOB")
        label, pattern = spec.split("=", 1)
        specs.append((label, [pattern]))
    if not specs:
        raise SystemExit("no arms: pass --preset or LABEL=GLOB arguments")

    arms: List[Tuple[str, Dict[Tuple[str, str], dict]]] = []
    missing: List[str] = []
    for label, patterns in specs:
        recs = load_arm(patterns, label=label)
        if not recs:
            missing.append(label)
            print(f"  MISSING  {label}: no records matched {patterns}")
            continue
        print(f"  loaded   {label}: {len(recs)} records")
        arms.append((label, recs))
    if len(arms) < 2:
        raise SystemExit("need at least two arms with records")

    lines, payload = build(arms, args.reference, args.n_boot)
    payload["missing_arms"] = missing
    if missing:
        lines.append("")
        lines.append(f"  Arms not found (excluded, not scored 0): {', '.join(missing)}")

    loaded_sets = {ds for _, recs in arms for ds, _ in recs}
    excluded = sorted(k for k in dataset_conflicts() if k[0] in loaded_sets)
    payload["excluded_conflicting_gold"] = [{"dataset": d, "qid": q} for d, q in excluded]
    if excluded:
        lines.append("")
        lines.append("  Excluded from every arm (listed twice in the source dataset with "
                     "different gold answers):")
        for d, q in excluded:
            lines.append(f"    {d}  qid={q}")

    text = "\n".join(lines)
    if args.out:
        base = args.out[:-4] if args.out.endswith((".txt", ".json")) else args.out
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        with open(base + ".txt", "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nReport written to: {base}.txt\nJSON summary written to: {base}.json")
    else:
        print(text)


if __name__ == "__main__":
    main()

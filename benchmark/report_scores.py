"""Every arm's score on every question set, with its marginal 95% interval.

One row per (arm, question set, metric) for every arm in
report_paired_stats.PRESETS. Records go through `load_arm`, so open-ended
answers are rescored with the current scorer.

  True/False, MCQ    accuracy                      Wilson interval
  open-ended         jaccard (primary), f1,        BCa bootstrap of the mean
                     precision, recall, overlap    (N_BOOT resamples)
                     exact_match                   Wilson interval

An unscorable answer counts as wrong / 0 and stays in the denominator. An arm
missing questions of a set is `partial`; one with no record is `not run`.
Differences between arms are tested in the paired reports (paired.*, rq/).

    poetry run python -m benchmark.report_scores                    # both benchmarks
    poetry run python -m benchmark.report_scores --benchmark biohopr

Writes benchmark/results/<benchmark>/stats/scores.{csv,txt}.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from benchmark.bench_utils import question_id, wilson_ci
    from benchmark.paired_stats import N_BOOT, mean_ci_bca
    from benchmark.report_paired_stats import PRESETS, graded_score, load_arm
except ImportError:  # running as a script from inside benchmark/
    from bench_utils import question_id, wilson_ci
    from paired_stats import N_BOOT, mean_ci_bca
    from report_paired_stats import PRESETS, graded_score, load_arm

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (file, label) in table order.
SETS = {
    "alzkb": [
        ("True_or_False_1hop.json", "TF 1-hop"),
        ("True_or_False_2hop.json", "TF 2-hop"),
        ("True_or_False_2hop_atleastone.json", "TF 2-hop disambiguated"),
        ("MCQ_1hop.json", "MCQ 1-hop"),
        ("MCQ_2hop.json", "MCQ 2-hop"),
        ("OpenEnded_1hop.json", "OE 1-hop"),
        ("OpenEnded_2hop.json", "OE 2-hop"),
    ],
    "biohopr": [("BioHopR_2hop.json", "BioHopR 2-hop")],
}
BINARY_METRICS = ["accuracy"]
SET_METRICS = ["jaccard", "exact_match", "f1", "precision", "recall", "overlap"]
SYSTEM_LABEL = {"vanilla": "LLM only", "graphcypher": "GraphCypherQAChain",
                "nexa-direct": "NEXA Direct", "nexa-flex": "NEXA Flex"}
BACKBONE_LABEL = {"gpt-4o-mini": "GPT-4o-mini", "gpt-5-mini": "GPT-5-mini", "qwen3.8-27b": "Qwen3.8-27B"}


def dataset_size(benchmark: str, dataset: str) -> int:
    """Number of distinct questions in a source question set."""
    with open(os.path.join(_REPO, "benchmark", "datasets", benchmark, dataset), encoding="utf-8") as fh:
        return len({question_id(it["question"]) for it in json.load(fh)})


def is_set_type(records: Sequence[dict]) -> bool:
    return any(r.get("type") in ("open_ended", "open_ended_list") for r in records)


def _bca(scores: List[float]) -> Tuple[float, float]:
    return mean_ci_bca(scores, n_boot=N_BOOT)


def build(benchmark: str) -> List[Dict[str, Any]]:
    """Score rows of one benchmark; BCa intervals are computed in parallel processes."""
    rows: List[Dict[str, Any]] = []
    jobs: List[Tuple[int, List[float]]] = []
    for label, globs in PRESETS[benchmark]:
        system, backbone = label.split("_", 1)
        recs = load_arm(globs, label=label) or {}
        for dataset, set_label in SETS[benchmark]:
            size = dataset_size(benchmark, dataset)
            keys = sorted(k for k in recs if k[0] == dataset)
            base = {"benchmark": benchmark, "question_set": set_label, "dataset": dataset,
                    "system": SYSTEM_LABEL[system], "backbone": BACKBONE_LABEL[backbone], "arm": label,
                    "set_size": size, "n": len(keys)}
            if not keys:
                rows.append({**base, "metric": "", "status": "not run"})
                continue
            status = "complete" if len(keys) == size else "partial"
            sub = [recs[k] for k in keys]
            unscored = sum(1 for r in sub if r.get("error") or r.get("model_answer") == "NO_RESPONSE")
            base = {**base, "status": status, "n_failed": unscored}
            if not is_set_type(sub):
                c = sum(1 for r in sub if r.get("correct"))
                lo, hi = wilson_ci(c, len(sub))
                rows.append({**base, "metric": "accuracy", "primary": True, "value_pct": 100 * c / len(sub),
                             "ci95_lo_pct": 100 * lo, "ci95_hi_pct": 100 * hi, "ci_method": "wilson"})
                continue
            for metric in SET_METRICS:
                row = {**base, "metric": metric, "primary": metric == "jaccard"}
                if metric == "exact_match":
                    c = sum(1 for r in sub if r.get("correct"))
                    lo, hi = wilson_ci(c, len(sub))
                    row.update(value_pct=100 * c / len(sub), ci95_lo_pct=100 * lo, ci95_hi_pct=100 * hi,
                               ci_method="wilson")
                else:
                    scores = [graded_score(r, metric) for r in sub]
                    row.update(value_pct=100 * sum(scores) / len(scores), ci_method="bca_bootstrap_mean")
                    jobs.append((len(rows), scores))
                rows.append(row)
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1) or 1) as pool:
        for (i, _), (lo, hi) in zip(jobs, pool.map(_bca, [s for _, s in jobs])):
            rows[i].update(ci95_lo_pct=100 * lo, ci95_hi_pct=100 * hi)
    return rows


FIELDS = ["benchmark", "question_set", "dataset", "system", "backbone", "arm", "metric", "primary",
          "status", "n", "set_size", "n_failed", "value_pct", "ci95_lo_pct", "ci95_hi_pct", "ci_method"]


def write(benchmark: str, rows: List[Dict[str, Any]]) -> str:
    """Write stats/scores.csv and scores.txt; return their path without extension."""
    out = os.path.join(_REPO, "benchmark", "results", benchmark, "stats", "scores")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    def cell(r: Optional[Dict[str, Any]]) -> str:
        if r is None or "value_pct" not in r:
            return "—"
        return f"{r['value_pct']:5.1f} [{r['ci95_lo_pct']:5.1f}-{r['ci95_hi_pct']:5.1f}]"

    by_set: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_set[r["question_set"]].append(r)
    lines = [f"{benchmark}: every arm on every question set, in %, with marginal 95% intervals",
             "accuracy and exact match: Wilson; jaccard, f1, precision, recall, overlap: BCa bootstrap "
             f"of the mean ({N_BOOT:,} resamples). Primary metric: accuracy (TF/MCQ), jaccard (open-ended).",
             "Open-ended answers rescored with the current scorer (names compared up to typography).",
             "failed = questions with no usable answer (error or no response), counted wrong / 0.", ""]
    for _, set_label in SETS[benchmark]:
        set_rows = by_set[set_label]
        metrics = [m for m in BINARY_METRICS + SET_METRICS if any(r.get("metric") == m for r in set_rows)]
        size = set_rows[0]["set_size"]
        lines.append(f"=== {set_label} (n = {size}) ===")
        lines.append(f"{'system':20s} {'backbone':12s} {'status':9s} {'n':>5s} {'failed':>6s}  "
                     + "  ".join(f"{m:^21s}" for m in metrics))
        for arm in dict.fromkeys(r["arm"] for r in set_rows):
            ar = {r.get("metric"): r for r in set_rows if r["arm"] == arm}
            first = next(iter(ar.values()))
            lines.append(f"{first['system']:20s} {first['backbone']:12s} {first['status']:9s} {first['n']:5d} "
                         f"{first.get('n_failed', 0):6d}  " + "  ".join(f"{cell(ar.get(m)):21s}" for m in metrics))
        lines.append("")
    with open(out + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", choices=["alzkb", "biohopr", "all"], default="all")
    args = parser.parse_args()
    for benchmark in (["alzkb", "biohopr"] if args.benchmark == "all" else [args.benchmark]):
        out = write(benchmark, build(benchmark))
        print(f"wrote {out}.csv / .txt")


if __name__ == "__main__":
    main()

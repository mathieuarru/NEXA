"""AlzKB results table for the paper, computed from the run records.

Each cell (system, backbone, question set) carries a marginal 95% interval:
a Wilson interval for accuracy and exact match, a BCa bootstrap of the mean for
graded open-ended scores (overlap, Jaccard). Comparisons between systems belong
to the paired analysis (benchmark/report_paired_stats.py), not to these
intervals.

Cells with no per-question records (published ESCARGOT results) come from
benchmark/published_results.py and are marked 'external' by `provenance()`.

Usage:
    poetry run python figures/data.py   # print the table, write stats/cells.{csv,json}
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmark.bench_utils import wilson_ci  # noqa: E402
from benchmark.published_results import ESCARGOT_CYPHER_ONLY, ESCARGOT_PUBLISHED  # noqa: E402
from benchmark.paired_stats import mean_ci_bca  # noqa: E402
from benchmark.report_paired_stats import PRESETS, load_arm  # noqa: E402

# (system_id, display label)
SYSTEMS = [
    ("llm_only", "LLM Only"),
    ("graphcypher", "GraphCypherQAChain"),
    ("escargot_nofb", "ESCARGOT (Cypher-only ablation)"),
    ("escargot", "ESCARGOT (published)"),
    ("nexa_direct", "NEXA Direct"),
    ("nexa_flex", "NEXA Flex"),
]

MODELS = ["gpt-4o-mini", "gpt-5-mini", "qwen3.8:27b-q8_0"]
MODEL_LABEL = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5-mini": "GPT-5-mini",
    "qwen3.8:27b-q8_0": "Qwen3.8-27B",
}

# (dataset_id, display label, n, metric)
DATASETS = [
    ("tf_1hop", "True/False\n1-hop", 560, "accuracy"),
    ("tf_2hop", "True/False\n2-hop", 540, "accuracy"),
    ("mcq_1hop", "MCQ\n1-hop", 498, "accuracy"),
    ("mcq_2hop", "MCQ\n2-hop", 419, "accuracy"),
    ("oe_1hop", "Open-ended\n1-hop", 508, "overlap"),
    ("oe_2hop", "Open-ended\n2-hop", 450, "overlap"),
]

DATASET_FILE = {
    "tf_1hop": "True_or_False_1hop.json",
    "tf_2hop": "True_or_False_2hop.json",
    "mcq_1hop": "MCQ_1hop.json",
    "mcq_2hop": "MCQ_2hop.json",
    "oe_1hop": "OpenEnded_1hop.json",
    "oe_2hop": "OpenEnded_2hop.json",
}

# (system_id, model) -> the arm label in PRESETS["alzkb"].
ARM_OF = {
    ("llm_only", "gpt-4o-mini"): "vanilla_gpt-4o-mini",
    ("llm_only", "gpt-5-mini"): "vanilla_gpt-5-mini",
    ("llm_only", "qwen3.8:27b-q8_0"): "vanilla_qwen3.8-27b",
    ("graphcypher", "gpt-4o-mini"): "graphcypher_gpt-4o-mini",
    ("graphcypher", "gpt-5-mini"): "graphcypher_gpt-5-mini",
    ("graphcypher", "qwen3.8:27b-q8_0"): "graphcypher_qwen3.8-27b",
    ("nexa_direct", "gpt-4o-mini"): "nexa-direct_gpt-4o-mini",
    ("nexa_direct", "gpt-5-mini"): "nexa-direct_gpt-5-mini",
    ("nexa_flex", "gpt-5-mini"): "nexa-flex_gpt-5-mini",
    ("nexa_direct", "qwen3.8:27b-q8_0"): "nexa-direct_qwen3.8-27b",
    ("nexa_flex", "qwen3.8:27b-q8_0"): "nexa-flex_qwen3.8-27b",
}

# External cells: ESCARGOT aggregates with no per-question records, GPT-4o-mini
# only. Shown for context, never paired. Keyed by model name, not position.
_EXT4O = "gpt-4o-mini"
_DID = {v: k for k, v in DATASET_FILE.items()}


def _external(metrics):
    """{dataset_id: {system_id: {model: value}}} from benchmark/published_results."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sid, table in (("escargot", ESCARGOT_PUBLISHED), ("escargot_nofb", ESCARGOT_CYPHER_ONLY)):
        for metric in metrics:
            for dfile, value in table.get(metric, {}).items():
                out.setdefault(_DID[dfile], {}).setdefault(sid, {})[_EXT4O] = value
    return out


# One table per metric, so a panel never borrows another metric's value. The
# published ESCARGOT results give only accuracy and overlap.
EXTERNAL = _external(("accuracy", "overlap"))
EXTERNAL_JACCARD = _external(("jaccard",))
EXTERNAL_EXACT = _external(("exact_match",))

# Metrics the open-ended panels can use (see use_oe_metric); overlap by default.
OE_METRICS = ("overlap", "jaccard", "exact_match")


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _cell(records, dataset_file: str, metric: str, common=None):
    """(score %, (lo, hi) %, n) for one arm on one question set.

    `common` restricts the cell to the questions every arm has answered.
    """
    keys = [k for k in records if k[0] == dataset_file and (common is None or k in common)]
    if not keys:
        return None, None, 0
    if metric in ("accuracy", "exact_match"):
        n = len(keys)
        c = sum(1 for k in keys if records[k].get("correct"))
        lo, hi = wilson_ci(c, n)
        return 100.0 * c / n, (100.0 * lo, 100.0 * hi), n
    # A record without `metrics` (unparseable answer) scores 0 and stays in the
    # denominator, as in the reported values.
    scores = []
    for k in keys:
        v = (records[k].get("metrics") or {}).get(metric)
        scores.append(float(v) if isinstance(v, (int, float)) else 0.0)
    lo, hi = mean_ci_bca(scores)
    return 100.0 * sum(scores) / len(scores), (100.0 * lo, 100.0 * hi), len(scores)


def _build():
    """(values, intervals, provenance, n) for every cell, keyed by question set then system."""
    arms = {label: load_arm(globs, label=label) for label, globs in PRESETS["alzkb"]}

    # An empty arm from ARM_OF is a path or loader error; without this check the
    # figure would silently lose those bars.
    empty = sorted({arm for arm in ARM_OF.values() if not arms.get(arm)})
    if empty:
        raise RuntimeError(
            "no records loaded for arm(s) " + ", ".join(empty) +
            "; these are declared in ARM_OF, so this is a path/loader failure, "
            "not a missing run. Check benchmark/report_paired_stats.PRESETS."
        )

    results: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    cis: Dict[str, Dict[str, Tuple[Optional[Tuple[float, float]], ...]]] = {}
    prov: Dict[str, Dict[str, Tuple[Optional[str], Optional[str]]]] = {}
    ns: Dict[str, Dict[str, Tuple[int, ...]]] = {}

    measured = [arms[a] for a in set(ARM_OF.values()) if arms.get(a)]
    for did, _, _, metric in DATASETS:
        dfile = DATASET_FILE[did]
        common = set.intersection(*({k for k in recs if k[0] == dfile} for recs in measured))
        results[did], cis[did], prov[did], ns[did] = {}, {}, {}, {}
        for sid, _ in SYSTEMS:
            vals, intervals, sources, counts = [], [], [], []
            for model in MODELS:
                arm = ARM_OF.get((sid, model))
                recs = arms.get(arm) if arm else None
                if recs:
                    v, ci, n = _cell(recs, dfile, metric, common)
                    vals.append(v)
                    intervals.append(ci)
                    counts.append(n)
                    sources.append("measured" if v is not None else None)
                    continue
                table = {"jaccard": EXTERNAL_JACCARD, "exact_match": EXTERNAL_EXACT}.get(metric, EXTERNAL)
                v = table.get(did, {}).get(sid, {}).get(model)
                vals.append(v)
                intervals.append(None)
                counts.append(0)
                sources.append("external" if v is not None else None)
            results[did][sid] = tuple(vals)
            cis[did][sid] = tuple(intervals)
            prov[did][sid] = tuple(sources)
            ns[did][sid] = tuple(counts)
    return results, cis, prov, ns


RESULTS, CIS, PROVENANCE, NS = _build()


def _sync_dataset_n() -> None:
    """Set each question set's n in DATASETS to the number of questions scored.

    Every measured arm is scored on the same questions; a disagreement raises.
    """
    for i, (did, label, n, m) in enumerate(DATASETS):
        seen = {c for counts in NS[did].values() for c in counts if c}
        if len(seen) > 1:
            raise RuntimeError(f"arms disagree on n for {did}: {sorted(seen)}")
        if seen:
            DATASETS[i] = (did, label, seen.pop(), m)


_sync_dataset_n()


def use_oe_metric(metric: str) -> None:
    """Rescore the open-ended panels with `metric`.

    The module's tables are updated in place, because plot_alzkb_results
    imports them by name.
    """
    if metric not in OE_METRICS:
        raise ValueError(f"open-ended metric must be one of {OE_METRICS}, not {metric!r}")
    for i, (did, label, n, m) in enumerate(DATASETS):
        if m != "accuracy":
            DATASETS[i] = (did, label, n, metric)
    results, cis, prov, ns = _build()
    for mine, fresh in ((RESULTS, results), (CIS, cis), (PROVENANCE, prov), (NS, ns)):
        mine.clear()
        mine.update(fresh)
    _sync_dataset_n()


# ---------------------------------------------------------------------------
# Accessors used by plot_alzkb_results.py
# ---------------------------------------------------------------------------

def value(dataset_id: str, system_id: str, model: str):
    """Score for one cell, or None when that run does not exist."""
    return RESULTS[dataset_id][system_id][MODELS.index(model)]


def ci(dataset_id: str, system_id: str, model: str):
    """Marginal 95% interval (lo, hi) in %, or None for a cell without records.

    Overlapping intervals are not evidence of no difference: comparisons are
    in the paired analysis.
    """
    return CIS[dataset_id][system_id][MODELS.index(model)]


def provenance(dataset_id: str, system_id: str, model: str):
    """'measured' (computed here from JSONL), 'external' (no records), or None."""
    return PROVENANCE[dataset_id][system_id][MODELS.index(model)]


def _report() -> None:
    missing = [
        (did, sid, m)
        for did, _, _, _ in DATASETS
        for sid, _ in SYSTEMS
        for m in MODELS
        if provenance(did, sid, m) == "external"
    ]
    print(f"{'question set':12s} {'system':16s} " + "".join(f"{MODEL_LABEL[m]:>30s}" for m in MODELS))
    for did, _, _, metric in DATASETS:
        for sid, _ in SYSTEMS:
            cells = []
            for m in MODELS:
                v, c, p = value(did, sid, m), ci(did, sid, m), provenance(did, sid, m)
                if v is None:
                    cells.append(f"{'—':>30s}")
                elif c:
                    cells.append(f"{v:5.1f} [{c[0]:5.1f}-{c[1]:5.1f}] {p[:4]}".rjust(30))
                else:
                    cells.append(f"{v:5.1f} {'(no records)':>16s} {p[:4]}".rjust(30))
            print(f"{did:12s} {sid:16s} " + "".join(cells))
    print(f"\n{len(missing)} cell(s) without per-question records (no interval, not pairable):")
    for did, sid, m in missing:
        print(f"   {did:10s} {sid:16s} {m}")


CELLS_BASE = os.path.join(_REPO, "benchmark", "results", "alzkb", "stats", "cells")


def write_cells(base: str = CELLS_BASE) -> str:
    """Write every cell of the table, one row each, to <base>.csv and <base>.json.

    Each row names its interval method and its provenance (measured here or
    taken from a publication). Returns `base`.
    """
    import csv
    from benchmark.report_paired_stats import dataset_conflicts
    rows = []
    for did, label, _, metric in DATASETS:
        for sid, sys_label in SYSTEMS:
            for m in MODELS:
                v = value(did, sid, m)
                if v is None:
                    continue
                c = ci(did, sid, m)
                p = provenance(did, sid, m)
                rows.append({
                    "question_set": DATASET_FILE[did],
                    "system": sys_label,
                    "backbone": MODEL_LABEL[m],
                    "metric": metric,
                    "n": NS[did][sid][MODELS.index(m)] or None,
                    "value_pct": round(v, 4),
                    "ci95_lo_pct": round(c[0], 4) if c else None,
                    "ci95_hi_pct": round(c[1], 4) if c else None,
                    "ci_method": (None if not c else
                                  "wilson" if metric in ("accuracy", "exact_match") else "bca_bootstrap_mean"),
                    "provenance": p,
                })
    with open(base + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    excluded = sorted(k for k in dataset_conflicts() if k[0] in DATASET_FILE.values())
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump({
            "note": ("Marginal 95% intervals, one per cell. Comparisons between systems "
                     "are in paired.* and must not be read off these intervals."),
            "excluded_conflicting_gold": [{"dataset": d, "qid": q} for d, q in excluded],
            "cells": rows,
        }, fh, indent=2, ensure_ascii=False)
    return base


if __name__ == "__main__":
    _report()
    out = write_cells()
    print(f"\ncells written to: {out}.csv / {out}.json")

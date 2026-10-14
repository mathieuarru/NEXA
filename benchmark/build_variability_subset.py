"""Stratified, seeded sub-samples of the AlzKB datasets for the run-to-run variability study.

The runner's --limit takes the first N rows, and the True/False files are
ordered by answer, so subsets are drawn here instead: True/False stratified by
answer, MCQ by correct option, open-ended uniformly at random. Each subset is a
plain benchmark JSON file for the runner's --type/--input;
subset_manifest.json records the seed and the selected qids.

Usage
-----
    poetry run python -m benchmark.build_variability_subset            # n=100, seed=42
    poetry run python -m benchmark.build_variability_subset --n 150 --seed 7

Writes benchmark/datasets/alzkb_variability/<Dataset>_sub.json and subset_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

try:
    from benchmark.bench_utils import question_id
except ImportError:  # executed as a plain script from inside benchmark/
    from bench_utils import question_id


# Dataset file prefix -> runner --type (as in BENCH_TYPES, benchmark.py).
PREFIX_TO_TYPE = {
    "MCQ_": "mcq",
    "True_or_False_": "tf",
    "OpenEnded_": "open_ended",
}


def bench_type_for(basename: str) -> Optional[str]:
    """Runner --type of a dataset file, from its name prefix."""
    for prefix, btype in PREFIX_TO_TYPE.items():
        if basename.startswith(prefix):
            return btype
    return None


def discover_datasets(datasets_dir: str) -> List[str]:
    """Full dataset JSON files of a known type, sorted."""
    out: List[str] = []
    for name in sorted(os.listdir(datasets_dir)):
        if not name.endswith(".json"):
            continue
        if bench_type_for(name):
            out.append(os.path.join(datasets_dir, name))
    return out


def stratum_key(item: Dict[str, Any]) -> str:
    """Categorical key used for stratification (the correct answer, lowercased)."""
    return str(item.get("answer", "")).strip().lower()


def allocate(n: int, sizes: Dict[str, int]) -> Dict[str, int]:
    """Proportional largest-remainder allocation of `n` draws over strata, capped by stratum size."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)  # take everything

    raw = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: min(sizes[k], int(math.floor(x))) for k, x in raw.items()}

    # Distribute the remainder by largest fractional part; deterministic label tiebreak.
    while sum(alloc.values()) < n:
        candidates = [k for k in sizes if alloc[k] < sizes[k]]
        if not candidates:
            break
        candidates.sort(key=lambda k: (-(raw[k] - math.floor(raw[k])), k))
        alloc[candidates[0]] += 1
    return alloc


def sample_indices(items: List[Dict[str, Any]], n: int, seed: int,
                   basename: str, strata_max: int) -> Dict[str, Any]:
    """Pick `n` item indices: stratified by answer when it has at most `strata_max` values, else uniform.

    Returns the indices with the sampling method and per-stratum counts for the manifest.
    """
    # Per-dataset RNG so adding/removing a dataset never shifts another's sample.
    rng = random.Random(f"{seed}:{basename}")

    total = len(items)
    if n >= total:
        return {"indices": list(range(total)), "method": "all", "strata": None}

    by_stratum: Dict[str, List[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_stratum[stratum_key(it)].append(i)

    distinct = len(by_stratum)
    if distinct <= strata_max:
        sizes = {k: len(v) for k, v in by_stratum.items()}
        alloc = allocate(n, sizes)
        chosen: List[int] = []
        strata_counts: Dict[str, int] = {}
        for k in sorted(by_stratum):
            c = alloc.get(k, 0)
            if c <= 0:
                continue
            chosen.extend(rng.sample(by_stratum[k], c))
            strata_counts[k] = c
        chosen.sort()
        return {"indices": chosen, "method": "stratified", "strata": strata_counts}

    # Too many distinct answers to stratify meaningfully (e.g. OpenEnded).
    chosen = sorted(rng.sample(range(total), n))
    return {"indices": chosen, "method": "random", "strata": None}


def build(datasets_dir: str, out_dir: str, n: int, seed: int, strata_max: int) -> Dict[str, Any]:
    """Write every subset and the manifest; return the manifest."""
    os.makedirs(out_dir, exist_ok=True)
    files = discover_datasets(datasets_dir)
    if not files:
        raise SystemExit(f"No benchmark datasets found under {datasets_dir!r}.")

    manifest: Dict[str, Any] = {
        "seed": seed,
        "n_requested": n,
        "strata_max": strata_max,
        "datasets": {},
    }

    print(f"Building variability subsets  (n={n}, seed={seed})")
    print(f"  datasets dir : {datasets_dir}")
    print(f"  output dir   : {out_dir}\n")
    print(f"  {'dataset':40s} {'type':11s} {'n':>4}  method       strata")
    print(f"  {'-'*40} {'-'*11} {'-'*4}  {'-'*11}  {'-'*24}")

    for path in files:
        basename = os.path.basename(path)
        base = os.path.splitext(basename)[0]
        btype = bench_type_for(basename)
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        res = sample_indices(items, n, seed, basename, strata_max)
        chosen_items = [items[i] for i in res["indices"]]
        qids = [question_id(it["question"]) for it in chosen_items]

        out_name = f"{base}_sub.json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chosen_items, f, ensure_ascii=False, indent=2)

        strata_str = (
            ", ".join(f"{k}:{v}" for k, v in sorted(res["strata"].items()))
            if res["strata"] else "-"
        )
        print(f"  {out_name:40s} {btype:11s} {len(chosen_items):>4}  "
              f"{res['method']:11s}  {strata_str}")

        manifest["datasets"][out_name] = {
            "source": basename,
            "type": btype,
            "n_selected": len(chosen_items),
            "n_in_source": len(items),
            "method": res["method"],
            "strata": res["strata"],
            "qids": qids,
        }

    manifest_path = os.path.join(out_dir, "subset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nManifest written to: {manifest_path}")
    print("Commit benchmark/datasets/alzkb_variability/ to version the exact sample.\n")

    print("Run each subset through the runner, e.g.:")
    for out_name, meta in manifest["datasets"].items():
        print(f"  --type {meta['type']:<10} --input {os.path.join(out_dir, out_name)}")
    return manifest


def main():
    here = os.path.dirname(__file__) or "."
    parser = argparse.ArgumentParser(description="Build stratified variability subsets.")
    parser.add_argument("--datasets-dir", default=os.path.join(here, "datasets", "alzkb"),
                        help="Directory containing the full benchmark JSON files.")
    parser.add_argument("--out-dir", default=os.path.join(here, "datasets", "alzkb_variability"),
                        help="Where to write the subset files + manifest.")
    parser.add_argument("--n", type=int, default=100,
                        help="Questions per dataset (default 100).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible selection (default 42).")
    parser.add_argument("--strata-max", type=int, default=12,
                        help="Max distinct answers to still stratify (default 12).")
    args = parser.parse_args()
    build(args.datasets_dir, args.out_dir, args.n, args.seed, args.strata_max)


if __name__ == "__main__":
    main()

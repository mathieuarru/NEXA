"""Repair AlzKB MCQ 1-hop entry 266 in every arm's results.

In the copy of MCQ_1hop.json the runs used, entry 266 ("Which of the following
diseases presents the symptom Paraparesis, Spastic?", gold 5) carried the text
of the Muscle Rigidity question, which was therefore asked twice (gold 4 and
gold 5) while Paraparesis was never asked. benchmark/datasets/alzkb/MCQ_1hop.json
matches the official ESCARGOT file (EpistasisLab/escargot); this script brings
the results in line with it:

  NEXA runner arms   keep both Muscle Rigidity records
                     -> drop the gold-5 one, add the Paraparesis record
  vanilla / GCQA     keep only the gold-4 record (they skip repeated questions)
                     -> their --resume run appends the Paraparesis record

    poetry run python -m benchmark.fix_mcq1hop_q266 status   # what every arm has / lacks
    poetry run python -m benchmark.fix_mcq1hop_q266 merge    # drop mislabeled records, fold in new ones

`merge` is idempotent. It first copies every file it changes to
archive/q266_backup/ (a local backup) and tags every added record with a
`patch_note`. The runs themselves: benchmark/run_fix_q266.sh.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Dict, List

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# One-question NEXA runs (--out-subdir alzkb_q266/<arm>), and backups of changed files.
FIX_DIR = os.path.join(_REPO, "benchmark", "results", "alzkb_q266")
BACKUP_DIR = os.path.join(_REPO, "archive", "q266_backup")

DATASET = "MCQ_1hop.json"
MUSCLE = "Which of the following diseases presents the symptom Muscle Rigidity?"
PARAPARESIS = "Which of the following diseases presents the symptom Paraparesis, Spastic?"
BOGUS_GOLD = "5"      # the gold that belonged to Paraparesis, Spastic

# arm -> (the JSONL holding its MCQ 1-hop records, whether it is a NEXA runner arm)
ARMS: Dict[str, tuple] = {
    arm: (f"benchmark/results/alzkb/{arm}/{pattern}", arm.startswith("nexa-"))
    for arm, pattern in [
        ("vanilla_gpt-4o-mini", "baseline_vanilla_gpt-4o-mini_2*.jsonl"),        # not the _atleastone_ run
        ("vanilla_gpt-5-mini", "baseline_vanilla_gpt-5-mini_run1_*.jsonl"),
        ("vanilla_qwen3.8-27b", "baseline_vanilla_qwen3.8_27b-q8_0_2*.jsonl"),  # not the _atleastone_ run
        ("graphcypher_gpt-4o-mini", "baseline_graphcypher_*_topk10000_*.jsonl"),
        ("graphcypher_gpt-5-mini", "baseline_graphcypher_*_run1_*.jsonl"),
        ("graphcypher_qwen3.8-27b", "baseline_graphcypher_Alzkb_121_2*.jsonl"),  # file name without the model name
        ("nexa-direct_gpt-4o-mini", "MCQ_1hop__*.jsonl"),
        ("nexa-direct_gpt-5-mini", "MCQ_1hop__*.jsonl"),
        ("nexa-flex_gpt-5-mini", "MCQ_1hop__*.jsonl"),
        ("nexa-direct_qwen3.8-27b", "MCQ_1hop__*.jsonl"),
        ("nexa-flex_qwen3.8-27b", "MCQ_1hop__*.jsonl"),
    ]
}


def _one(pattern: str) -> str:
    hits = sorted(glob.glob(os.path.join(_REPO, pattern)))
    hits = [h for h in hits if not h.endswith(".meta.json")]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one file for {pattern}, found {len(hits)}: {hits}")
    return hits[0]


def _read(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _is_mcq1(rec: dict) -> bool:
    return (rec.get("file") or "").split("__")[0] == DATASET


def _is_bogus(rec: dict) -> bool:
    return _is_mcq1(rec) and (rec.get("question") or "").startswith(MUSCLE) \
        and str(rec.get("expected")).strip() == BOGUS_GOLD


def _has(recs: List[dict], prefix: str, gold: str = None) -> bool:
    return any(_is_mcq1(r) and (r.get("question") or "").startswith(prefix)
               and (gold is None or str(r.get("expected")).strip() == gold) for r in recs)


def _fix_records(arm: str) -> List[dict]:
    recs = []
    for path in sorted(glob.glob(os.path.join(FIX_DIR, arm, "MCQ_1hop__*.jsonl"))):
        recs += [r for r in _read(path) if (r.get("question") or "").startswith(PARAPARESIS)]
    return recs


def status() -> None:
    """Print what each arm holds and what it still needs."""
    print(f"{'arm':26s} {'MCQ1':>5} {'MR gold 4':>10} {'MR gold 5':>10} {'Paraparesis':>12} {'fix run':>8}  next")
    for arm, (pattern, nexa) in ARMS.items():
        try:
            path = _one(pattern)
        except SystemExit as e:
            print(f"{arm:26s} {e}")
            continue
        recs = _read(path)
        n = sum(1 for r in recs if _is_mcq1(r))
        legit, bogus, para = _has(recs, MUSCLE, "4"), any(_is_bogus(r) for r in recs), _has(recs, PARAPARESIS)
        fix = bool(_fix_records(arm)) if nexa else None
        nxt = ("done" if para and not bogus and n == 498 else
               "merge" if bogus or (nexa and fix and not para) else
               "run Paraparesis")
        print(f"{arm:26s} {n:5d} {'yes' if legit else 'NO':>10} {'yes' if bogus else '-':>10} "
              f"{'yes' if para else '-':>12} {('yes' if fix else '-') if nexa else 'n/a':>8}  {nxt}")


def merge() -> None:
    """Drop the mislabeled gold-5 record and add the Paraparesis record, backing up each changed file."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for arm, (pattern, nexa) in ARMS.items():
        path = _one(pattern)
        recs = _read(path)
        kept = [r for r in recs if not _is_bogus(r)]
        added = []
        if nexa and not _has(kept, PARAPARESIS):
            for r in _fix_records(arm):
                r["patch_note"] = ("MCQ 1-hop entry 266 restored to the official ESCARGOT question; "
                                   "this record was run separately and merged by fix_mcq1hop_q266.py")
                added.append(r)
        if len(kept) == len(recs) and not added:
            continue
        backup = os.path.join(BACKUP_DIR, arm, os.path.basename(path))
        if not os.path.exists(backup):                 # keep the original file, not a later state
            os.makedirs(os.path.dirname(backup), exist_ok=True)
            shutil.copy2(path, backup)
        with open(path, "w", encoding="utf-8") as fh:
            for r in kept + added:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {arm:26s} removed {len(recs) - len(kept)} mislabeled, added {len(added)} Paraparesis "
              f"-> {os.path.relpath(path, _REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("status", "merge"))
    args = ap.parse_args()
    if args.action == "merge":
        merge()
        print()
    status()


if __name__ == "__main__":
    sys.exit(main())

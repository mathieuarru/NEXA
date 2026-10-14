"""Shared helpers for the benchmark runner, the baselines and the scorers.

One definition of record identity, answer parsing, set metrics and basic
statistics, so that runs and reports cannot disagree on them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import unicodedata
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from scipy import stats

METRIC_NAMES = ["exact_match", "precision", "recall", "f1", "jaccard", "overlap"]


# ---------------------------------------------------------------------------
# Shared answer schema
# ---------------------------------------------------------------------------
# Shared with the baselines so every system gets the identical output schema:
# LangChain sends the docstring and Field descriptions to the LLM.

class OpenEndedListAnswer(BaseModel):
    """Open-ended answers as a real list, for KGs whose entity names contain commas.
    """
    thinking: str = Field(description="A detailed explanation of the reasoning used to arrive at the answer, based on the provided knowledge graph data.")
    answer: List[str] = Field(
        description=(
            "The entities that answer the question, ONE ENTITY PER LIST ITEM, copied "
            "verbatim from the knowledge graph results. Never merge several entities "
            "into one item and never split one entity across items: many biomedical "
            "names contain commas (e.g. 'dermatitis herpetiformis, familial') and must "
            "stay whole inside a single item. Example: "
            '["cluster headache, familial", "Kaposi\'s sarcoma", "autism"]'
        )
    )

_HOPS_RE = re.compile(r"(\d+)\s*hop", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Filesystem / JSONL
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield the records of a JSONL file, skipping blank and malformed lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """All records of a JSONL file; [] if the file does not exist."""
    if not os.path.isfile(path):
        return []
    return list(iter_jsonl(path))


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Record identity
# ---------------------------------------------------------------------------

def question_id(question: str) -> str:
    """Stable short id of a question text (SHA-1 prefix), used to pair records across runs."""
    return hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:12]


def hops_from_name(name: str) -> Optional[int]:
    """Hop count in a dataset file name ("MCQ_2hop.json" -> 2), or None."""
    m = _HOPS_RE.search(name or "")
    return int(m.group(1)) if m else None


def git_commit_short() -> Optional[str]:
    """Short hash of the current git commit, or None outside a repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Answer normalization + set metrics (open-ended)
# ---------------------------------------------------------------------------

def entity_list_to_set(raw: Any) -> Set[str]:
    """Entity set from a list-native answer (BioHopR); never splits on commas.

    Many BioHopR names contain a comma ("cluster headache, familial"). Names are
    kept verbatim: no case or Unicode folding, no whitespace stripping (raw
    BioHopR encodes PrimeKG's "(disease)" marker as a trailing space, resolved
    by the dataset builder). Scoring normalises names separately, in
    :func:`entity_key`. A bare string is one entity.
    """
    if raw is None:
        return set()
    items = raw if isinstance(raw, (list, tuple, set)) else [raw]
    return {text for text in (str(item) for item in items) if text}


def csv_to_set(raw: Any) -> Set[str]:
    """Split a comma-separated string into a set of trimmed tokens (case kept).

    For the AlzKB datasets, whose gold is a CSV string: a name containing a
    comma is split, but identically on the gold and prediction sides. Never use
    it on a list-native gold; use :func:`entity_list_to_set` there.
    """
    if not raw:
        return set()
    if not isinstance(raw, str):
        raw = str(raw)
    return {t.strip() for t in raw.split(",") if t and t.strip()}


def list_to_set(raw: Any) -> Set[str]:
    """Trimmed strings of a list/tuple/set (case kept); a string goes through :func:`csv_to_set`."""
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(t).strip() for t in raw if str(t).strip()}
    return csv_to_set(raw)


# Typographic variants mapped to the ASCII form the graphs store (quotes, primes,
# dashes); invisible characters (soft hyphen, zero-width spaces) are removed.
_TYPOGRAPHY = {ord(c): "'" for c in "\u2018\u2019\u201a\u201b\u2032\u00b4\u0060"}
_TYPOGRAPHY.update({ord(c): '"' for c in "\u201c\u201d\u201e\u201f\u2033"})
_TYPOGRAPHY.update({ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"})
_TYPOGRAPHY.update({ord(c): None for c in "\u00ad\u200b\u200c\u200d\u2060\ufeff"})
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_RE = re.compile(r"[\s.]+$")


def entity_key(name: str) -> str:
    """Normalised form under which two entity names count as the same answer.

    Only typography is normalised: Unicode NFKC ("K⁺" -> "K+"), curly quotes
    and dashes to ASCII, invisible characters removed, whitespace collapsed, a
    trailing period dropped, then case folded. Any other spelling difference
    ("Hodgkins" vs "Hodgkin's", a hyphen for a space) stays a different answer,
    and other punctuation is kept (PrimeKG holds both "Carney complex" and
    "Carney complex,"). Folding merges two gold names of one question in 2 of
    958 AlzKB open-ended golds and in no BioHopR gold.
    """
    text = unicodedata.normalize("NFKC", str(name)).translate(_TYPOGRAPHY)
    text = _TRAILING_RE.sub("", _WHITESPACE_RE.sub(" ", text).strip())
    return text.casefold()


def same_entities(expected: Set[str], predicted: Set[str]) -> bool:
    """Exact-set match under :func:`entity_key`."""
    return {k for k in map(entity_key, expected) if k} == {k for k in map(entity_key, predicted) if k}


def gold_covered(expected: Set[str], predicted: Set[str]) -> bool:
    """Every gold entity was predicted, whatever else came with it, under :func:`entity_key`."""
    return {k for k in map(entity_key, expected) if k} <= {k for k in map(entity_key, predicted) if k}


def compute_set_metrics(expected: Set[str], predicted: Set[str]) -> Dict[str, float]:
    """Set metrics of a predicted entity set against the gold, under :func:`entity_key`."""
    expected = {k for k in map(entity_key, expected) if k}
    predicted = {k for k in map(entity_key, predicted) if k}
    if not expected and not predicted:
        return {name: 1.0 for name in METRIC_NAMES}

    inter = expected & predicted
    union = expected | predicted
    precision = len(inter) / len(predicted) if predicted else 0.0
    recall = len(inter) / len(expected) if expected else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    jaccard = len(inter) / len(union) if union else 0.0
    overlap = len(inter) / min(len(expected), len(predicted)) if expected and predicted else 0.0
    exact = 1.0 if predicted == expected else 0.0

    return {
        "exact_match": exact,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "overlap": overlap,
    }


def answer_sets(rec: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """(gold, predicted) entity sets of one open-ended record, parsed per its type.

    `open_ended` golds are CSV strings; `open_ended_list` golds are JSON arrays,
    where splitting on commas would shred any name containing one.
    """
    if rec.get("type") == "open_ended_list":
        return (entity_list_to_set(rec.get("expected") or []),
                entity_list_to_set(rec.get("model_answer") or []))
    return (csv_to_set(rec.get("expected", "")),
            list_to_set(rec.get("model_answer", "")))


def rescore_open_ended(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute an open-ended record's `metrics` and `correct` with the current scorer.

    Stored scores reflect the scorer at run time; this recomputes them from the
    stored answers. A record without `metrics` (a failed baseline answer,
    scored 0) is left unchanged.
    """
    if rec.get("type") not in ("open_ended", "open_ended_list") or not isinstance(rec.get("metrics"), dict):
        return rec
    gold, pred = answer_sets(rec)
    metrics = compute_set_metrics(gold, pred)
    if "gold_covered" in rec["metrics"]:
        metrics["gold_covered"] = 1.0 if gold_covered(gold, pred) else 0.0
    rec["metrics"] = {**rec["metrics"], **metrics}
    rec["correct"] = metrics["exact_match"] == 1.0
    return rec


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on discordant pair counts.

    b = questions correct in A only, c = correct in B only. Computed as the
    exact two-sided binomial test of min(b, c) out of b + c at p = 0.5 (SciPy,
    which stays exact and finite for large b + c).
    """
    n = b + c
    if n == 0:
        return 1.0
    return float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def t_critical_95(df: int) -> float:
    """Two-sided 95% Student-t critical value, exact for any degrees of freedom."""
    if df < 1:
        return float("nan")
    return float(stats.t.ppf(0.975, df))


def mean_ci95_across_runs(values: List[float]) -> Optional[Tuple[float, float]]:
    """95% CI for the mean of repeated runs: mean +/- t(.975, R-1) * SD / sqrt(R).

    Covers run-to-run (decoding) variability only: the runs replay the same
    questions. Question-sampling uncertainty is `wilson_ci` on a single run;
    a multi-run mean passed to `wilson_ci` would ignore the spread between runs.
    Returns None with fewer than 2 runs.
    """
    n = len(values)
    if n < 2:
        return None
    m = mean(values)
    half = t_critical_95(n - 1) * stdev(values) / math.sqrt(n)
    return (max(0.0, m - half), min(1.0, m + half))


def percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0,100]) over a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def stdev(vals: List[float]) -> float:
    """Sample standard deviation (n - 1 denominator); 0 for fewer than 2 values."""
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

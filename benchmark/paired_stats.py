"""Paired inference for systems evaluated on the same questions.

Overlap of two marginal intervals is the wrong test for paired data: it keeps
the shared per-question difficulty and misses real differences, and the naive
Wald interval on a paired difference can exclude zero when the exact test does
not. One paired estimator per outcome type:

  newcombe_paired_diff_ci  binary outcomes (accuracy, exact match); interval
                           companion to the exact McNemar test.
  paired_bootstrap_diff    graded outcomes in [0, 1] (F1, Jaccard, recall):
                           BCa bootstrap interval and sign-flip permutation p.
  holm_adjust              Holm correction within a declared family.

References
----------
Newcombe RG (1998). Improved confidence intervals for the difference between
binomial proportions based on paired data. Statistics in Medicine 17:2635-2650.
Method 10 (MOVER on Wilson intervals with a correlation correction).

Efron B, Tibshirani RJ (1993). An Introduction to the Bootstrap, ch. 14 (BCa).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

try:
    from benchmark.bench_utils import mcnemar_exact_p, wilson_ci
except ImportError:  # running as a script from inside benchmark/
    from bench_utils import mcnemar_exact_p, wilson_ci


# ---------------------------------------------------------------------------
# Binary outcomes
# ---------------------------------------------------------------------------

def discordance(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Tuple[int, int, int, int]:
    """2x2 counts for two arms over the same, aligned questions.

    Returns (n11, n10, n01, n00): both right, A only, B only, both wrong.
    The middle two are McNemar's b and c.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(f"arms are not aligned: {len(a_correct)} vs {len(b_correct)}")
    n11 = n10 = n01 = n00 = 0
    for a, b in zip(a_correct, b_correct):
        if a and b:
            n11 += 1
        elif a:
            n10 += 1
        elif b:
            n01 += 1
        else:
            n00 += 1
    return n11, n10, n01, n00


def newcombe_paired_diff_ci(
    n11: int, n10: int, n01: int, n00: int, z: float = 1.96
) -> Tuple[float, float]:
    """95% CI for p(A correct) - p(B correct) on paired data (Newcombe method 10).

    Combines the two arms' Wilson intervals (MOVER), corrected by the observed
    phi correlation between the arms. A table with an empty margin uses phi = 0,
    as Newcombe prescribes.
    """
    n = n11 + n10 + n01 + n00
    if n <= 0:
        return (0.0, 0.0)

    p1 = (n11 + n10) / n            # arm A
    p2 = (n11 + n01) / n            # arm B
    d = p1 - p2

    l1, u1 = wilson_ci(n11 + n10, n, z)
    l2, u2 = wilson_ci(n11 + n01, n, z)

    margin = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    phi = ((n11 * n00 - n10 * n01) / math.sqrt(margin)) if margin > 0 else 0.0
    phi = max(-1.0, min(1.0, phi))

    # max(0, .): a large positive phi can make the radicand slightly negative.
    lo_rad = (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    hi_rad = (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2

    lo = d - math.sqrt(max(0.0, lo_rad))
    hi = d + math.sqrt(max(0.0, hi_rad))
    return (max(-1.0, lo), min(1.0, hi))


def _z(level: float) -> float:
    return float(stats.norm.ppf(1 - (1 - level) / 2))


def compare_binary(a_correct: Sequence[bool], b_correct: Sequence[bool],
                   levels: Sequence[float] = ()) -> Dict[str, Any]:
    """Paired comparison of two arms on a binary outcome: Newcombe 95% CI and exact McNemar p.

    `levels` adds intervals at other confidence levels (e.g. 0.90) under `ci_by_level`.
    """
    n11, n10, n01, n00 = discordance(a_correct, b_correct)
    n = n11 + n10 + n01 + n00
    lo, hi = newcombe_paired_diff_ci(n11, n10, n01, n00)
    acc_a = (n11 + n10) / n if n else float("nan")
    acc_b = (n11 + n01) / n if n else float("nan")
    by_level = {lvl: newcombe_paired_diff_ci(n11, n10, n01, n00, z=_z(lvl)) for lvl in levels}
    return {
        "n": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "delta": acc_a - acc_b,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_by_level": by_level,
        "b_only_a": n10,
        "c_only_b": n01,
        "p": mcnemar_exact_p(n10, n01),
        "test": "mcnemar_exact",
    }


# ---------------------------------------------------------------------------
# Graded outcomes
# ---------------------------------------------------------------------------

# Resamples for every bootstrap interval and permutation test: bounds stable to
# 0.1 pp, and a p floor (1/N_BOOT) low enough for Holm in the 54-comparison family.
N_BOOT = 100_000

# Resampling runs in blocks to bound memory. The Generator stream does not depend
# on the block size, so results are bit-identical to one (n_boot x n) draw.
_ELEMENT_BUDGET = 20_000_000


def _resample_means(values: np.ndarray, n_boot: int, rng) -> np.ndarray:
    n = values.size
    block = max(1, _ELEMENT_BUDGET // n)
    out = np.empty(n_boot)
    for start in range(0, n_boot, block):
        k = min(block, n_boot - start)
        out[start:start + k] = values[rng.integers(0, n, size=(k, n))].mean(axis=1)
    return out


def _sign_flip_means(d: np.ndarray, n_boot: int, rng) -> np.ndarray:
    n = d.size
    block = max(1, _ELEMENT_BUDGET // n)
    out = np.empty(n_boot)
    for start in range(0, n_boot, block):
        k = min(block, n_boot - start)
        out[start:start + k] = (rng.choice((-1.0, 1.0), size=(k, n)) * d).mean(axis=1)
    return out


def _bca_interval(boot: np.ndarray, theta: float, x: np.ndarray, alpha: float) -> Tuple[float, float]:
    """BCa interval of a mean from its bootstrap replicates."""
    n_boot, n = boot.size, x.size
    prop = float((boot < theta).mean())
    prop = min(max(prop, 1.0 / (2 * n_boot)), 1.0 - 1.0 / (2 * n_boot))
    z0 = float(stats.norm.ppf(prop))
    # Jackknife of a mean, in closed form: theta_(i) = (S - x_i) / (n - 1).
    jack = (x.sum() - x) / (n - 1)
    centred = jack.mean() - jack
    denom = 6.0 * (float((centred ** 2).sum()) ** 1.5)
    acc = float((centred ** 3).sum()) / denom if denom > 0 else 0.0
    zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - acc * (z0 + zl)))
    a2 = stats.norm.cdf(z0 + (z0 + zu) / (1 - acc * (z0 + zu)))
    lo, hi = (float(v) for v in np.quantile(boot, [a1, a2]))
    return lo, hi


def paired_bootstrap_diff(
    a_scores: Sequence[float],
    b_scores: Sequence[float],
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = 20260921,
    levels: Sequence[float] = (),
) -> Dict[str, Any]:
    """BCa bootstrap CI and sign-flip permutation p for the mean paired difference.

    For per-question scores in [0, 1] (F1, Jaccard, recall). Questions are
    resampled with their pair of scores, so shared difficulty cancels. BCa
    because the differences are strongly skewed (mostly 0, a few large);
    acceleration from the closed-form jackknife of a mean. The p-value flips
    the sign of each difference at random (Monte-Carlo, floor 1 / n_boot), drawn
    after the bootstrap from the same seeded generator. `levels` adds intervals
    from the same replicates under `ci_by_level`.
    """
    d = np.asarray(a_scores, dtype=float) - np.asarray(b_scores, dtype=float)
    n = d.size
    if n == 0:
        return {"n": 0, "mean_a": float("nan"), "mean_b": float("nan"),
                "delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "ci_by_level": {}, "p": 1.0, "test": "paired_bootstrap_bca"}

    theta = float(d.mean())
    rng = np.random.default_rng(seed)
    boot = _resample_means(d, n_boot, rng)

    if np.allclose(d, d[0]):
        # No variation: every resample gives the same value, BCa is undefined.
        lo = hi = theta
        by_level = {lvl: (theta, theta) for lvl in levels}
    else:
        lo, hi = _bca_interval(boot, theta, d, alpha)
        by_level = {lvl: _bca_interval(boot, theta, d, 1 - lvl) for lvl in levels}

    null = _sign_flip_means(d, n_boot, rng)
    p = float((np.abs(null) >= abs(theta) - 1e-15).mean())
    p = max(p, 1.0 / n_boot)

    return {
        "n": n,
        "mean_a": float(np.mean(a_scores)),
        "mean_b": float(np.mean(b_scores)),
        "delta": theta,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_by_level": by_level,
        "p": p,
        "test": "paired_bootstrap_bca",
    }


def mean_ci_bca(
    scores: Sequence[float],
    n_boot: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = 20260921,
) -> Tuple[float, float]:
    """BCa 95% CI for the mean of a per-question score in [0, 1].

    Companion to `wilson_ci` for graded outcomes, which are bounded and skewed
    (most questions at 1.0), so neither a binomial nor a normal interval fits.
    """
    x = np.asarray(scores, dtype=float)
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    theta = float(x.mean())
    if n < 3 or np.allclose(x, x[0]):
        return (theta, theta)
    rng = np.random.default_rng(seed)
    boot = _resample_means(x, n_boot, rng)
    lo, hi = _bca_interval(boot, theta, x, alpha)
    return (max(0.0, lo), min(1.0, hi))


# ---------------------------------------------------------------------------
# Published aggregates: worst case over every question-level pairing
# ---------------------------------------------------------------------------

def mcnemar_bounds_from_aggregates(k_a: int, n: int, pct_b: float,
                                   decimals: int = 1) -> Dict[str, Any]:
    """Range of McNemar p-values compatible with a published accuracy.

    Against a system known only by its accuracy, the discordant counts are
    unknown but constrained: b - c = k_a - k_b, and each is capped by the
    margins. The largest p over every feasible (b, c) holds whatever the
    question-level agreement was. Every count k_b that rounds to `pct_b` at
    `decimals` is tried.
    """
    half = 0.5 * 10 ** (-decimals)
    k_bs = [k for k in range(n + 1) if abs(100.0 * k / n - pct_b) <= half + 1e-9]
    if not k_bs:
        raise ValueError(f"no count out of {n} rounds to {pct_b}%")
    best, worst, worst_bc = 1.0, 0.0, None
    for k_b in k_bs:
        d = k_a - k_b
        for c in range(max(0, -d), min(k_b, n - k_a) + 1):
            b = c + d
            if b < 0 or b > min(k_a, n - k_b):
                continue
            p = mcnemar_exact_p(b, c)
            best = min(best, p)
            if p > worst:
                worst, worst_bc = p, (b, c)
    return {"n": n, "k_a": k_a, "k_b_candidates": k_bs, "p_best": best,
            "p_worst": worst, "worst_b_c": worst_bc}


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------

def holm_adjust(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, in input order.

    Controls the familywise error rate; adjusted values are monotone in the raw ones.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        out[i] = min(1.0, running)
    return out


def stars(p: Optional[float]) -> str:
    """Compact significance marker for tables and figures."""
    if p is None or not (p == p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

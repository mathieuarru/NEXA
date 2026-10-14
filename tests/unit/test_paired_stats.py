"""Paired estimators behind the comparison tables.

Checked by spot values, coverage simulation and agreement with the exact
McNemar test.
"""

import math

import numpy as np
import pytest

from benchmark.bench_utils import mcnemar_exact_p, wilson_ci
from benchmark.paired_stats import (
    compare_binary,
    discordance,
    holm_adjust,
    mean_ci_bca,
    newcombe_paired_diff_ci,
    paired_bootstrap_diff,
    stars,
)


# ---------------------------------------------------------------------------
# discordance
# ---------------------------------------------------------------------------

def test_discordance_counts_the_four_cells():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    assert discordance(a, b) == (1, 2, 1, 1)


def test_discordance_rejects_misaligned_arms():
    with pytest.raises(ValueError):
        discordance([True, False], [True])


# ---------------------------------------------------------------------------
# Newcombe interval
# ---------------------------------------------------------------------------

def test_newcombe_is_antisymmetric_under_swapping_the_arms():
    lo, hi = newcombe_paired_diff_ci(415, 4, 0, 0)
    lo_s, hi_s = newcombe_paired_diff_ci(415, 0, 4, 0)
    assert lo == pytest.approx(-hi_s, abs=1e-12)
    assert hi == pytest.approx(-lo_s, abs=1e-12)


def test_newcombe_contains_the_point_estimate():
    for n11, n10, n01, n00 in [(415, 4, 0, 0), (100, 30, 10, 60), (0, 5, 5, 0), (50, 0, 0, 50)]:
        n = n11 + n10 + n01 + n00
        d = (n10 - n01) / n
        lo, hi = newcombe_paired_diff_ci(n11, n10, n01, n00)
        assert lo <= d <= hi


def test_newcombe_is_degenerate_when_the_arms_never_disagree():
    lo, hi = newcombe_paired_diff_ci(50, 0, 0, 50)
    assert lo <= 0.0 <= hi
    assert hi - lo < 0.1  # perfect agreement is a strong statement, not a wide one


def test_newcombe_fixes_the_wald_inconsistency_on_MCQ_2hop():
    """Newcombe, unlike Wald, includes zero on real MCQ 2-hop counts where McNemar p = 0.125."""
    n11, n10, n01, n00 = 415, 4, 0, 0
    n = n11 + n10 + n01 + n00

    d = (n10 - n01) / n
    se = math.sqrt((n10 + n01) - (n10 - n01) ** 2 / n) / n
    wald_lo = d - 1.96 * se
    assert wald_lo > 0.0                      # Wald excludes zero

    lo, hi = newcombe_paired_diff_ci(n11, n10, n01, n00)
    assert lo < 0.0 < hi                      # the score interval does not
    assert mcnemar_exact_p(n10, n01) > 0.05   # consistent with the exact test


@pytest.mark.parametrize("n11,n10,n01,n00", [
    (2600, 156, 24, 200), (300, 91, 24, 145), (2000, 6, 39, 900),
    (400, 11, 0, 97), (100, 27, 34, 399), (1, 1, 1, 1),
])
def test_newcombe_agrees_with_mcnemar_on_whether_zero_is_excluded(n11, n10, n01, n00):
    """Interval and exact test agree on significance at 0.05 for the counts in the paper."""
    lo, hi = newcombe_paired_diff_ci(n11, n10, n01, n00)
    excludes_zero = lo > 0.0 or hi < 0.0
    assert excludes_zero == (mcnemar_exact_p(n10, n01) < 0.05)


def test_newcombe_covers_the_true_difference_at_the_nominal_rate():
    """Simulated coverage of a known paired difference is near nominal, with bounded width."""
    rng = np.random.default_rng(7)
    n, trials = 120, 1200
    p_a, p_b, rho = 0.80, 0.70, 0.6      # correlated arms, true difference 0.10
    truth = p_a - p_b

    covered = 0
    widths = []
    for _ in range(trials):
        shared = rng.random(n) < rho
        base = rng.random(n)
        a = np.where(shared, base < p_a, rng.random(n) < p_a)
        b = np.where(shared, base < p_b, rng.random(n) < p_b)
        lo, hi = newcombe_paired_diff_ci(*discordance(list(a), list(b)))
        covered += lo <= truth <= hi
        widths.append(hi - lo)

    assert covered / trials >= 0.94
    assert float(np.mean(widths)) < 0.40


# ---------------------------------------------------------------------------
# compare_binary
# ---------------------------------------------------------------------------

def test_compare_binary_reports_the_arms_in_the_order_given():
    a = [True] * 8 + [False] * 2
    b = [True] * 5 + [False] * 5
    res = compare_binary(a, b)
    assert res["n"] == 10
    assert res["acc_a"] == pytest.approx(0.8)
    assert res["acc_b"] == pytest.approx(0.5)
    assert res["delta"] == pytest.approx(0.3)
    assert res["b_only_a"] == 3 and res["c_only_b"] == 0
    assert res["test"] == "mcnemar_exact"


def test_paired_interval_is_tighter_than_the_unpaired_one_when_arms_agree():
    """The paired interval is tighter than the unpaired combination of two Wilson intervals."""
    n11, n10, n01, n00 = 480, 15, 5, 0     # 500 questions, high agreement
    n = n11 + n10 + n01 + n00
    lo, hi = newcombe_paired_diff_ci(n11, n10, n01, n00)

    l1, u1 = wilson_ci(n11 + n10, n)
    l2, u2 = wilson_ci(n11 + n01, n)
    unpaired_lo, unpaired_hi = l1 - u2, u1 - l2

    # Pairing detects a difference the unpaired interval does not.
    assert lo > 0.0
    assert unpaired_lo < 0.0

    # Modest width gain: both marginals are near the ceiling.
    assert (hi - lo) < 0.85 * (unpaired_hi - unpaired_lo)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_paired_bootstrap_finds_a_real_shift():
    rng = np.random.default_rng(3)
    a = np.clip(rng.normal(0.80, 0.10, 400), 0, 1)
    b = np.clip(a - 0.08, 0, 1)
    res = paired_bootstrap_diff(a, b, n_boot=2000)
    assert res["delta"] > 0
    assert res["ci_lo"] > 0
    assert res["p"] < 0.01


def test_paired_bootstrap_does_not_invent_one():
    rng = np.random.default_rng(4)
    a = np.clip(rng.normal(0.70, 0.15, 400), 0, 1)
    b = np.clip(rng.normal(0.70, 0.15, 400), 0, 1)
    res = paired_bootstrap_diff(a, b, n_boot=2000)
    assert res["ci_lo"] < 0 < res["ci_hi"]
    assert res["p"] > 0.05


def test_paired_bootstrap_is_deterministic_for_a_fixed_seed():
    a, b = [0.9, 0.8, 1.0, 0.4] * 30, [0.7, 0.8, 0.9, 0.5] * 30
    first = paired_bootstrap_diff(a, b, n_boot=500)
    second = paired_bootstrap_diff(a, b, n_boot=500)
    assert first == second


def test_paired_bootstrap_p_never_claims_more_than_the_resolution_it_has():
    a, b = [1.0] * 200, [0.0] * 200
    res = paired_bootstrap_diff(a, b, n_boot=1000)
    assert res["p"] == pytest.approx(1.0 / 1000)


def test_paired_bootstrap_handles_identical_arms():
    a = b = [0.5, 0.7, 0.9] * 40
    res = paired_bootstrap_diff(a, b, n_boot=500)
    assert res["delta"] == 0.0
    assert res["ci_lo"] == res["ci_hi"] == 0.0


def test_mean_ci_bca_brackets_the_mean_and_stays_in_range():
    rng = np.random.default_rng(11)
    x = np.clip(rng.beta(9, 1, 500), 0, 1)     # piled up near 1.0, like an overlap score
    lo, hi = mean_ci_bca(x, n_boot=2000)
    assert 0.0 <= lo <= float(x.mean()) <= hi <= 1.0


def test_mean_ci_bca_is_degenerate_on_a_constant_score():
    lo, hi = mean_ci_bca([1.0] * 50, n_boot=500)
    assert lo == hi == 1.0


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------

def test_holm_preserves_input_order():
    p = [0.04, 0.001, 0.30]
    assert holm_adjust(p) == [pytest.approx(0.08), pytest.approx(0.003), pytest.approx(0.30)]


def test_holm_is_monotone_in_the_raw_p():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    adj = holm_adjust(p)
    assert adj == sorted(adj)


def test_holm_never_exceeds_one_and_is_never_smaller_than_the_raw_p():
    p = [0.6, 0.7, 0.8, 0.9]
    adj = holm_adjust(p)
    assert all(a <= 1.0 for a in adj)
    assert all(a >= r - 1e-12 for a, r in zip(adj, p))


def test_holm_is_never_more_conservative_than_bonferroni():
    p = [0.001, 0.01, 0.02, 0.5]
    adj = holm_adjust(p)
    assert all(a <= min(1.0, len(p) * r) + 1e-12 for a, r in zip(adj, p))


def test_holm_on_an_empty_family():
    assert holm_adjust([]) == []


def test_stars_thresholds():
    assert stars(0.0001) == "***"
    assert stars(0.005) == "**"
    assert stars(0.04) == "*"
    assert stars(0.2) == "ns"
    assert stars(None) == ""
    assert stars(float("nan")) == ""


# ---------------------------------------------------------------------------
# Research-question analysis (report_rq.py)
# ---------------------------------------------------------------------------

from benchmark import paired_stats as ps
from benchmark.paired_stats import mcnemar_bounds_from_aggregates


def test_blocked_resampling_is_bit_identical_to_one_block(monkeypatch):
    """Resampling in memory blocks gives bit-identical results."""
    rng = np.random.default_rng(5)
    a = rng.beta(8, 2, 300)
    b = np.clip(a - rng.normal(0.02, 0.05, 300), 0, 1)
    ref = ps.paired_bootstrap_diff(a, b, n_boot=3000, levels=(0.90,))
    ref_mean = ps.mean_ci_bca(a, n_boot=3000)
    monkeypatch.setattr(ps, "_ELEMENT_BUDGET", 300 * 7)       # blocks of 7 resamples
    assert ps.paired_bootstrap_diff(a, b, n_boot=3000, levels=(0.90,)) == ref
    assert ps.mean_ci_bca(a, n_boot=3000) == ref_mean


def test_ninety_percent_interval_is_nested_in_the_ninety_five():
    a = [True] * 80 + [False] * 20
    b = [True] * 70 + [False] * 30
    r = compare_binary(a, b, levels=(0.90,))
    lo90, hi90 = r["ci_by_level"][0.90]
    assert r["ci_lo"] < lo90 <= r["delta"] <= hi90 < r["ci_hi"]

    rng = np.random.default_rng(9)
    x = rng.beta(9, 1, 400)
    y = np.clip(x - 0.03, 0, 1)
    g = paired_bootstrap_diff(x, y, n_boot=4000, levels=(0.90,))
    lo90, hi90 = g["ci_by_level"][0.90]
    assert g["ci_lo"] <= lo90 <= g["delta"] <= hi90 <= g["ci_hi"]


def test_aggregate_bounds_match_brute_force_over_every_pairing():
    """Aggregate-only McNemar bounds equal the extremes over every compatible 2x2 table."""
    for n, k_a, k_b in [(20, 17, 12), (20, 12, 17), (30, 25, 25), (15, 15, 9), (25, 3, 1)]:
        ps_all = []
        for n11 in range(0, min(k_a, k_b) + 1):
            n10, n01 = k_a - n11, k_b - n11
            n00 = n - n11 - n10 - n01
            if min(n10, n01, n00) < 0:
                continue
            ps_all.append(mcnemar_exact_p(n10, n01))
        pct_b = round(100 * k_b / n, 1)
        r = mcnemar_bounds_from_aggregates(k_a, n, pct_b)
        assert k_b in r["k_b_candidates"]
        if r["k_b_candidates"] == [k_b]:
            assert r["p_worst"] == pytest.approx(max(ps_all))
            assert r["p_best"] == pytest.approx(min(ps_all))
        assert r["p_best"] <= r["p_worst"]


def test_aggregate_bounds_consider_every_count_the_rounding_allows():
    # A 1-decimal percentage fits one count on n = 1000, several on n = 2000.
    assert mcnemar_bounds_from_aggregates(900, 1000, 85.3)["k_b_candidates"] == [853]
    assert mcnemar_bounds_from_aggregates(1800, 2000, 85.3)["k_b_candidates"] == [1705, 1706, 1707]
    with pytest.raises(ValueError):
        mcnemar_bounds_from_aggregates(10, 7, 33.3)


def test_holm_with_placeholders_is_an_upper_bound_on_the_final_values():
    """Counting pending comparisons as p = 1 gives Holm values the real results can only lower."""
    rng = np.random.default_rng(2026)
    for _ in range(3000):
        m = int(rng.integers(2, 31))
        k = int(rng.integers(1, m))                        # number pending
        known = list(rng.beta(0.4, 2.0, m - k))
        provisional = holm_adjust(known + [1.0] * k)[: m - k]
        final = holm_adjust(known + list(rng.uniform(0, 1, k) ** rng.uniform(0.2, 4)))[: m - k]
        assert all(f <= p + 1e-12 for f, p in zip(final, provisional))


def test_research_question_family_spans_both_benchmarks():
    """Each research question is one Holm family across AlzKB and BioHopR, recorded in each report."""
    from benchmark.report_rq import apply_holm_across

    def row(p, status="ok"):
        return {"status": status, "p": p}

    results = {"alzkb": {"RQ2": {"rows": [row(0.001), row(0.02), row(0.4), row(None, "not run on this set")]}},
               "biohopr": {"RQ2": {"rows": [row(0.01), row(None, "pending")]}}}
    apply_holm_across(results, "RQ2")

    pooled = holm_adjust([0.001, 0.02, 0.4, 0.01, 1.0])            # the pending one counts as p = 1
    got = [r.get("p_holm") for b in ("alzkb", "biohopr") for r in results[b]["RQ2"]["rows"]]
    assert got == [pooled[0], pooled[1], pooled[2], None, pooled[3], None]
    for b in results:
        fam = results[b]["RQ2"]["family"]
        assert fam["size"] == 5 and fam["pending"] == 1
        assert fam["members"] == {"alzkb": 3, "biohopr": 2}

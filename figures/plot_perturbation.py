"""Perturbed entity names: empty-result correction of NEXA Direct, on AlzKB and BioHopR.

Writes four figures to figures/out/, each with panel a = AlzKB v1.2.1 and
b = BioHopR / PrimeKG v2.1:

  ablation_perturbation_accuracy            grouped bars: score of complete NEXA
  ablation_perturbation_jaccard             Direct and of the three arms without
                                            the empty-result correction or one of
                                            its steps, per perturbation, with 95%
                                            intervals
  ablation_perturbation_difference          forest plots: each arm's paired
  ablation_perturbation_difference_jaccard  difference from complete NEXA Direct,
                                            filled when significant after Holm
                                            correction over the 12 comparisons
                                            of the benchmark

Metrics:

  accuracy  exact answer (MCQ option, or exact set match for open-ended
            questions), read from benchmark/results/*_perturbation/stats/ablation.json;
            Wilson intervals, McNemar / Newcombe for the differences.
  jaccard   BioHopR: Jaccard index. AlzKB: pooled primary metric (accuracy for
            MCQ, Jaccard index for open-ended). Computed here from the run
            records; BCa bootstrap intervals and sign-flip tests, Holm over the
            same 12 comparisons.

The original-name group is the unperturbed control.

Usage:
    poetry run python figures/plot_perturbation.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import style

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmark.ablation_direct import (DATASETS as ABLATION_DATASETS, EXPERIMENTS,  # noqa: E402
                                       arm_dir, load_experiment_arm, load_json)
from benchmark.bench_utils import wilson_ci  # noqa: E402
from benchmark.paired_stats import holm_adjust, mean_ci_bca, paired_bootstrap_diff  # noqa: E402
from benchmark.report_paired_stats import graded_score  # noqa: E402

RESULTS = os.path.join(_REPO, "benchmark", "results")
OUT_DIR = os.path.join(_HERE, "out")

# (experiment key in ablation_direct.EXPERIMENTS, panel title, name of the
# graded metric on that benchmark)
PANELS = [
    ("alzkb-perturbation", "AlzKB v1.2.1", "pooled primary metric"),
    ("biohopr-perturbation", "BioHopR · PrimeKG v2.1", "Jaccard index"),
]
SUBTITLE = "Entity-name perturbations · GPT-4o-mini"

# Kinds of perturbation, in reading order: the unperturbed control first.
PERTURBATIONS = [
    ("original", "Original name"),
    ("case", "Case variant"),
    ("typo", "Typographical error"),
    ("alias", "Full gene name"),
]

# All arms are NEXA Direct: the complete system keeps its blue, the ablated arms
# take three other logo hues (not NEXA Flex's orange).
FULL = ("full", "Complete NEXA Direct", style.LOGO["blue"])
ABLATED = [
    # Same arm and same name as in the ablation figure (plot_ablation.py).
    ("no-empty-loop", "No empty-result correction", style.LOGO["yellow"]),
    ("no-db-matching", "No database value matching", style.LOGO["violet"]),
    ("no-llm-selection", "No LLM candidate selection", style.LOGO["green"]),
]

# (bar stem, difference stem, bar axis, difference axis) per metric
METRICS = {
    "accuracy": ("ablation_perturbation_accuracy", "ablation_perturbation_difference",
                 "Accuracy (%)", "Accuracy difference relative to complete NEXA Direct (pp)"),
    "jaccard": ("ablation_perturbation_jaccard", "ablation_perturbation_difference_jaccard",
                "Score (%)", "Score difference relative to complete NEXA Direct (pp)"),
}

# Bars, as in the results figures.
GROUP_PAD = 0.14
BAR_GAP = 0.0224
VALUE_PAD = 2.0
VALUE_SIZE = 5.4
YMAX = 104

# Forest plot, as in the ablation figure.
ROW_H = 3.1
GROUP_GAP = 3.6
MARKER = 4.6
LEFT_PAD = 2.5
FILL = style.SERIES["nexa_direct"]["fill"]


# ---------------------------------------------------------------------------
# Data: {"cells": {arm: {pert: (n, value %, (lo, hi) %)}},
#        "diff": {arm: {pert: {"delta", "ci", "p_holm"} in fractions}}}
# ---------------------------------------------------------------------------

def accuracy_data(key: str) -> dict:
    """From the perturbation report: the reported accuracies and paired tests."""
    path = os.path.join(RESULTS, EXPERIMENTS[key].name, "stats", "ablation.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run `poetry run python -m benchmark.ablation_direct report` first")
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    cells = defaultdict(dict)
    for arm, row in report["table"].items():
        for pert, cell in row.items():
            n = cell["n"]
            k = round(cell["accuracy"] * n)
            lo, hi = wilson_ci(k, n)
            cells[arm][pert] = (n, 100 * k / n, (100 * lo, 100 * hi))
    return {"cells": cells, "diff": report["vs_full"]}


def graded_data(key: str) -> dict:
    """From the run records: each item's graded score, BCa intervals, sign-flip tests."""
    exp = EXPERIMENTS[key]
    items = load_json(os.path.join(ABLATION_DATASETS, exp.name, "manifest.json"))["items"]
    by_pert = defaultdict(list)
    for qid, meta in items.items():
        by_pert[meta["perturbation"]].append(qid)
    arms = {arm: {qid: r for (_, qid), r in load_experiment_arm(exp.name, arm_dir(arm)).items()}
            for arm, _, _ in (FULL, *ABLATED)}

    def score(record: dict) -> float:
        # MCQ has no set to compare: it keeps its accuracy (the pooled primary metric).
        if record.get("type") == "mcq":
            return float(bool(record.get("correct")))
        return graded_score(record, "jaccard")

    cells = defaultdict(dict)
    for arm, recs in arms.items():
        for pert, _ in PERTURBATIONS:
            vals = [score(recs[q]) for q in by_pert[pert] if q in recs]
            if vals:
                lo, hi = mean_ci_bca(vals)
                cells[arm][pert] = (len(vals), 100 * sum(vals) / len(vals), (100 * lo, 100 * hi))

    diff, family = defaultdict(dict), []
    full = arms["full"]
    for arm, _, _ in ABLATED:
        for pert, _ in PERTURBATIONS:
            qs = [q for q in by_pert[pert] if q in arms[arm] and q in full]
            if not qs:
                continue
            r = paired_bootstrap_diff([score(arms[arm][q]) for q in qs], [score(full[q]) for q in qs])
            diff[arm][pert] = {"n": r["n"], "delta": r["delta"], "ci": [r["ci_lo"], r["ci_hi"]], "p": r["p"]}
            family.append((arm, pert))
    for (arm, pert), adj in zip(family, holm_adjust([diff[a][p]["p"] for a, p in family])):
        diff[arm][pert]["p_holm"] = adj
    return {"cells": cells, "diff": diff}


def panel_data(metric: str):
    """[(data, panel title, graded-metric name)] for each benchmark."""
    loader = accuracy_data if metric == "accuracy" else graded_data
    return [(loader(key), title, graded_name) for key, title, graded_name in PANELS]


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def count(n: int) -> str:
    """Integer with a narrow no-break space as thousands separator."""
    return f"{n:,}".replace(",", " ")


def subtitle(metric: str, graded_name: str) -> str:
    return SUBTITLE if metric == "accuracy" else f"{SUBTITLE} · {graded_name}"


def save(fig, stem: str) -> str:
    """Save and close the figure; return its path without extension."""
    os.makedirs(OUT_DIR, exist_ok=True)
    style.save(fig, os.path.join(OUT_DIR, stem))
    plt.close(fig)
    return os.path.join(OUT_DIR, stem)


def legend_key(fill, label):
    return Line2D([], [], marker="o", linestyle="none", markersize=MARKER, label=label,
                  markerfacecolor=fill, markeredgecolor=style.EDGE, markeredgewidth=0.8)


# ---------------------------------------------------------------------------
# Grouped bars: score per arm and perturbation
# ---------------------------------------------------------------------------

def label_vertical(ax, x: float, v: float, renderer, bounds) -> None:
    """Value along the bar, below the interval; above the bar if it does not fit."""
    lo, hi = bounds
    txt = ax.text(x, lo - VALUE_PAD, f"{v:.1f}", rotation=90, rotation_mode="anchor",
                  ha="right", va="center", fontsize=VALUE_SIZE, color=style.INK_SECONDARY)
    inv = ax.transData.inverted()
    span = (inv.transform((0, txt.get_window_extent(renderer=renderer).height))[1]
            - inv.transform((0, 0))[1])
    if span + 2 * VALUE_PAD > lo:
        txt.set_y(hi + VALUE_PAD)
        txt.set_ha("left")


def draw_bars(metric: str, panels) -> str:
    """Grouped bars of each arm's score per perturbation."""
    style.apply()
    stem, _, ylabel, _ = METRICS[metric]
    arms = [FULL, *ABLATED]
    slot = (1.0 - GROUP_PAD) / len(arms)
    width = max(slot - BAR_GAP, slot * 0.6)

    fig, axes = plt.subplots(1, len(panels), figsize=(style.WIDTH_FULL, 66 * style.MM), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.135, wspace=0.07)

    placed = []
    for ax, (data, title, graded_name), letter in zip(axes, panels, "ab"):
        cells = data["cells"]
        for gi, (pert, _) in enumerate(PERTURBATIONS):
            for j, (arm, _, fill) in enumerate(arms):
                cell = cells.get(arm, {}).get(pert)
                if not cell:
                    continue
                _, v, (lo, hi) = cell
                x = gi + (j - (len(arms) - 1) / 2) * slot
                ax.bar([x], [v], width=width, facecolor=fill, edgecolor=style.EDGE,
                       linewidth=style.EDGE_WIDTH)
                ax.errorbar([x], [v], yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]], fmt="none",
                            ecolor=style.INK_SECONDARY, elinewidth=0.6, capsize=1.3, capthick=0.6,
                            zorder=3)
                placed.append((ax, x, v, (lo, hi)))
        ns = [cells["full"][p][0] for p, _ in PERTURBATIONS]
        ax.set_xticks(np.arange(len(PERTURBATIONS)))
        ax.set_xticklabels([f"{label}\nn = {count(n)}" for (_, label), n in zip(PERTURBATIONS, ns)])
        ax.set_xlim(-0.5, len(PERTURBATIONS) - 0.5)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, YMAX)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title}\n{subtitle(metric, graded_name)}", fontsize=6.5,
                     color=style.INK_SECONDARY, pad=3)
        style.panel_label(ax, letter, dx=-0.07, dy=1.02)
    axes[0].set_ylabel(ylabel)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=fill, edgecolor=style.EDGE,
                             linewidth=style.EDGE_WIDTH) for _, _, fill in arms]
    fig.legend(handles, [label for _, label, _ in arms], loc="upper center",
               bbox_to_anchor=(0.53, 1.0), ncol=len(arms))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, x, v, bounds in placed:
        label_vertical(ax, x, v, renderer, bounds)
    return save(fig, stem)


# ---------------------------------------------------------------------------
# Forest plot: paired difference from complete NEXA Direct
# ---------------------------------------------------------------------------

def draw_difference(metric: str, panels) -> str:
    """Forest plot of each ablated arm's paired difference from the complete system."""
    style.apply()
    _, stem, _, xlabel = METRICS[metric]

    n_rows = len(PERTURBATIONS) * len(ABLATED)
    height_mm = n_rows * ROW_H + len(PERTURBATIONS) * GROUP_GAP + 26
    fig, axes = plt.subplots(1, len(panels), figsize=(style.WIDTH_FULL, height_mm * style.MM),
                             sharey=True, sharex=True)
    fig.subplots_adjust(left=0.3, right=0.985, top=1 - 14.5 / height_mm, bottom=11.5 / height_mm,
                        wspace=0.06)

    lo_all, hi_all = [0.0], [0.0]
    titles, ticks, labels = [], [], []
    for ax, (data, title, graded_name) in zip(axes, panels):
        y = 0.0
        for pert, pert_label in PERTURBATIONS:
            y -= GROUP_GAP / ROW_H
            if ax is axes[0]:
                titles.append(ax.text(-0.02, y + 0.55, pert_label, transform=ax.get_yaxis_transform(),
                                      ha="right", va="bottom", fontsize=6.4, fontweight="bold",
                                      color=style.INK_SECONDARY))
            # n differs between panels, so each panel prints its own on the
            # group-title row.
            n = data["cells"]["full"][pert][0]
            ax.text(0.012, y + 0.55, f"n = {count(n)}", transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=5.8, color=style.INK_MUTED)
            for arm, label, _ in ABLATED:
                if ax is axes[0]:
                    ticks.append(y)
                    labels.append(label)
                r = data["diff"].get(arm, {}).get(pert)
                if not r:
                    ax.text(0, y, "  not run", ha="left", va="center", fontsize=5.6,
                            fontstyle="italic", color=style.INK_MUTED, zorder=4)
                    y -= 1.0
                    continue
                lo, hi = (100 * v for v in r["ci"])
                lo_all.append(lo)
                hi_all.append(hi)
                filled = (r.get("p_holm") or 1.0) < 0.05
                ax.plot([lo, hi], [y, y], color=style.EDGE, linewidth=0.9, solid_capstyle="butt", zorder=3)
                ax.plot([100 * r["delta"]], [y], marker="o", markersize=MARKER, zorder=4,
                        markerfacecolor=FILL if filled else "white", markeredgecolor=style.EDGE,
                        markeredgewidth=0.8, linestyle="none")
                y -= 1.0

        ax.set_title(f"{title}\n{subtitle(metric, graded_name)}", fontsize=6.5,
                     color=style.INK_SECONDARY, pad=3)
        ax.axvline(0, color=style.INK_SECONDARY, linewidth=0.7, zorder=2)
        ax.set_ylim(y + 0.3, 0.2)
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xlabel, fontsize=6.6)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)

    span = max(abs(min(lo_all)), abs(max(hi_all)))
    axes[0].set_xlim(min(lo_all) - 0.04 * span, max(hi_all) + 0.04 * span)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels, fontsize=6.0)

    for _ in range(2):   # left margin from the labels' measured extent
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        x0 = min(t.get_window_extent(renderer).x0 for t in [*axes[0].get_yticklabels(), *titles])
        shift = (x0 - LEFT_PAD * style.MM * fig.dpi) / fig.bbox.width
        fig.subplots_adjust(left=fig.subplotpars.left - shift)
    for ax, letter in zip(axes, "ab"):
        style.panel_label(ax, letter, dx=-0.03, dy=1.06)

    centre = (fig.subplotpars.left + fig.subplotpars.right) / 2
    fig.legend(handles=[legend_key(FILL, "significant after Holm correction"),
                        legend_key("white", "not significant")],
               loc="upper center", bbox_to_anchor=(centre, 1.0), ncol=2, fontsize=5.8,
               handletextpad=0.6, handlelength=1.8, columnspacing=1.2, frameon=False)
    return save(fig, stem)


def main() -> None:
    for metric in METRICS:
        panels = panel_data(metric)
        for out in (draw_bars(metric, panels), draw_difference(metric, panels)):
            print(f"wrote {out}.png / .tiff")


if __name__ == "__main__":
    main()

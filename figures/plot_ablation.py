"""Ablation of NEXA Direct: each component's removal, as a difference from the complete system.

Reads benchmark/results/{biohopr,alzkb}_ablation/stats/ablation.json (written
by `poetry run python -m benchmark.ablation_direct report`) and draws a
two-panel forest plot, figures/out/ablation_direct.{png,tiff}:

  a  BioHopR / PrimeKG v2.1, 1 000 sampled questions, Jaccard index
  b  AlzKB v1.2.1, 1 000 sampled questions over the six subsets, pooled primary
     metric (accuracy for True/False and MCQ, Jaccard index for open-ended)

Each row is one ablated arm: its paired difference from complete NEXA Direct
with the 95% BCa bootstrap interval. Filled markers are significant after Holm
correction over the ablations of that benchmark. The panels share the x axis.

Usage:
    poetry run python figures/plot_ablation.py
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import style

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_HERE), "benchmark", "results")
OUT_DIR = os.path.join(_HERE, "out")
STEM = "ablation_direct"

# (group title, [(arm label in ablation.json, row label)]), top to bottom.
GROUPS = [
    ("Few-shot examples", [
        ("no-examples", "No few-shot examples"),
        ("generic-examples", "Generic examples"),
    ]),
    ("Recovery path", [
        ("bare", "No recovery"),
        ("no-empty-loop", "No empty-result correction"),
        ("one-shot", "No decomposed fallback"),
        ("no-db-matching", "No database value matching"),
        ("no-llm-selection", "No LLM candidate selection"),
        ("no-syntax", "No syntactic correction"),
    ]),
    ("Generation strategy", [
        ("decomposed", "Fully decomposed strategy"),
    ]),
]

# (experiment directory, panel title, metric name, where the row lives in an arm's entry)
PANELS = [
    ("biohopr_ablation", "BioHopR · PrimeKG v2.1", "Jaccard index",
     lambda arm: arm["sets"]["BioHopR_2hop.json"]["all"]),
    ("alzkb_ablation", "AlzKB v1.2.1 · six benchmark subsets", "pooled primary metric",
     lambda arm: arm["pooled"]["all"]),
]

ROW_H = 3.1          # mm per row, as in the research-question plots
GROUP_GAP = 3.6      # mm between groups, where the group title sits
MARKER = 4.6         # pt
LEFT_PAD = 2.5       # mm between the figure edge and the widest label
FILL = style.SERIES["nexa_direct"]["fill"]   # every arm is NEXA Direct


def load(experiment: str) -> dict:
    """The ablation report of one experiment."""
    path = os.path.join(RESULTS, experiment, "stats", "ablation.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run `poetry run python -m benchmark.ablation_direct report` first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def draw() -> str:
    """Draw and save the figure; return its path without extension."""
    style.apply()
    panels = [(load(exp), title, metric, row) for exp, title, metric, row in PANELS]

    n_rows = sum(len(arms) for _, arms in GROUPS)
    height_mm = n_rows * ROW_H + len(GROUPS) * GROUP_GAP + 26
    fig, axes = plt.subplots(1, len(panels), figsize=(style.WIDTH_FULL, height_mm * style.MM),
                             sharey=True, sharex=True)
    fig.subplots_adjust(left=0.3, right=0.985, top=1 - 14.5 / height_mm, bottom=11.5 / height_mm,
                        wspace=0.06)

    lo_all, hi_all = [0.0], [0.0]
    titles, ticks, labels = [], [], []
    for ax, (results, title, metric, row_of) in zip(axes, panels):
        y, reference, n = 0.0, None, None
        for group_title, arms in GROUPS:
            y -= GROUP_GAP / ROW_H
            if ax is axes[0]:
                titles.append(ax.text(-0.02, y + 0.55, group_title, transform=ax.get_yaxis_transform(),
                                      ha="right", va="bottom", fontsize=6.4, fontweight="bold",
                                      color=style.INK_SECONDARY))
            for arm, label in arms:
                entry = results["arms"].get(arm)
                if ax is axes[0]:
                    ticks.append(y)
                    labels.append(label)
                if not entry:
                    ax.text(0, y, "  not run", ha="left", va="center", fontsize=5.6,
                            fontstyle="italic", color=style.INK_MUTED, zorder=4)
                    y -= 1.0
                    continue
                r = row_of(entry)
                reference, n = r["b"], r["n"]
                lo, hi = (100 * v for v in r["ci"])
                lo_all.append(lo)
                hi_all.append(hi)
                filled = r.get("p_holm", 1.0) < 0.05
                ax.plot([lo, hi], [y, y], color=style.EDGE, linewidth=0.9, solid_capstyle="butt", zorder=3)
                ax.plot([100 * r["delta"]], [y], marker="o", markersize=MARKER, zorder=4,
                        markerfacecolor=FILL if filled else "white", markeredgecolor=style.EDGE,
                        markeredgewidth=0.8, linestyle="none")
                y -= 1.0

        count = f"{n:,}".replace(",", " ") if n else "—"
        ax.set_title(f"{title}\nn = {count} · {metric} · complete system = {100 * reference:.1f}",
                     fontsize=6.5, color=style.INK_SECONDARY, pad=3)
        ax.axvline(0, color=style.INK_SECONDARY, linewidth=0.7, zorder=2)
        ax.set_ylim(y + 0.3, 0.2)
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Difference from complete NEXA Direct (pp)", fontsize=6.6)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)

    span = max(abs(min(lo_all)), abs(max(hi_all)))
    axes[0].set_xlim(min(lo_all) - 0.06 * span, max(hi_all) + 0.06 * span)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels, fontsize=6.0)

    # Fit the left margin to the widest label; twice, because the group titles
    # move with the axes.
    for _ in range(2):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        x0 = min(t.get_window_extent(renderer).x0 for t in [*axes[0].get_yticklabels(), *titles])
        shift = (x0 - LEFT_PAD * style.MM * fig.dpi) / fig.bbox.width
        fig.subplots_adjust(left=fig.subplotpars.left - shift)
    for ax, letter in zip(axes, "ab"):
        style.panel_label(ax, letter, dx=-0.03, dy=1.06)

    def key(fill, label):
        return Line2D([], [], marker="o", linestyle="none", markersize=MARKER, label=label,
                      markerfacecolor=fill, markeredgecolor=style.EDGE, markeredgewidth=0.8)

    centre = (fig.subplotpars.left + fig.subplotpars.right) / 2
    fig.legend(handles=[key(FILL, "significant after Holm correction"), key("white", "not significant")],
               loc="upper center", bbox_to_anchor=(centre, 1.0), ncol=2, fontsize=5.8,
               handletextpad=0.6, handlelength=1.8, columnspacing=1.2, frameon=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    style.save(fig, os.path.join(OUT_DIR, STEM))
    plt.close(fig)
    return os.path.join(OUT_DIR, STEM)


def main() -> None:
    out = draw()
    print(f"wrote {out}.png / .tiff")


if __name__ == "__main__":
    main()

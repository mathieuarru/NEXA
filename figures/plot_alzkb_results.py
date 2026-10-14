"""AlzKB cross-system results: one bar per system and backbone, with 95% intervals.

Figures written to figures/out/ (PNG and TIFF):

  alzkb_results_overlap                 the six question sets (x = backbone,
                                        colour = system), open-ended panels
                                        scored with the overlap coefficient,
                                        the metric ESCARGOT published
  alzkb_results_jaccard                 the same, open-ended panels scored
                                        with the Jaccard index
  alzkb_open_ended_jaccard              the two open-ended panels alone,
                                        Jaccard index
  alzkb_open_ended_jaccard_exact_match  the open-ended panels, Jaccard index
                                        and exact set match

Not every system ran on every backbone, so group sizes differ; bar width is
constant. The intervals are marginal.

Usage:
    poetry run python figures/plot_alzkb_results.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import data
import style
from data import (DATASETS, MODEL_LABEL, MODELS, RESULTS, SYSTEMS,
                  ci, value)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Metric names as written mid-sentence.
METRIC_INLINE = {"accuracy": "accuracy", "overlap": "overlap coefficient",
                 "jaccard": "Jaccard index", "exact_match": "exact set match"}


# Bar pitch is set by the backbone with the most systems. BAR_GAP is absolute
# (data units), so the gap between bars stays the same at any bar width.
GROUP_PAD = 0.14    # fraction of a group slot left empty between groups
BAR_GAP = 0.0224    # data units of white space between bars inside a group

VALUE_PAD = 2.0     # data units between the bar end and its label
VALUE_SIZE = 5.4    # pt

YMAX = 104


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def systems_for(model: str):
    """Systems that ran on this backbone, in the canonical display order."""
    m = MODELS.index(model)
    return [
        (sid, label)
        for sid, label in SYSTEMS
        if any(RESULTS[did][sid][m] is not None for did, _, _, _ in DATASETS)
    ]


def slot_geometry(n_max: int):
    """(pitch, width) of the bars in a group of at most `n_max` bars.

    The width never falls below 60% of the pitch.
    """
    slot = (1.0 - GROUP_PAD) / n_max
    return slot, max(slot - BAR_GAP, slot * 0.6)


def bar_kwargs(sid: str):
    """Matplotlib bar style of a system."""
    spec = style.SERIES[sid]
    return dict(
        facecolor=spec["fill"],
        edgecolor=spec["edge"],
        hatch=spec["hatch"],
        linewidth=style.EDGE_WIDTH,
    )


def draw_ci(ax, x: float, did: str, sid: str, model: str) -> None:
    """Draw the marginal 95% interval of one bar as a capped error bar.

    External cells (no per-question records) have no interval and get none.
    """
    bounds = ci(did, sid, model)
    if bounds is None:
        return
    lo, hi = bounds
    v = value(did, sid, model)
    ax.errorbar(
        [x], [v], yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
        fmt="none", ecolor=style.INK_SECONDARY,
        elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=3,
    )


def style_value_axis(ax, axis: str = "y") -> None:
    """0-100 score axis with its grid lines."""
    setter = ax.set_ylim if axis == "y" else ax.set_xlim
    ticks = ax.set_yticks if axis == "y" else ax.set_xticks
    setter(0, YMAX)
    ticks(range(0, 101, 20))
    ax.grid(axis=axis, visible=True)
    ax.grid(axis="x" if axis == "y" else "y", visible=False)


def label_vertical(ax, x: float, v: float, renderer, sid: str, bounds=None) -> None:
    """Write the bar's value along it, rotated to read bottom to top.

    The label goes inside the bar, below the interval's lower end, and moves
    above the bar when its rendered height does not fit. Hatched bars always
    get the label above.
    """
    outside = style.SERIES[sid]["hatch"] is not None
    lo, hi = bounds if bounds else (v, v)
    top = hi + VALUE_PAD if outside else lo - VALUE_PAD
    txt = ax.text(
        x, top, f"{v:.1f}",
        rotation=90, rotation_mode="anchor",
        ha="left" if outside else "right", va="center",
        fontsize=VALUE_SIZE, color=style.INK_SECONDARY,
    )
    if outside:
        return
    inv = ax.transData.inverted()
    span = (
        inv.transform((0, txt.get_window_extent(renderer=renderer).height))[1]
        - inv.transform((0, 0))[1]
    )
    # The label must fit below the interval's lower end.
    if span + 2 * VALUE_PAD > lo:
        txt.set_y(hi + VALUE_PAD)
        txt.set_ha("left")


def add_legend(fig, y: float = 1.0, x: float = 0.5, systems=None) -> None:
    """One-row figure legend, one entry per system."""
    systems = SYSTEMS if systems is None else systems
    handles = [
        plt.Rectangle((0, 0), 1, 1, **bar_kwargs(sid)) for sid, _ in systems
    ]
    fig.legend(
        handles, [label for _, label in systems],
        loc="upper center", bbox_to_anchor=(x, y), ncol=len(systems),
    )


# ---------------------------------------------------------------------------
# Layout: one small multiple per question set, backbones on x
# ---------------------------------------------------------------------------

def layout_facets():
    """Six panels, one per question set, with the backbones on x."""
    n_max = max(len(systems_for(m)) for m in MODELS)
    slot, width = slot_geometry(n_max)

    fig, axes = plt.subplots(
        2, 3, figsize=(style.WIDTH_FULL, 104 * style.MM), sharey=True,
    )
    # left leaves room for the y label and its tick labels
    fig.subplots_adjust(left=0.078, right=0.997, top=0.865, bottom=0.085,
                        hspace=0.30, wspace=0.07)

    placed = {}
    for ax, (did, label, n, metric), letter in zip(
        axes.ravel(), DATASETS, "abcdef"
    ):
        bars = []
        for gi, model in enumerate(MODELS):
            m = MODELS.index(model)
            present = [
                (sid, lab) for sid, lab in systems_for(model)
                if RESULTS[did][sid][m] is not None
            ]
            for j, (sid, _) in enumerate(present):
                x = gi + (j - (len(present) - 1) / 2) * slot
                v = RESULTS[did][sid][m]
                ax.bar([x], [v], width=width, **bar_kwargs(sid))
                draw_ci(ax, x, did, sid, model)
                bars.append((x, v, sid, ci(did, sid, model)))
        placed[ax] = bars

        style_value_axis(ax)
        ax.set_xlim(-0.5, len(MODELS) - 0.5)
        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
        ax.tick_params(axis="x", length=0)
        ax.set_title(
            f"{label.replace(chr(10), ' ')}\n"
            f"n = {n} · {METRIC_INLINE[metric]}",
            fontsize=6.5, color=style.INK_SECONDARY, pad=3,
        )
        # level with the title; higher up it would collide with the legend
        style.panel_label(ax, letter, dx=-0.10, dy=1.02)

    for ax in axes[:, 0]:
        ax.set_ylabel("Score (%)")
    add_legend(fig, x=0.53)
    return fig, placed


# ---------------------------------------------------------------------------
# Layout: the open-ended question sets only
# ---------------------------------------------------------------------------

def _oe_sets():
    return [d for d in DATASETS if d[3] != "accuracy"]


def _oe_systems():
    """Systems with a bar in the open-ended panels, and the largest group size."""
    oe = _oe_sets()

    def has_value(sid, model):
        m = MODELS.index(model)
        return any(RESULTS[did][sid][m] is not None for did, _, _, _ in oe)

    shown = [(sid, lab) for sid, lab in SYSTEMS if any(has_value(sid, m) for m in MODELS)]
    n_max = max(sum(1 for sid, _ in shown if has_value(sid, m)) for m in MODELS)
    return shown, n_max


def _draw_oe_row(axes, shown, slot, width, letters):
    """One panel per open-ended set, scored with the current open-ended metric."""
    placed = {}
    for ax, (did, label, n, metric), letter in zip(axes, _oe_sets(), letters):
        bars = []
        for gi, model in enumerate(MODELS):
            m = MODELS.index(model)
            present = [(sid, lab) for sid, lab in shown if RESULTS[did][sid][m] is not None]
            for j, (sid, _) in enumerate(present):
                x = gi + (j - (len(present) - 1) / 2) * slot
                v = RESULTS[did][sid][m]
                ax.bar([x], [v], width=width, **bar_kwargs(sid))
                draw_ci(ax, x, did, sid, model)
                bars.append((x, v, sid, ci(did, sid, model)))
        placed[ax] = bars

        style_value_axis(ax)
        ax.set_xlim(-0.5, len(MODELS) - 0.5)
        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels([MODEL_LABEL[mm] for mm in MODELS])
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{label.replace(chr(10), ' ')}\n"
                     f"n = {n} · {METRIC_INLINE[metric]}",
                     fontsize=6.5, color=style.INK_SECONDARY, pad=3)
        style.panel_label(ax, letter, dx=-0.07, dy=1.02)
    return placed


def layout_open_ended():
    """The two open-ended panels alone, side by side.

    Bar pitch and legend come from these panels only, so the legend omits
    systems without a bar here (the published ESCARGOT results report no Jaccard).
    """
    shown, n_max = _oe_systems()
    slot, width = slot_geometry(n_max)
    fig, axes = plt.subplots(1, len(_oe_sets()), figsize=(style.WIDTH_FULL, 62 * style.MM),
                             sharey=True)
    fig.subplots_adjust(left=0.078, right=0.997, top=0.79, bottom=0.10, wspace=0.07)
    placed = _draw_oe_row(axes, shown, slot, width, "ab")
    axes[0].set_ylabel("Score (%)")
    add_legend(fig, x=0.53, systems=shown)
    return fig, placed


OE_TWO_METRICS = ("jaccard", "exact_match")


def layout_open_ended_two_metrics():
    """Open-ended panels scored with the Jaccard index (top) and exact set match (bottom).

    The metric is switched in `data` row by row. Both rows share the bar
    pitch, and the legend lists every system with a bar in either row.
    """
    shown_by_row, n_max = {}, 0
    for metric in OE_TWO_METRICS:
        data.use_oe_metric(metric)
        shown_by_row[metric], n = _oe_systems()
        n_max = max(n_max, n)
    union = [(sid, lab) for sid, lab in SYSTEMS
             if any(sid in dict(shown_by_row[m]) for m in OE_TWO_METRICS)]
    slot, width = slot_geometry(n_max)

    fig, axes = plt.subplots(2, len(_oe_sets()), figsize=(style.WIDTH_FULL, 112 * style.MM),
                             sharey=True)
    fig.subplots_adjust(left=0.078, right=0.997, top=0.875, bottom=0.055,
                        hspace=0.36, wspace=0.07)
    placed = {}
    for row_axes, metric, letters in zip(axes, OE_TWO_METRICS, ("ab", "cd")):
        data.use_oe_metric(metric)
        placed.update(_draw_oe_row(row_axes, shown_by_row[metric], slot, width, letters))
    for ax in axes[:, 0]:
        ax.set_ylabel("Score (%)")
    add_legend(fig, x=0.53, systems=union)
    return fig, placed


LAYOUTS = {
    "facets": layout_facets,
    "open-ended": layout_open_ended,
    "open-ended-two-metrics": layout_open_ended_two_metrics,
}

# (output name, layout, open-ended metric)
FIGURES = [
    ("alzkb_results_overlap", "facets", "overlap"),
    ("alzkb_results_jaccard", "facets", "jaccard"),
    ("alzkb_open_ended_jaccard", "open-ended", "jaccard"),
    ("alzkb_open_ended_jaccard_exact_match", "open-ended-two-metrics", None),
]


# ---------------------------------------------------------------------------

def render(stem: str, layout: str, metric) -> None:
    """Draw one figure and write it to figures/out/."""
    style.apply()
    if metric:                 # None: the layout sets the metric of each row itself
        data.use_oe_metric(metric)
    fig, placed = LAYOUTS[layout]()

    # Draw once so the transforms are final, then place the value labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, bars in placed.items():
        for pos, v, sid, bounds in bars:
            label_vertical(ax, pos, v, renderer, sid, bounds)

    os.makedirs(OUT_DIR, exist_ok=True)
    for path in style.save(fig, os.path.join(OUT_DIR, stem)):
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    for stem, layout, metric in FIGURES:
        render(stem, layout, metric)
    print(f"font: {style.resolve_sans()}")


if __name__ == "__main__":
    main()

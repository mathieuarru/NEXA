"""Forest plots for the three research questions, per benchmark and overall.

Reads benchmark/results/{alzkb,biohopr}/stats/rq/report.json (written by
`poetry run python -m benchmark.report_rq`) and draws, in figures/out/:

  <benchmark>_rq1_nexa_vs_graphcypherqa   NEXA minus GraphCypherQAChain, same backbone
  <benchmark>_rq2_flex_vs_direct          Flex minus Direct, same backbone
  <benchmark>_rq3_qwen_vs_gpt5mini        Qwen3.8-27B minus GPT-5-mini, same mode

for alzkb and biohopr, and the same three as overall_* with both benchmarks on
a shared axis. Each research question is one Holm family across both
benchmarks.

Each row is a paired difference with its 95% interval; right of zero, the first
system of the contrast is higher. A filled marker is significant after Holm
correction. Rows are labelled in text and coloured by NEXA mode (blue Direct,
orange Flex). The disambiguated True/False 2-hop rows, below a rule, are a
sensitivity analysis with a Holm family of their own.

Usage:
    poetry run python figures/plot_research_questions.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import style

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(_HERE), "benchmark", "results")


def report_path(benchmark: str) -> str:
    """Path of a benchmark's research-question report."""
    return os.path.join(RESULTS, benchmark, "stats", "rq", "report.json")


OUT_DIR = os.path.join(_HERE, "out")


def mode_fill(mode: str) -> str:
    """Fill colour of a NEXA mode, as in the bar figures."""
    return style.SERIES["nexa_flex" if mode == "Flex" else "nexa_direct"]["fill"]


BACKBONE_ORDER = ["GPT-4o-mini", "GPT-5-mini", "Qwen3.8-27B"]
METRIC_NAME = {"accuracy": "accuracy", "jaccard": "Jaccard"}

ROW_H = 3.1          # mm per contrast row
GROUP_GAP = 3.6      # mm between groups, where the group title sits
TITLE_LINE = 2.7     # mm added to that gap for each extra line of a group title
HEADER_GAP = 4.4     # mm for a benchmark header, overall figures only
HEADER_FONT = 7.0    # pt
LEFT_PAD = 2.5       # mm between the figure edge and the widest label
FIG_W = 0.72         # fraction of the full text width
MARKER = 4.6         # pt


def _mode(row: dict) -> str:
    """The NEXA mode of the row's first arm, the one the difference is about."""
    return "Flex" if "flex" in row["a"] else "Direct"


def _row_label(row: dict, rq: str) -> str:
    if rq == "RQ3":
        return row["label"]                       # the mode; backbones are fixed
    if rq == "RQ2":
        return row["backbone"]                    # the contrast is fixed
    return f"{row['backbone']} · {row['label']}"


def _groups(rows, sets):
    """[(set_label, metric, [rows in backbone order])] keeping the declared set order."""
    out = []
    for ds, label, metric in sets:
        # Rows whose arm has not run are kept and marked.
        rs = [r for r in rows if r["dataset"] == ds]
        rs.sort(key=lambda r: (BACKBONE_ORDER.index(r["backbone"]) if r["backbone"] in BACKBONE_ORDER else 0,
                               0 if _mode(r) == "Direct" else 1))
        if rs:
            out.append((label, metric, rs))
    return out


def _title(set_label: str, metric: str, is_sens: bool) -> str:
    """Group title; the long sensitivity-set name puts its metric on a second line."""
    return f"{set_label}{chr(10) if is_sens else ' · '}{METRIC_NAME[metric]}"


def forest(rq: str, blocks, ci_key: str, xlabel: str, fname: str) -> str:
    """One forest plot from one or several benchmarks.

    `blocks` is [(header, report, sets, sensitivity sets)], drawn top to bottom.
    A header (overall figures) opens its block below a rule; the sensitivity
    sets sit below a lighter rule.
    """
    sections = []
    for header, data, sets, sens_sets in blocks:
        q, s = data[rq], data.get(f"{rq}_sensitivity", {"rows": []})
        sections.append((header, [(g, False) for g in _groups(q["rows"], sets)] +
                                 [(g, True) for g in _groups(s["rows"], sens_sets)]))
    groups = [g for _, gs in sections for g in gs]

    n_rows = sum(len(rs) for (_, _, rs), _ in groups)
    extra_lines = sum(_title(lbl, m, sens).count("\n") for (lbl, m, _), sens in groups)
    n_headers = sum(1 for header, _ in sections if header)
    height_mm = (n_rows * ROW_H + len(groups) * GROUP_GAP + extra_lines * TITLE_LINE
                 + n_headers * HEADER_GAP + 26)
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL * FIG_W, height_mm * style.MM))
    fig.subplots_adjust(left=0.345, right=0.975, top=1 - 11 / height_mm, bottom=15 / height_mm)

    y, ticks, labels, lo_all, hi_all = 0.0, [], [], [0.0], [0.0]
    sens_rules, block_rules = [], []
    titles = []
    for i, (header, section) in enumerate(sections):
        if header:
            if i:
                block_rules.append(y + 0.35)
            y -= HEADER_GAP / ROW_H
            titles.append(ax.text(-0.02, y + 0.55, header, transform=ax.get_yaxis_transform(),
                                  ha="right", va="bottom", fontsize=HEADER_FONT, fontweight="bold",
                                  color=style.INK_PRIMARY))
        sens_started = False
        for (set_label, metric, rs), is_sens in section:
            if is_sens and not sens_started:
                sens_rules.append(y + 0.35)
                sens_started = True
            title = _title(set_label, metric, is_sens)
            y -= (GROUP_GAP + title.count("\n") * TITLE_LINE) / ROW_H
            titles.append(ax.text(-0.02, y + 0.55, title, transform=ax.get_yaxis_transform(),
                                  ha="right", va="bottom", multialignment="right",
                                  fontsize=6.4, fontweight="bold", color=style.INK_SECONDARY))
            y = _draw_rows(ax, rq, rs, ci_key, y, ticks, labels, lo_all, hi_all)

    return _finish(fig, ax, rq, xlabel, fname, y, ticks, labels, lo_all, hi_all, titles,
                   sens_rules, block_rules)


def _draw_rows(ax, rq, rs, ci_key, y, ticks, labels, lo_all, hi_all):
    """The rows of one group, from `y` down; returns the next free `y`."""
    for r in rs:
        if r["status"] != "ok":
            # Name the missing arm; the row label names the NEXA side, which ran.
            absent = " and ".join(arm_name(a) for a in (r.get("missing") or [r["b"]]))
            why = "not run yet" if r["status"] == "pending" else "not run on this set"
            ax.text(0, y, f"  {absent} {why}", ha="left", va="center", fontsize=5.6,
                    fontstyle="italic", color=style.INK_MUTED, zorder=4)
            ticks.append(y)
            labels.append(_row_label(r, rq))
            y -= 1.0
            continue
        lo, hi = r[ci_key]
        lo_all.append(lo)
        hi_all.append(hi)
        # RQ2 (Flex minus Direct) is a single series, drawn in the Flex colour.
        color = mode_fill("Flex") if rq == "RQ2" else mode_fill(_mode(r))
        filled = bool(r.get("significant"))
        ax.plot([lo, hi], [y, y], color=style.EDGE, linewidth=0.9, solid_capstyle="butt", zorder=3)
        ax.plot([r["delta"]], [y], marker="o", markersize=MARKER, zorder=4,
                markerfacecolor=color if filled else "white", markeredgecolor=style.EDGE,
                markeredgewidth=0.8, linestyle="none")
        ticks.append(y)
        # An incomplete run is compared on the questions it has reached.
        labels.append(_row_label(r, rq) + (" (partial)" if r.get("partial") else ""))
        y -= 1.0
    return y


def _finish(fig, ax, rq, xlabel, fname, y, ticks, labels, lo_all, hi_all, titles, sens_rules, block_rules):
    """Axes, rules, margins and legend; save the figure and return its path stem."""
    span = max(abs(min(lo_all)), abs(max(hi_all)))
    left = min(lo_all) - 0.08 * span
    right = max(hi_all) + 0.08 * span
    ax.axvline(0, color=style.INK_SECONDARY, linewidth=0.7, zorder=2)
    for rule in sens_rules:
        ax.axhline(rule, color=style.BASELINE, linewidth=0.6, zorder=1)
    for rule in block_rules:  # between benchmarks: stronger than the sensitivity rule
        ax.axhline(rule, color=style.INK_SECONDARY, linewidth=0.8, zorder=1)

    ax.set_xlim(left, right)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.set_ylim(y + 0.3, 0.2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel, fontsize=6.6)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    # Fit the left margin to the widest label; twice, because the titles move
    # with the axes.
    for _ in range(2):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        x0 = min(t.get_window_extent(renderer).x0 for t in [*ax.get_yticklabels(), *titles])
        shift = (x0 - LEFT_PAD * style.MM * fig.dpi) / fig.bbox.width
        fig.subplots_adjust(left=fig.subplotpars.left - shift)

    def key(fill, label):
        return Line2D([], [], marker="o", linestyle="none", markersize=MARKER, label=label,
                      markerfacecolor=fill, markeredgecolor=style.EDGE, markeredgewidth=0.8)

    sig_fill = mode_fill("Flex") if rq == "RQ2" else style.INK_MUTED
    handles = [key(mode_fill("Direct"), "NEXA Direct"), key(mode_fill("Flex"), "NEXA Flex"),
               key(sig_fill, "significant after Holm correction"), key("white", "not significant")]
    if rq == "RQ2":
        handles = handles[2:]   # single series: only the significance encoding needs a key
    ncol = len(handles) if len(handles) <= 4 else (len(handles) + 1) // 2
    centre = (fig.subplotpars.left + fig.subplotpars.right) / 2   # over the plot, not the page
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(centre, 1.0),
               ncol=ncol, fontsize=5.8, handletextpad=0.6, handlelength=1.8, columnspacing=1.2,
               frameon=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    style.save(fig, os.path.join(OUT_DIR, fname))
    plt.close(fig)
    return os.path.join(OUT_DIR, fname)


SYSTEM_NAME = {"graphcypher": "GraphCypherQAChain", "nexa-direct": "NEXA Direct", "nexa-flex": "NEXA Flex",
               "vanilla": "LLM only"}
MODEL_NAME = {"gpt-4o-mini": "GPT-4o-mini", "gpt-5-mini": "GPT-5-mini", "qwen3.8-27b": "Qwen3.8-27B"}


def arm_name(arm: str) -> str:
    """'graphcypher_qwen3.8-27b' -> 'GraphCypherQAChain Qwen3.8-27B'."""
    system, _, model = arm.partition("_")
    return f"{SYSTEM_NAME.get(system, system)} {MODEL_NAME.get(model, model)}"


BENCHMARK_SETS = {
    "alzkb": ([("True_or_False_1hop.json", "TF 1-hop", "accuracy"),
               ("True_or_False_2hop.json", "TF 2-hop", "accuracy"),
               ("MCQ_1hop.json", "MCQ 1-hop", "accuracy"),
               ("MCQ_2hop.json", "MCQ 2-hop", "accuracy"),
               ("OpenEnded_1hop.json", "OE 1-hop", "jaccard"),
               ("OpenEnded_2hop.json", "OE 2-hop", "jaccard")],
              [("True_or_False_2hop_atleastone.json", "TF 2-hop disambiguated", "accuracy")]),
    "biohopr": ([("BioHopR_2hop.json", "BioHopR 2-hop", "jaccard")], []),
}

# (question, axis title, output name after the benchmark prefix)
FIGURES = [
    ("RQ1", "Score difference, NEXA relative to GraphCypherQAChain (pp)", "rq1_nexa_vs_graphcypherqa"),
    ("RQ2", "Score difference, Flex relative to Direct (pp)", "rq2_flex_vs_direct"),
    ("RQ3", "Score difference, Qwen3.8-27B relative to GPT-5-mini (pp)", "rq3_qwen_vs_gpt5mini"),
]


BENCHMARK_HEADER = {"alzkb": "AlzKB v1.2.1", "biohopr": "BioHopR · PrimeKG v2.1"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=[*BENCHMARK_SETS, "overall", "all"], default="all",
                    help="default: every benchmark whose report exists, then the overall figures")
    args = ap.parse_args()
    chosen = list(BENCHMARK_SETS) if args.benchmark in ("all", "overall") else [args.benchmark]

    reports = {}
    for benchmark in chosen:
        path = report_path(benchmark)
        if not os.path.exists(path):
            msg = f"{path} not found; run `poetry run python -m benchmark.report_rq` first"
            if args.benchmark == "all":
                print("skipped: " + msg)
                continue
            raise SystemExit(msg)
        with open(path, encoding="utf-8") as fh:
            reports[benchmark] = json.load(fh)

    style.apply()
    if args.benchmark != "overall":
        for benchmark, data in reports.items():
            sets, sens = BENCHMARK_SETS[benchmark]
            for rq, xlabel, stem in FIGURES:
                out = forest(rq, [(None, data, sets, sens)], "ci95", xlabel, f"{benchmark}_{stem}")
                print(f"wrote {out}.png / .tiff")

    # Overall: every benchmark in one figure per question, AlzKB first.
    if args.benchmark in ("all", "overall") and len(reports) == len(BENCHMARK_SETS):
        blocks = [(BENCHMARK_HEADER[b], reports[b], *BENCHMARK_SETS[b]) for b in BENCHMARK_SETS]
        for rq, xlabel, stem in FIGURES:
            out = forest(rq, blocks, "ci95", xlabel, f"overall_{stem}")
            print(f"wrote {out}.png / .tiff")


if __name__ == "__main__":
    main()

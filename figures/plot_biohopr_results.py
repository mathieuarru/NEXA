"""BioHopR / PrimeKG cross-system results: one panel, x = backbone, colour = system.

BioHopR is a single set of 7 633 two-hop questions whose gold answer is a list
of entities. Figures are drawn for exact set match and the Jaccard index by
default; `--metric` selects one metric. The overlap coefficient is not used
because nearly every NEXA arm scores 99+ on it.

  --subset full     (default) only the arms that answered all 7 633 questions.
  --subset matched  every arm on the questions all arms have answered,
                    incomplete runs included. The dataset is ordered, so this
                    slice is not a random sample and must not be compared with
                    the full-set figure.

Output in figures/out/: biohopr_results_<metric>, with a _matched suffix for
the matched subset.

Usage:
    poetry run python figures/plot_biohopr_results.py                # exact match and Jaccard
    poetry run python figures/plot_biohopr_results.py --metric f1    # one metric only
    poetry run python figures/plot_biohopr_results.py --subset matched
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import style

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from benchmark.bench_utils import wilson_ci  # noqa: E402
from benchmark.paired_stats import mean_ci_bca  # noqa: E402
from benchmark.report_paired_stats import load_arm  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

SYSTEMS = [
    ("llm_only", "LLM Only"),
    ("graphcypher", "GraphCypherQAChain"),
    ("nexa_direct", "NEXA Direct"),
    ("nexa_flex", "NEXA Flex"),
]

MODELS = ["gpt-4o-mini", "gpt-5-mini", "qwen3.8:27b-q8_0"]
MODEL_LABEL = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5-mini": "GPT-5-mini",
    "qwen3.8:27b-q8_0": "Qwen3.8-27B",
}

# (system_id, model) -> glob for that arm's records.
ARMS = {
    ("llm_only", "gpt-4o-mini"): "benchmark/results/biohopr/vanilla_gpt-4o-mini/*.jsonl",
    ("llm_only", "gpt-5-mini"): "benchmark/results/biohopr/vanilla_gpt-5-mini/*.jsonl",
    ("graphcypher", "gpt-4o-mini"): "benchmark/results/biohopr/graphcypher_gpt-4o-mini/*.jsonl",
    ("graphcypher", "gpt-5-mini"): "benchmark/results/biohopr/graphcypher_gpt-5-mini/*.jsonl",
    ("nexa_direct", "gpt-4o-mini"): "benchmark/results/biohopr/nexa-direct_gpt-4o-mini/*.jsonl",
    ("nexa_direct", "gpt-5-mini"): "benchmark/results/biohopr/nexa-direct_gpt-5-mini/*.jsonl",
    ("nexa_direct", "qwen3.8:27b-q8_0"): "benchmark/results/biohopr/nexa-direct_qwen3.8-27b/*.jsonl",
    ("nexa_flex", "gpt-5-mini"): "benchmark/results/biohopr/nexa-flex_gpt-5-mini/*.jsonl",
    ("nexa_flex", "qwen3.8:27b-q8_0"): "benchmark/results/biohopr/nexa-flex_qwen3.8-27b/*.jsonl",
    ("llm_only", "qwen3.8:27b-q8_0"): "benchmark/results/biohopr/vanilla_qwen3.8-27b/*.jsonl",
    ("graphcypher", "qwen3.8:27b-q8_0"): "benchmark/results/biohopr/graphcypher_qwen3.8-27b/*.jsonl",
}

N_FULL = 7633
METRIC_LABEL = {
    "exact_match": "Exact set match (%)",
    "f1": "F1 (%)",
    "recall": "Recall (%)",
    "precision": "Precision (%)",
    "jaccard": "Jaccard index (%)",
}

GROUP_PAD = 0.14
BAR_GAP = 0.0224
# Cap the bar width so a sparse group keeps the proportions of a full one.
BAR_WIDTH_MAX = 0.17
VALUE_PAD = 2.0
VALUE_SIZE = 5.4
YMAX = 104    # as in the AlzKB figure: a little room above 100 % for the value labels


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_all():
    """Every arm's records, keyed by (system, model) then qid.

    Arms with no records are left out.
    """
    out = {}
    for key, pattern in ARMS.items():
        recs = load_arm([pattern])
        if recs:
            out[key] = {qid: rec for (_, qid), rec in recs.items()}
    return out


def score(records, qids, metric: str):
    """(value %, (lo, hi) %) for one arm over `qids`.

    Exact match takes the Wilson interval, graded metrics a BCa bootstrap of
    the mean. A record without `metrics` scores 0 and stays in the denominator.
    """
    present = [q for q in qids if q in records]
    n = len(present)
    if not n:
        return None, None
    if metric == "exact_match":
        c = sum(1 for q in present if records[q].get("correct"))
        lo, hi = wilson_ci(c, n)
        return 100.0 * c / n, (100.0 * lo, 100.0 * hi)
    vals = []
    for q in present:
        v = (records[q].get("metrics") or {}).get(metric)
        vals.append(float(v) if isinstance(v, (int, float)) else 0.0)
    lo, hi = mean_ci_bca(vals)
    return 100.0 * sum(vals) / n, (100.0 * lo, 100.0 * hi)


def select(arms, subset: str):
    """The question ids every bar is computed over (sorted, so the bootstrap is reproducible), and the models to draw."""
    complete = {k: v for k, v in arms.items() if len(v) >= N_FULL}
    if subset == "full":
        qids = sorted(set.intersection(*(set(v) for v in complete.values())))
        models = [m for m in MODELS if any((s, m) in complete for s, _ in SYSTEMS)]
        return qids, models, complete
    # matched: intersect over all arms, incomplete ones included.
    qids = sorted(set.intersection(*(set(v) for v in arms.values())))
    models = [m for m in MODELS if any((s, m) in arms for s, _ in SYSTEMS)]
    return qids, models, arms


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def bar_kwargs(sid: str):
    """Matplotlib bar style of a system."""
    spec = style.SERIES[sid]
    return dict(facecolor=spec["fill"], edgecolor=spec["edge"],
                hatch=spec["hatch"], linewidth=style.EDGE_WIDTH)


def label_vertical(ax, x: float, v: float, renderer, bounds) -> None:
    """Value along the bar, starting below the interval so the whisker misses it."""
    lo, hi = bounds if bounds else (v, v)
    txt = ax.text(x, lo - VALUE_PAD, f"{v:.1f}", rotation=90,
                  rotation_mode="anchor", ha="right", va="center",
                  fontsize=VALUE_SIZE, color=style.INK_SECONDARY)
    inv = ax.transData.inverted()
    span = (inv.transform((0, txt.get_window_extent(renderer=renderer).height))[1]
            - inv.transform((0, 0))[1])
    if span + 2 * VALUE_PAD > lo:
        txt.set_y(hi + VALUE_PAD)
        txt.set_ha("left")


def render(subset: str, metric: str):
    """Draw and save the figure for one subset and metric; return the qids used."""
    style.apply()

    arms = load_all()
    qids, models, pool = select(arms, subset)
    n_max = max(sum(1 for s, _ in SYSTEMS if (s, m) in pool) for m in models)
    slot = (1.0 - GROUP_PAD) / n_max
    width = min(max(slot - BAR_GAP, slot * 0.6), BAR_WIDTH_MAX)

    # Sized like one panel of the AlzKB figure; legend above the title, as there.
    height_mm = 63
    fig, ax = plt.subplots(figsize=(78 * style.MM, height_mm * style.MM))
    fig.subplots_adjust(left=0.165, right=0.985, top=1 - 13.5 / height_mm, bottom=7 / height_mm)

    placed = []
    for gi, model in enumerate(models):
        present = [(s, lab) for s, lab in SYSTEMS if (s, model) in pool]
        for j, (sid, _) in enumerate(present):
            v, bounds = score(pool[(sid, model)], qids, metric)
            if v is None:
                continue
            x = gi + (j - (len(present) - 1) / 2) * slot
            ax.bar([x], [v], width=width, **bar_kwargs(sid))
            if bounds:
                ax.errorbar([x], [v],
                            yerr=[[max(0.0, v - bounds[0])], [max(0.0, bounds[1] - v)]],
                            fmt="none", ecolor=style.INK_SECONDARY,
                            elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=3)
            placed.append((x, v, bounds))

    ax.set_ylim(0, YMAX)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(METRIC_LABEL[metric])

    # Narrow no-break space as the thousands separator.
    count = f"{len(qids):,}".replace(",", " ")
    # Only the matched subset is flagged in the title.
    scope = "" if subset == "full" else " · matched subset, not a random sample"
    ax.set_title(f"BioHopR 2-hop · PrimeKG v2.1\nn = {count}{scope}",
                 fontsize=6.5, color=style.INK_SECONDARY, pad=3)

    # Two columns: four entries in one row do not fit a 78 mm figure.
    handles = [plt.Rectangle((0, 0), 1, 1, **bar_kwargs(sid)) for sid, _ in SYSTEMS]
    fig.legend(handles, [lab for _, lab in SYSTEMS], loc="upper center",
               bbox_to_anchor=(0.575, 1.0), ncol=2, fontsize=6.0, handlelength=1.1,
               handletextpad=0.5, columnspacing=1.4, labelspacing=0.35, borderaxespad=0.3)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for x, v, bounds in placed:
        label_vertical(ax, x, v, renderer, bounds)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"biohopr_results_{metric}" + ("" if subset == "full" else "_matched")
    for path in style.save(fig, os.path.join(OUT_DIR, stem)):
        print(f"wrote {path}")
    plt.close(fig)
    return qids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=("full", "matched"), default="full")
    ap.add_argument("--metric", choices=sorted(METRIC_LABEL),
                    help="default: exact_match and jaccard")
    args = ap.parse_args()

    metrics = [args.metric] if args.metric else ["exact_match", "jaccard"]
    for metric in metrics:
        render(args.subset, metric)
    print(f"font: {style.resolve_sans()}")


if __name__ == "__main__":
    main()

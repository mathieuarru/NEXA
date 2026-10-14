"""Shared style for the paper figures.

Target: two-column journal layout, 180 mm full text width, 7 pt sans-serif,
600 dpi PNG and TIFF output.

Palette: the five node colours of the NEXA logo
(nexa_frontend/nexa_logo/NEXA_network_crop.png), as pale fills with a uniform
near-black edge. Each system keeps one hue in every figure:

    LLM only            violet
    GraphCypherQAChain  yellow
    ESCARGOT            green (hatched for the Cypher-only ablation)
    NEXA Direct         blue
    NEXA Flex           orange

The assignment keeps adjacent bars distinguishable, also under simulated
colour-vision deficiency; the one weak pair (GraphCypherQAChain next to hatched
ESCARGOT) is told apart by the hatch. Fills have low contrast against white,
so every bar also carries its value label.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager

MM = 1 / 25.4
WIDTH_FULL = 180 * MM   # full text width, two-column layout

# --- ink -------------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- series ----------------------------------------------------------------
# fill = logo node colour; edge = uniform near-black outline; hatch = a variant
# of the same system.
EDGE = "#111111"
EDGE_WIDTH = 0.5

# Export resolution, used by savefig.dpi and save().
DPI = 600

# Dense hatching: with few lines, a small legend swatch can show a single
# diagonal and read as a struck-out box.
HATCH = "//////"

LOGO = {
    "violet": "#d6a4e0",   # top-left node
    "orange": "#f4a878",   # top-right node
    "blue":   "#8da6ea",   # centre node
    "green":  "#a9dda1",   # bottom-left node
    "yellow": "#f6e08c",   # bottom-right node
}

# One hue per system; the two ESCARGOT variants share a hue and differ by hatch.
HUES = {
    "llm_only": "violet",
    "graphcypher": "yellow",
    "escargot": "green",
    "nexa_direct": "blue",
    "nexa_flex": "orange",
}

SERIES = {
    "llm_only":      {"fill": LOGO[HUES["llm_only"]], "edge": EDGE, "hatch": None},
    "graphcypher":   {"fill": LOGO[HUES["graphcypher"]], "edge": EDGE, "hatch": None},
    "escargot_nofb": {"fill": LOGO[HUES["escargot"]], "edge": EDGE, "hatch": HATCH},
    "escargot":      {"fill": LOGO[HUES["escargot"]], "edge": EDGE, "hatch": None},
    "nexa_direct":   {"fill": LOGO[HUES["nexa_direct"]], "edge": EDGE, "hatch": None},
    "nexa_flex":     {"fill": LOGO[HUES["nexa_flex"]], "edge": EDGE, "hatch": None},
}

_SANS_PREFERENCE = [
    "Helvetica",
    "Arial",
    "Nimbus Sans",
    "TeX Gyre Heros",
    "Liberation Sans",
    "DejaVu Sans",
]


def resolve_sans() -> str:
    """First installed font of the sans-serif preference list."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _SANS_PREFERENCE:
        if name in available:
            return name
    return "DejaVu Sans"


def apply() -> str:
    """Install the rcParams and return the sans family actually resolved."""
    sans = resolve_sans()
    mpl.rcParams.update({
        # Embed TrueType fonts: most production pipelines reject Type 3.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "font.family": "sans-serif",
        "font.sans-serif": [sans],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,

        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,

        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,

        # Solid hairline grid, horizontal only: dashed lines read as thresholds.
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,

        "hatch.linewidth": 0.35,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handleheight": 0.9,
        "legend.columnspacing": 1.1,
        "legend.handletextpad": 0.45,

        "figure.dpi": 200,
        "savefig.dpi": DPI,
        # No tight cropping, so the exported width is exactly the figure width.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "savefig.facecolor": "white",
    })
    return sans


def save(fig, stem: str) -> list[str]:
    """Write `stem`.png and `stem`.tiff at 600 dpi from one rendering.

    The TIFF is flattened onto white (RGB, no alpha) and LZW-compressed, as
    journal submission systems expect.
    """
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches=None)
    with open(stem + ".png", "wb") as fh:
        fh.write(buf.getvalue())
    img = Image.open(io.BytesIO(buf.getvalue()))
    rgb = Image.new("RGB", img.size, "white")
    rgb.paste(img, mask=img.getchannel("A") if img.mode == "RGBA" else None)
    rgb.save(stem + ".tiff", dpi=(DPI, DPI), compression="tiff_lzw")
    return [stem + ".png", stem + ".tiff"]


def panel_label(ax, letter: str, dx: float = -0.055, dy: float = 1.0) -> None:
    """Bold panel letter, placed outside the axes at the top-left."""
    ax.text(
        dx, dy, letter,
        transform=ax.transAxes,
        fontsize=8, fontweight="bold", color=INK_PRIMARY,
        ha="left", va="bottom",
    )

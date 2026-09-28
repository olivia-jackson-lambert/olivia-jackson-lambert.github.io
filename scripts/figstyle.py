"""Shared matplotlib style for every case-study figure, so charts match the site.

Usage (from any figure script):

    import sys; sys.path.insert(0, "<repo>/scripts")
    from figstyle import apply, tidy, INK, SLATE, RUST, DIM, GRID, MUTED
    apply()
    ...
    tidy(ax)                                   # hairline axes, no top/right spines
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")

Rules the style encodes:
- Lora for all text, including small labels (MONO is kept as a name for older
  scripts but now also points at Lora, so every chart uses one face).
- No chart titles inside the figure; the caption in the article carries the title.
- White background, hairline light grid on the value axis only, no top/right spines.
- Colours come from the site palette below. Slate is the default series colour,
  rust is for the one thing the reader should notice, dim/grey for context.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONT_DIR = Path(__file__).resolve().parent / "fonts"

INK = "#16181b"      # text and primary lines
SLATE = "#7189a6"    # default series
RUST = "#c9503f"     # emphasis: the one thing to look at
DIM = "#3a3f48"      # secondary dark series
MUTED = "#8a8f98"    # notes, annotations, context series
GRID = "#e6e6ec"     # gridlines
LIGHT = "#c9ced6"    # light context fills
SEQ = [SLATE, RUST, DIM, "#a9b8ca", "#e3a397", MUTED]  # order for multi-series plots

SANS = "Lora"
MONO = "Lora"


def apply():
    for f in FONT_DIR.glob("*.ttf"):
        fm.fontManager.addfont(str(f))
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": SANS,
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.edgecolor": "#b9bec6",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SEQ),
        "xtick.color": "#5a606a",
        "ytick.color": "#5a606a",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2,
        "image.cmap": "Blues",
    })


def tidy(ax, grid_axis="y"):
    """Hairline finish for one axes."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)
    return ax


def note(ax, text, x=1.0, y=-0.16, ha="right"):
    """Small grey source or method note under an axes."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="top", fontsize=8.5, color=MUTED)

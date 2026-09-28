"""
Figures for the flight delay severity case study.

Both charts are redrawn from numbers hardcoded in the team's Databricks notebook
("Phase 3/Tree Models Comparison.py"): the six completed random forest CV grid
points, and the random forest blind-test (2019) confusion matrix counts. No model
is retrained and nothing is recomputed from raw data.

The two 4-panel tree-model figures and the pipeline diagrams need DBFS model
checkpoints or were drawn elsewhere, so they are not regenerated here.

Run from this folder:
    /Users/oliviajackson/Documents/portfolio/.venv/bin/python make_figures.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import apply, tidy, INK, SLATE, RUST, DIM, MUTED  # noqa: E402

OUT = HERE / "assets"
apply()


# ---------------------------------------------------------------------------
# 1. Stage 1 random forest grid: validation kappa_w vs severe recall
#    (200 trees, maxDepth 12, severe multiplier 1.5, 3-fold CV means)
# ---------------------------------------------------------------------------
grid = [
    # subset,    minInst, val kappa_w, val severe recall
    ("sqrt",     10, 0.2573, 0.5849),
    ("sqrt",     20, 0.2572, 0.5880),
    ("onethird", 10, 0.2502, 0.5694),
    ("onethird", 20, 0.2504, 0.5705),
    ("0.5",      10, 0.2508, 0.5650),
    ("0.5",      20, 0.2497, 0.5683),
]
colour = {"sqrt": RUST, "onethird": SLATE, "0.5": DIM}
marker = {10: "o", 20: "s"}

fig, ax = plt.subplots(figsize=(7.2, 4.4))
for subset, mi, k, r in grid:
    ax.scatter(k, r, s=90, marker=marker[mi], color=colour[subset],
               edgecolors="white", linewidths=1.2, zorder=3)

ax.axhline(0.50, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(0.2494, 0.503, "Gate 2 floor: 0.50 severe recall", color=MUTED,
        fontsize=9, va="bottom", ha="left")

ax.annotate("Selected: sqrt, minInstances 10", xy=(0.2573, 0.5849),
            xytext=(0.2548, 0.5775), fontsize=9.5, color=RUST, ha="center",
            arrowprops=dict(arrowstyle="-", color=RUST, lw=0.9,
                            shrinkA=2, shrinkB=6))

ax.set_xlim(0.2490, 0.2582)
ax.set_ylim(0.49, 0.60)
ax.set_xlabel("Validation quadratic-weighted kappa ($\\kappa_w$)")
ax.set_ylabel("Validation severe recall (2+ hr tier)")
tidy(ax)

handles = [
    Line2D([0], [0], marker="o", ls="", color=RUST, markersize=8, label="sqrt"),
    Line2D([0], [0], marker="o", ls="", color=SLATE, markersize=8, label="onethird"),
    Line2D([0], [0], marker="o", ls="", color=DIM, markersize=8, label="0.5"),
    Line2D([0], [0], marker="o", ls="", color=MUTED, markersize=7, label="minInstances 10"),
    Line2D([0], [0], marker="s", ls="", color=MUTED, markersize=7, label="minInstances 20"),
]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
          title="Feature subset", title_fontsize=9.5, alignment="left")
fig.savefig(OUT / "rf_cv_tradeoff.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Selected random forest (1.5x severe weight) on the 2019 blind test
# ---------------------------------------------------------------------------
cm = np.array([
    [4247091, 599695, 37324, 972378],   # true on time
    [304305, 283549, 76305, 234873],    # true 15-59 min
    [59833, 40228, 85655, 104471],      # true 1-2 hr
    [36954, 16593, 15783, 125282],      # true 2+ hr
])
share = cm / cm.sum(axis=1, keepdims=True)
tiers = ["On time", "15–59 min", "1–2 hr", "2+ hr"]


def fmt_count(v):
    return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"


cmap = LinearSegmentedColormap.from_list("slate", ["#ffffff", "#c9d3e0", SLATE, "#3d5470"])
fig, ax = plt.subplots(figsize=(6.4, 5.0))
im = ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)

for i in range(4):
    for j in range(4):
        dark = share[i, j] > 0.5
        c = "white" if dark else INK
        ax.text(j, i - 0.1, f"{share[i, j]:.1%}", ha="center", va="center",
                fontsize=11, color=c, fontweight="semibold")
        ax.text(j, i + 0.2, fmt_count(cm[i, j]), ha="center", va="center",
                fontsize=8.5, color="white" if dark else MUTED)

# The cell the decision rule cares about
ax.add_patch(Rectangle((2.5, 2.5), 1, 1, fill=False, ec=RUST, lw=2.2))

ax.set_xticks(range(4), tiers)
ax.set_yticks(range(4), tiers)
ax.xaxis.set_ticks_position("bottom")
ax.set_xlabel("Predicted tier")
ax.set_ylabel("True tier")
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Share of flights in the true tier")
cb.outline.set_visible(False)
cb.ax.tick_params(length=0)
cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

fig.savefig(OUT / "rf_confusion_matrix.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("wrote rf_cv_tradeoff.png, rf_confusion_matrix.png")

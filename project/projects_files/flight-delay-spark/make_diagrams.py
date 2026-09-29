"""
Diagrams and panel re-grids for the flight delay severity case study.

1. workflow_pipeline.png, rf_pipeline_v2.png, xgb_pipelinev2.png
   Redrawn top to bottom in the site style. Content is transcribed from the
   team's original diagrams. Two corrections come from the team's final report
   ("Phase 3/Phase 3 - Final Report.py"):
   - Random forest: the severe weight was chosen by re-running 3-fold CV at
     m3 in {1.0, 1.2, 1.5}, then the winner was retrained on the full window.
     The old diagram labelled this whole step "Final Training".
   - XGBoost: the blind-tested model is max_depth 5, n_estimators 100, m3 1.0,
     as the old diagram said. Kept as is.

2. tree_models_confusion_matrices_4panel.png, tree_models_feature_importance_4panel.png
   The numbers behind these live only in DBFS model checkpoints, so the
   existing 1x4 images are cropped panel by panel and re-pasted as 2x2 grids.
   No pixel content changes; only the old in-image super-title is dropped.
   The step skips any image that is already a 2x2 grid, so re-running is safe.

Run from this folder:
    /Users/oliviajackson/Documents/portfolio/.venv/bin/python make_diagrams.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import (apply, INK, SLATE, RUST, DIM, MUTED, LIGHT,  # noqa: E402
                      SLATE_TINT, RUST_TINT, PAPER)

OUT = HERE / "assets"
apply()
# Lora has no subscript, arrow or set glyphs; render those through mathtext in Lora
plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Lora",
                     "mathtext.it": "Lora", "mathtext.bf": "Lora:semibold"})

TITLE_PT, SUB_PT = 9.8, 8.4
TITLE_LH, SUB_LH = 0.19, 0.165   # line heights in inches


# ---------------------------------------------------------------------------
# Drawing helpers (axes in inches, y grows downward)
# ---------------------------------------------------------------------------
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, title, subs=(), fill=PAPER, edge=LIGHT, lw=0.9,
        title_color=INK, sub_color=DIM, chip=None, dashed=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.07",
                                fc=fill, ec=edge, lw=lw,
                                ls=(0, (3, 2)) if dashed else "-"))
    tlines = title.split("\n") if title else []
    total = len(tlines) * TITLE_LH + len(subs) * SUB_LH + (0.3 if chip else 0)
    y = cy - total / 2
    for t in tlines:
        ax.text(cx, y + TITLE_LH / 2, t, ha="center", va="center",
                fontsize=TITLE_PT, fontweight="semibold", color=title_color)
        y += TITLE_LH
    for s in subs:
        ax.text(cx, y + SUB_LH / 2, s, ha="center", va="center",
                fontsize=SUB_PT, color=sub_color)
        y += SUB_LH
    if chip:
        ax.text(cx, y + 0.17, chip, ha="center", va="center", fontsize=7.8,
                fontweight="semibold", color=INK,
                bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.25",
                          fc="white", ec=SLATE, lw=0.8))


def arrow(ax, p, q, color=SLATE, dashed=False, lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 ls=(0, (3, 2)) if dashed else "-"))


def group(ax, x0, y0, x1, y1, label):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc="white", ec=MUTED, lw=0.8, ls=(0, (3, 2))))
    ax.text((x0 + x1) / 2, y0, label, ha="center", va="center", fontsize=8,
            style="italic", color=MUTED,
            bbox=dict(fc="white", ec="none", pad=2))


def note(ax, x, y, lines):
    for i, t in enumerate(lines):
        ax.text(x, y + i * 0.18, t, ha="left", va="top", fontsize=7.8,
                style="italic", color=MUTED)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.08,
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1a. Project workflow
# ---------------------------------------------------------------------------
W, H = 7.0, 7.75
fig, ax = canvas(W, H)
cx = W / 2

y_raw, y_prep, y_fe, y_split, y_mod, y_eval = 0.45, 1.75, 3.05, 4.2, 5.75, 7.3
box(ax, cx, y_raw, 3.2, 0.72, "Raw OTPW Dataset",
    ["31M Flights, 214 Columns, 2015–2019"])

prep = [(1.2, "Clean &\nDeduplicate", ["Drop Cancelled or", "Diverted Flights"]),
        (3.5, "Column\nScoping", ["Drop High-Null and", "Target Leakage Columns"]),
        (5.8, "Label\nConstruction", ["4 Delay Severity", "Tiers (0–3)"])]
for x, t, s in prep:
    box(ax, x, y_prep, 2.1, 1.05, t, s, fill=SLATE_TINT, edge=SLATE, lw=0.7)
    arrow(ax, (cx, y_raw + 0.36), (x, y_prep - 0.525))
    arrow(ax, (x, y_prep + 0.525), (cx, y_fe - 0.33))

box(ax, cx, y_fe, 2.6, 0.66, "Feature Engineering", ["84 Features"],
    fill=SLATE_TINT, edge=SLATE, lw=0.7)
arrow(ax, (cx, y_fe + 0.33), (cx, y_split - 0.45))
box(ax, cx, y_split, 3.3, 0.9, "Temporal Split",
    ["Train: 2015–2018 (3-Fold Sliding-Window CV)", "Test: 2019"])

models = [("Binomial\nLogistic\nRegression", ["(Feasibility Check)"]),
          ("Multinomial\nLR", ["(Baseline)"]),
          ("Random\nForest", []),
          ("Gradient\nBoosted\nDecision Trees", []),
          ("Multi-Layer\nPerceptron", ["(Neural Network)"])]
mw, gap = 1.26, 0.1
x0 = cx - (5 * mw + 4 * gap) / 2 + mw / 2
for i, (t, s) in enumerate(models):
    x = x0 + i * (mw + gap)
    box(ax, x, y_mod, mw, 1.1, t, s, fill=SLATE_TINT, edge=SLATE, lw=0.7)
    arrow(ax, (cx, y_split + 0.45), (x, y_mod - 0.55))
    arrow(ax, (x, y_mod + 0.55), (cx, y_eval - 0.36))

box(ax, cx, y_eval, 2.6, 0.72, "Evaluation", ["Gate-Aware Decision Rule"],
    fill=RUST_TINT, edge=RUST, lw=1.1)
save(fig, "workflow_pipeline.png")


# ---------------------------------------------------------------------------
# 1b / 1c. Model pipelines: preprocessing and model on the left, tuning on the right
# ---------------------------------------------------------------------------
def model_pipeline(name, group_label, clf_title, clf_subs, clf_key,
                   tuning, notes, height):
    fig, ax = canvas(7.0, height)
    lx, rx, bw = 1.75, 5.15, 2.7

    ax.text(lx, 0.12, "Pipeline", ha="center", va="center", fontsize=9,
            fontweight="semibold", color=MUTED)
    ax.text(rx, 0.12, "Tuning and Training", ha="center", va="center",
            fontsize=9, fontweight="semibold", color=MUTED)

    ys = [0.7, 1.95, 2.8, 3.65, 4.8, 5.85]
    box(ax, lx, ys[0], bw, 0.62, "Feature Checkpoint", ["73 Features"])
    group(ax, lx - bw / 2 - 0.12, 1.35, lx + bw / 2 + 0.12, 4.18, group_label)
    box(ax, lx, ys[1], bw - 0.3, 0.6, "Imputer", ["Median · 66 Numeric"],
        fill=SLATE_TINT, edge=SLATE, lw=0.7)
    box(ax, lx, ys[2], bw - 0.3, 0.6, "String Indexer", ["7 Categoricals"],
        fill=SLATE_TINT, edge=SLATE, lw=0.7)
    box(ax, lx, ys[3], bw - 0.3, 0.6, "Vector Assembler", ["73 $\\rightarrow$ 1 Vector"],
        fill=SLATE_TINT, edge=SLATE, lw=0.7)
    if clf_key:
        box(ax, lx, ys[4], bw, 0.72, clf_title, clf_subs, fill=RUST_TINT,
            edge=RUST, lw=1.1)
    else:
        box(ax, lx, ys[4], bw, 0.72, clf_title, clf_subs, fill=SLATE_TINT,
            edge=SLATE, lw=0.7)
    box(ax, lx, ys[5], bw, 0.62, "Evaluation", ["Blind Test 2019"])
    arrow(ax, (lx, ys[0] + 0.31), (lx, ys[1] - 0.3))
    arrow(ax, (lx, ys[1] + 0.3), (lx, ys[2] - 0.3))
    arrow(ax, (lx, ys[2] + 0.3), (lx, ys[3] - 0.3))
    arrow(ax, (lx, ys[3] + 0.3), (lx, ys[4] - 0.36))
    arrow(ax, (lx, ys[4] + 0.36), (lx, ys[5] - 0.31))

    for i, step in enumerate(tuning):
        box(ax, rx, step["y"], bw, step["h"], step["title"], step["subs"],
            fill=PAPER if i < len(tuning) - 1 else SLATE_TINT,
            edge=LIGHT if i < len(tuning) - 1 else SLATE, lw=0.8,
            chip=step.get("chip"))
        if i:
            prev = tuning[i - 1]
            p = (rx, prev["y"] + prev["h"] / 2)
            q = (rx, step["y"] - step["h"] / 2)
            link = step.get("link")
            col = RUST if step.get("broken") else SLATE
            arrow(ax, p, q, color=col, dashed=step.get("broken", False))
            if link:
                my = (p[1] + q[1]) / 2
                for k, t in enumerate(link):
                    ax.text(rx + 0.12, my + (k - (len(link) - 1) / 2) * 0.17, t,
                            ha="left", va="center", fontsize=8,
                            style="italic", color=col)
    note(ax, 0.4, height - 0.55, notes)
    save(fig, name)


model_pipeline(
    "rf_pipeline_v2.png",
    "Preprocessing Sub-Pipeline",
    "Random Forest Classifier", ["200 Trees · Depth 12"], True,
    [
        dict(y=0.85, h=1.0, title="Stage 1 · Structure Search",
             subs=["6 of 18 Configs × 3 Folds", "Severe Weight $m_3$ = 1.5"],
             chip="CV Folds"),
        dict(y=2.65, h=1.0, title="Stage 2 · Weight Selection",
             subs=["$m_3 \\in \\{1.0, 1.2, 1.5\\}$ × 3 Folds", "Best Structure Only"],
             chip="CV Folds", link=["Best Config"]),
        dict(y=4.45, h=1.0, title="Final Training",
             subs=["$m_3$ = 1.5 · 200 Trees · Depth 12"],
             chip="Full Train 2015–2018 · 23.9M", link=["$m_3$ = 1.5 Selected"]),
    ],
    ["All Preprocessing Statistics and Class Weights Are Fit on Training Data Only.",
     "2019 Is Never Seen During Training or Tuning."],
    height=6.95,
)

model_pipeline(
    "xgb_pipelinev2.png",
    "Shared Preprocessing Sub-Pipeline",
    "SparkXGB Classifier", ["GPU · multi:softprob", "Sequential Boosting"], False,
    [
        dict(y=0.95, h=1.2, title="Stage 1 · Hyperparameter Search",
             subs=["4 of 8 Configs × 3 Folds", "Early Stopping",
                   "Severe Boost $m_3$ = 1.5"],
             chip="CV Folds"),
        dict(y=3.6, h=1.2, title="Stage 2 · Final Training",
             subs=["max_depth = 5 · n_estimators = 100",
                   "No Severe Boost, $m_3$ = 1.0"],
             chip="Full Train 2015–2018 · 23.9M", broken=True,
             link=["Selected Config Did Not", "Complete; Reduced Model",
                   "Trained Instead"]),
    ],
    ["Training Runs as a Spark Barrier Stage: All Workers Must Survive the Entire Fit,",
     "so an Interrupted Run Restarts from Boosting Round Zero. All Preprocessing Statistics",
     "and Class Weights Are Fit on Training Data Only; 2019 Is Never Seen in Training or Tuning."],
    height=7.2,
)


# ---------------------------------------------------------------------------
# 2. Re-grid the two 1x4 panel figures into 2x2 (crop and paste only)
# ---------------------------------------------------------------------------
# (file, body top row, panel column cuts) measured from whitespace in the 1x4 originals
REGRID = {
    "tree_models_confusion_matrices_4panel.png": (90, [0, 843, 1655, 2467, None]),
    "tree_models_feature_importance_4panel.png": (95, [0, 898, 1792, 2687, None]),
}
for fname, (top, cuts) in REGRID.items():
    im = Image.open(OUT / fname).convert("RGB")
    w, h = im.size
    if w / h < 2:          # already re-gridded
        continue
    cuts = [c if c is not None else w for c in cuts]
    panels = [im.crop((cuts[i], top, cuts[i + 1], h)) for i in range(4)]
    pw = max(p.width for p in panels)
    ph = max(p.height for p in panels)
    gx, gy, pad = 40, 50, 20
    grid = Image.new("RGB", (2 * pw + gx + 2 * pad, 2 * ph + gy + 2 * pad), "white")
    for i, p in enumerate(panels):
        r, c = divmod(i, 2)
        # right-align so the plotting areas line up in each column
        x = pad + c * (pw + gx) + (pw - p.width)
        y = pad + r * (ph + gy)
        grid.paste(p, (x, y))
    grid.save(OUT / fname)

print("wrote workflow_pipeline.png, rf_pipeline_v2.png, xgb_pipelinev2.png, "
      "and 2x2 re-grids of the two 4-panel figures")

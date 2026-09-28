"""
Figures for the Mars rover image classification project.

Everything is computed from the team's real Stage 3 test predictions
(stage_3_iter5_test_predictions.csv, 1,305 held-out images) and the real
MSL image set. No numbers are recomputed from a retrained model.

Styling comes from the site's shared scripts/figstyle.py. Figures carry no
titles; the captions in mars-terrain.qmd do that job.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, classification_report

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import apply, tidy, INK, SLATE, RUST, DIM, MUTED, LIGHT  # noqa: E402

SRC = Path("/Users/oliviajackson/Downloads/207_SWEOP_Project/Code")
OUT = HERE / "assets"

CLS = ['APXS','APXS Cal Target','Chemcam Cal Target','Chemin Inlet Open','Drill','Drill Holes',
       'DRT Front','DRT Side','Ground','Horizon','Inlet','MAHLI','MAHLI Cal Target','Mastcam',
       'Mastcam Cal Target','Observation Tray','Portion Box','Portion Tube','Portion Tube Opening',
       'REMS UV Sensor','Rover Rear Deck','Scoop','Sun','Turret','Wheel']

# Dust robustness, measured in the team's noise notebook. The drops are taken
# from the notebook's unrounded values, so they differ from subtracting the
# rounded figures by up to 0.0001.
DUST = {"clean_acc": 0.7648, "dust_acc": 0.7226, "clean_f1": 0.6656, "dust_f1": 0.6371}
DROP = {"acc": 0.0421, "f1": 0.0285}


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig); print("  wrote", name)


def bar_labels(ax, bars, vals):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .015, f"{v:.4f}", ha="center",
                va="bottom", fontsize=10, color=INK)


def main():
    apply(); OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC / "model_outputs/stage_3_iter5_test_predictions.csv")
    present = sorted(df.y_true.unique())

    # ---------- 1. example image per class ----------
    def square(img, size=260):
        """Centre-crop to square then resize, so every cell matches."""
        w, h = img.size
        m = min(w, h)
        img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
        return img.resize((size, size), Image.LANCZOS)

    picks, missing = [], []
    for c in present:
        path = None
        for fn in df[df.y_true == c]["filename"]:
            cand = SRC / "msl-images" / fn
            if cand.exists():
                path = cand; break
        if path is None:
            missing.append(CLS[c])
        else:
            picks.append((CLS[c], square(Image.open(path).convert("RGB"))))
    if missing:
        print("   no image on disk for:", ", ".join(missing))

    ncol = 5
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol, 2.1 * nrow))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (name, img) in zip(np.ravel(axes), picks):
        ax.imshow(img)
        ax.set_title(name, fontsize=8.8, pad=4, color=INK)
    fig.tight_layout(h_pad=1.0, w_pad=0.5)
    save(fig, "class_examples.png")

    # ---------- 2. F1 against class size ----------
    from scipy.stats import spearmanr
    rep = classification_report(df.y_true, df.y_pred, labels=present,
                                target_names=[CLS[c] for c in present],
                                output_dict=True, zero_division=0)
    sup = np.array([rep[CLS[c]]["support"] for c in present], float)
    f1s = np.array([rep[CLS[c]]["f1-score"] for c in present])
    is_cal = np.array(["Cal Target" in CLS[c] for c in present])
    rho, pval = spearmanr(sup, f1s)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.scatter(sup[~is_cal], f1s[~is_cal], s=60, color=SLATE, zorder=3,
               edgecolor="white", linewidth=.6, label="Instruments and terrain")
    ax.scatter(sup[is_cal], f1s[is_cal], s=70, color=RUST, marker="D", zorder=3,
               edgecolor="white", linewidth=.6, label="Calibration targets")
    ax.set_xscale("log")
    ticks = [2, 5, 10, 20, 50, 100, 200, 300]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Test images in the class (log scale)")
    ax.set_ylabel("F1 score on held-out images")
    ax.set_ylim(0, 1.09)
    ax.set_xlim(sup.min() * 0.62, sup.max() * 2.15)   # room for edge labels

    # Greedy label placement: try offsets in order, keep the first that does
    # not overlap an already-placed label or sit outside the axes.
    to_label = [(x, y, CLS[c]) for x, y, c in zip(sup, f1s, present)
                if y < .45 or "Cal Target" in CLS[c] or x > 200]
    candidates = [(8, 4, "left"), (8, -12, "left"), (-8, 4, "right"), (-8, -12, "right"),
                  (0, 11, "center"), (0, -16, "center"), (8, 12, "left"), (-8, 12, "right")]
    fig.canvas.draw()
    placed = []
    # Hand-placed labels where the greedy search would cross a neighbouring marker.
    fixed = {"APXS Cal Target": (-9, 6, "right"), "Chemcam Cal Target": (9, 6, "left"),
             "Mastcam Cal Target": (10, 22, "left"), "MAHLI Cal Target": (10, -14, "left")}
    arrow = dict(arrowstyle="-", color=LIGHT, lw=.7, shrinkA=0, shrinkB=4)
    for x, y, name in sorted(to_label, key=lambda t: -t[1]):
        for dx, dy, ha in ([fixed[name]] if name in fixed else candidates):
            txt = ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                              ha=ha, fontsize=8.5, color=DIM, arrowprops=arrow)
            fig.canvas.draw()
            bb = txt.get_window_extent()
            ax_bb = ax.get_window_extent()
            inside = ax_bb.x0 <= bb.x0 and bb.x1 <= ax_bb.x1 and bb.y1 <= ax_bb.y1
            clash = any(bb.overlaps(q) for q in placed)
            if name in fixed or (inside and not clash):
                placed.append(bb.expanded(1.06, 1.35))
                break
            txt.remove()
        else:
            txt = ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 4),
                              fontsize=8.5, color=DIM, arrowprops=arrow)
            fig.canvas.draw(); placed.append(txt.get_window_extent())
    tidy(ax, grid_axis="both")
    ax.legend(loc="lower right")
    ax.text(0.02, 0.04, f"Spearman rank correlation {rho:+.2f}, p = {pval:.2f}",
            transform=ax.transAxes, fontsize=9, color=MUTED,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    fig.tight_layout()
    save(fig, "f1_vs_class_size.png")

    # ---------- 3. confusion matrix, present classes only ----------
    cm = confusion_matrix(df.y_true, df.y_pred, labels=present).astype(float)
    cmn = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(8.8, 7.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    lab = [CLS[c] for c in present]
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8.5)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Share of the true class's test images", fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=0)
    fig.tight_layout()
    save(fig, "confusion_matrix.png")

    # ---------- 4. dust robustness ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, (ck, dk, drop, ylab) in zip(
            (a1, a2),
            (("clean_acc", "dust_acc", DROP["acc"], "Test accuracy"),
             ("clean_f1", "dust_f1", DROP["f1"], "Test macro-F1"))):
        vals = [DUST[ck], DUST[dk]]
        bars = ax.bar(["Clean", "Simulated dust"], vals, color=[SLATE, RUST], width=.55)
        bar_labels(ax, bars, vals)
        ax.set_ylim(0, 0.9)
        ax.set_ylabel(ylab)
        ax.text(0.5, 0.97, f"Drop of {drop:.4f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5, color=MUTED)
        tidy(ax)
    fig.tight_layout(w_pad=3)
    save(fig, "dust_robustness.png")

    # ---------- 5. macro-F1 denominator ----------
    variants = [
        ("22 classes\npresent in test", f1_score(df.y_true, df.y_pred, labels=present, average="macro", zero_division=0)),
        ("24 classes\n(as reported)", f1_score(df.y_true, df.y_pred, average="macro", zero_division=0)),
        ("All 25\nlabel indices", f1_score(df.y_true, df.y_pred, labels=list(range(25)), average="macro", zero_division=0)),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    vals = [v[1] for v in variants]
    bars = ax.bar([v[0] for v in variants], vals, color=[SLATE, RUST, LIGHT], width=.55)
    bar_labels(ax, bars, vals)
    ax.set_ylim(0, 0.85); ax.set_ylabel("Test macro-F1")
    tidy(ax)
    fig.tight_layout()
    save(fig, "macro_f1_denominator.png")


if __name__ == "__main__":
    main()

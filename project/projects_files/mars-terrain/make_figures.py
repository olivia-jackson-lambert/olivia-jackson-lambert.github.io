"""
Figures for the Mars rover image classification project.

Everything is computed from the team's real Stage 3 test predictions
(stage_3_iter5_test_predictions.csv, 1,305 held-out images) and the real
MSL image set. No numbers are recomputed from a retrained model.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, classification_report

SRC = Path("/Users/oliviajackson/Downloads/207_SWEOP_Project/Code")
FONTS = Path("/Users/oliviajackson/Documents/portfolio/projects/high-energy-particle-classifier/assets/fonts")
OUT = Path("assets")
SLATE, RUST, INK, GRID = "#3A506B", "#9E2A2B", "#1b1b1f", "#e6e6ec"
TITLE_PAD = 16

CLS = ['APXS','APXS Cal Target','Chemcam Cal Target','Chemin Inlet Open','Drill','Drill Holes',
       'DRT Front','DRT Side','Ground','Horizon','Inlet','MAHLI','MAHLI Cal Target','Mastcam',
       'Mastcam Cal Target','Observation Tray','Portion Box','Portion Tube','Portion Tube Opening',
       'REMS UV Sensor','Rover Rear Deck','Scoop','Sun','Turret','Wheel']

# Dust robustness, measured in the team's noise notebook
DUST = {"clean_acc": 0.7648, "dust_acc": 0.7226, "clean_f1": 0.6656, "dust_f1": 0.6371}


def house_style():
    for f in ("Lora-Regular.ttf", "Lora-SemiBold.ttf"):
        if (FONTS / f).exists():
            fm.fontManager.addfont(str(FONTS / f))
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "font.family": "Lora" if (FONTS / "Lora-Regular.ttf").exists() else "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10,
    })


def tidy(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c9c9d2")
    ax.tick_params(colors=INK, labelsize=9)


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig); print("  wrote", name)


def main():
    house_style(); OUT.mkdir(parents=True, exist_ok=True)
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
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.85 * ncol, 2.06 * nrow))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (name, img) in zip(np.ravel(axes), picks):
        ax.imshow(img)
        ax.set_title(name, fontsize=8.4, pad=4, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Curiosity Image Classes", fontsize=13, fontweight="semibold", color=INK, y=1.005)
    fig.text(0.5, -0.004,
             "Calibration targets are built to be unmistakable. The instruments are not.",
             ha="center", fontsize=8.6, color="#8a8f98", style="italic")
    fig.tight_layout(h_pad=1.0, w_pad=0.4)
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
    ax.scatter(sup[~is_cal], f1s[~is_cal], s=64, color=SLATE, zorder=3, label="Instruments and terrain")
    ax.scatter(sup[is_cal], f1s[is_cal], s=74, color=RUST, marker="D", zorder=3, label="Calibration targets")
    ax.set_xscale("log")
    ax.set_xlabel("Test examples in the class (log scale)")
    ax.set_ylabel("F1 score")
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
    for x, y, name in sorted(to_label, key=lambda t: -t[1]):
        for dx, dy, ha in candidates:
            txt = ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                              ha=ha, fontsize=8.2, color=INK,
                              arrowprops=dict(arrowstyle="-", color="#9aa1ab",
                                              lw=.7, shrinkA=0, shrinkB=4))
            fig.canvas.draw()
            bb = txt.get_window_extent()
            ax_bb = ax.get_window_extent()
            inside = ax_bb.x0 <= bb.x0 and bb.x1 <= ax_bb.x1 and bb.y1 <= ax_bb.y1
            clash = any(bb.overlaps(q) for q in placed)
            if inside and not clash:
                placed.append(bb.expanded(1.06, 1.35))
                break
            txt.remove()
        else:
            txt = ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 4),
                              fontsize=8.2, color=INK,
                              arrowprops=dict(arrowstyle="-", color="#9aa1ab",
                                              lw=.7, shrinkA=0, shrinkB=4))
            fig.canvas.draw(); placed.append(txt.get_window_extent())
    ax.set_title("Class Size Does Not Predict Performance", pad=TITLE_PAD, color=INK)
    ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right"); tidy(ax)
    ax.text(0.02, 0.04, f"Spearman rho = {rho:+.2f}, p = {pval:.2f}",
            transform=ax.transAxes, fontsize=9, color="#8a8f98", style="italic")
    fig.tight_layout()
    save(fig, "f1_vs_class_size.png")

    # ---------- 3. confusion matrix, present classes only ----------
    cm = confusion_matrix(df.y_true, df.y_pred, labels=present).astype(float)
    cmn = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    lab = [CLS[c] for c in present]
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Stage 3 Confusion Matrix (Normalized)", pad=TITLE_PAD, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=INK)
    fig.tight_layout()
    save(fig, "confusion_matrix.png")

    # ---------- 4. dust robustness ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, (ck, dk, title, ylab) in zip(
            (a1, a2),
            (("clean_acc", "dust_acc", "Accuracy", "Test accuracy"),
             ("clean_f1", "dust_f1", "Macro-F1", "Macro-F1"))):
        vals = [DUST[ck], DUST[dk]]
        bars = ax.bar(["Clean", "Simulated dust"], vals, color=[SLATE, RUST], width=.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.4f}", ha="center",
                    fontsize=10, color=INK)
        drop = vals[0] - vals[1]
        ax.set_ylim(0, max(vals) * 1.28)
        ax.set_title(f"{title}: {drop:.4f} drop", pad=TITLE_PAD, color=INK)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True); tidy(ax)
    fig.suptitle("Robustness to Simulated Dust on the Optics", fontsize=13,
                 fontweight="semibold", color=INK, y=1.01)
    fig.tight_layout(w_pad=2.4)
    save(fig, "dust_robustness.png")

    # ---------- 5. macro-F1 denominator ----------
    variants = [
        ("22 classes\npresent in test", f1_score(df.y_true, df.y_pred, labels=present, average="macro", zero_division=0)),
        ("24 classes\n(as reported)", f1_score(df.y_true, df.y_pred, average="macro", zero_division=0)),
        ("all 25\nclasses", f1_score(df.y_true, df.y_pred, labels=list(range(25)), average="macro", zero_division=0)),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar([v[0] for v in variants], [v[1] for v in variants],
                  color=[SLATE, RUST, "#b6bcc6"], width=.55)
    for b, (_, v) in zip(bars, variants):
        ax.text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.4f}", ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, 0.88); ax.set_ylabel("Macro-F1")
    ax.set_title("The Same Model, Three Macro-F1 Values", pad=TITLE_PAD, color=INK)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True); tidy(ax)
    ax.text(0.5, -0.24, "Drill Holes, Sun and Turret have no test examples. Including them\n"
                        "as zeros costs roughly 0.09 of macro-F1.",
            transform=ax.transAxes, ha="center", fontsize=8.6, color="#8a8f98", style="italic")
    fig.tight_layout()
    save(fig, "macro_f1_denominator.png")


if __name__ == "__main__":
    main()

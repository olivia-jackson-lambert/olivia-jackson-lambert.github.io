"""
Figures for the particle classifier case study, in the site style (scripts/figstyle.py).

Inputs stay in the source project and are never copied into this repo:
  data/particle_images.npz, data/particle_truth_array.npy, models/lr_1e-3_baseline_history.csv
Test-set predictions come from a cache written by running the two saved Keras models
(models/initial_model.keras, models/lr_1e-3_baseline.keras) over the held-out split.
Pass it with --preds; without it the confusion matrices and prediction grid are skipped.

Class order follows the one-hot encoder in the training notebook, which sorts the PDG
codes: 11 electron, 13 muon, 22 photon, 211 pion, 2212 proton. The notebook's
particle_names list did not match this order, which is why older figures were mislabelled.

    python make_figures.py --preds /path/to/preds.npz
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import apply, tidy, INK, SLATE, RUST, DIM, GRID, MUTED, MONO  # noqa: E402

SRC = Path("/Users/oliviajackson/Documents/portfolio/projects/high-energy-particle-classifier")
ASSETS = HERE / "assets"

CODES = [11, 13, 22, 211, 2212]
NAMES = ["Electron", "Muon", "Photon", "Pion", "Proton"]
PLANES = ["XY plane", "YZ plane", "ZX plane"]
DETECTOR_CMAP = "inferno"
GROUND = "#0b0b0b"

# Per-epoch log of the initial model (notebook cell output, 5 epochs at lr 1e-3).
INIT_HISTORY = pd.DataFrame({
    "epoch": [1, 2, 3, 4, 5],
    "cat_acc": [0.6639, 0.7219, 0.7634, 0.8004, 0.8273],
    "loss": [0.7107, 0.5818, 0.5292, 0.4723, 0.4277],
    "val_cat_acc": [0.6096, 0.7472, 0.8008, 0.8344, 0.5362],
    "val_loss": [0.7340, 0.6209, 0.4950, 0.4219, 1.1053],
})


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print("saved", name)


def fig_to_frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames, name, ms=1800):
    w = max(f.width for f in frames); h = max(f.height for f in frames)
    canvas = [Image.new("RGB", (w, h), "white") for _ in frames]
    for c, f in zip(canvas, frames):
        c.paste(f, (0, 0))
    pal = [c.quantize(colors=255, method=Image.Quantize.MEDIANCUT) for c in canvas]
    pal[0].save(ASSETS / name, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)
    print("saved", name)


def load_data():
    X = sp.load_npz(SRC / "data/particle_images.npz")
    truth = np.load(SRC / "data/particle_truth_array.npy")
    train_idx, test_idx = train_test_split(np.arange(len(truth)), train_size=50000, random_state=11)
    return X, truth, train_idx, test_idx


def planes(X, i):
    return X.getrow(int(i)).toarray().reshape(256, 256, 3).transpose(2, 0, 1)


def detector_axes(ax):
    ax.set_facecolor(GROUND)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#d0d3d9"); s.set_linewidth(0.6)


def show_plane(ax, img):
    """Draw one projection. A 2 x 2 max pool (display only) keeps one-pixel tracks visible at page size."""
    detector_axes(ax)
    img = img.reshape(128, 2, 128, 2).max(axis=(1, 3))
    nz = img[img > 0]
    vmax = np.percentile(nz, 98) if nz.size else 1.0
    ax.imshow(img, cmap=DETECTOR_CMAP, vmin=0, vmax=vmax, interpolation="nearest")


# 1. Example tracks per class across the momentum range (animated, one frame per class)
def particle_types(X, truth, train_idx):
    frames = []
    for code, name in zip(CODES, NAMES):
        idx = train_idx[truth[train_idx, 0].astype(int) == code]
        order = idx[np.argsort(truth[idx, 1])]
        picks = order[np.linspace(0, len(order) - 1, 5, dtype=int)]
        fig, axes = plt.subplots(3, 5, figsize=(10, 6.6), gridspec_kw=dict(wspace=0.06, hspace=0.06))
        for c, i in enumerate(picks):
            imgs = planes(X, i)
            for r in range(3):
                ax = axes[r, c]
                show_plane(ax, imgs[r])
                if c == 0:
                    ax.set_ylabel(PLANES[r], fontsize=10, color=DIM, labelpad=6)
                if r == 0:
                    ax.set_title(f"{truth[i, 1]:,.0f} MeV", fontsize=9.5, color=DIM, pad=5, fontfamily=MONO)
        fig.text(0.125, 0.955, name, fontsize=15, fontweight="semibold", color=INK, ha="left")
        fig.text(0.9, 0.955, "Initial momentum increases left to right", fontsize=9.5, color=MUTED, ha="right")
        frames.append(fig_to_frame(fig))
    save_gif(frames, "particle_types.gif")


# 2. Truth feature distributions per class (animated, one frame per class)
def truth_arrays(truth, train_idx):
    labels = ["Total momentum (MeV)", "$p_x$ (MeV)", "$p_y$ (MeV)", "$p_z$ (MeV)",
              "Production $x$ (cm)", "Production $y$ (cm)", "Production $z$ (cm)"]
    cols = range(1, 8)
    tr = truth[train_idx]
    lims = {c: np.percentile(tr[:, c], [1, 99]) for c in cols}
    bins = {c: np.linspace(*lims[c], 41) for c in cols}
    ymax = max(np.histogram(tr[tr[:, 0].astype(int) == code, c], bins=bins[c])[0].max()
               for code in CODES for c in cols) * 1.08
    frames = []
    for code, name in zip(CODES, NAMES):
        sub = tr[tr[:, 0].astype(int) == code]
        fig, axes = plt.subplots(1, 7, figsize=(15, 2.7), sharey=True, gridspec_kw=dict(wspace=0.12))
        for ax, c, lab in zip(axes, cols, labels):
            ax.hist(sub[:, c], bins=bins[c], color=SLATE, edgecolor="none")
            ax.set_xlim(*lims[c]); ax.set_ylim(0, ymax)
            ax.set_xlabel(lab, fontsize=9.5)
            ax.tick_params(labelsize=8.5)
            tidy(ax)
        axes[0].set_ylabel("Events")
        fig.text(0.125, 1.0, name, fontsize=14, fontweight="semibold", color=INK, ha="left", va="bottom")
        fig.text(0.9, 1.0, f"n = {len(sub):,} training events", fontsize=9, color=MUTED,
                 ha="right", va="bottom", fontfamily=MONO)
        frames.append(fig_to_frame(fig))
    save_gif(frames, "truth_arrays.gif")


# 3. Architecture diagram
def architecture():
    rows = [
        ("Input", "Flattened image, 196,608 values"),
        ("Reshape", "256 × 256 × 3 (one channel per projection)"),
        ("Conv block 1", "16 filters, stride 2  ·  2 × (Conv, BN, ReLU)  ·  MaxPool  ·  Dropout 0.10"),
        ("Conv block 2", "32 filters  ·  2 × (Conv, BN, ReLU)  ·  MaxPool  ·  Dropout 0.15"),
        ("Conv block 3", "64 filters  ·  2 × (Conv, BN, ReLU)  ·  MaxPool  ·  Dropout 0.20"),
        ("Head", "Global average pool  ·  Dense 64  ·  Dropout 0.25  ·  Softmax over 5 classes"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, len(rows)); ax.axis("off")
    h = 0.66
    for k, (head, sub) in enumerate(rows):
        y = len(rows) - k - 0.5
        conv = head.startswith("Conv")
        ax.add_patch(FancyBboxPatch((0.03, y - h / 2), 0.94, h, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="#eef1f5" if conv else "white", ec=SLATE if conv else "#b9bec6", lw=1.1))
        ax.text(0.07, y + 0.1, head, fontsize=11, fontweight="semibold", color=INK, va="center")
        ax.text(0.07, y - 0.15, sub, fontsize=9, color=DIM, va="center")
        if k < len(rows) - 1:
            ax.annotate("", xy=(0.5, y - h / 2 - 0.3 + 0.02), xytext=(0.5, y - h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=0, shrinkB=0))
    save(fig, "init_cnn_architecture_blocks.png")


# 4. Training curves
def curves(hist, prefix, marks=None):
    e = hist["epoch"].to_numpy()
    for metric, ylabel, fname in [("loss", "Categorical cross-entropy", "training_loss"),
                                  ("cat_acc", "Accuracy", "training_accuracy")]:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(e, hist[metric], color=SLATE, marker="o", ms=3.5, label="Training")
        ax.plot(e, hist["val_" + metric], color=RUST, marker="o", ms=3.5, label="Validation")
        if marks:
            for ep in marks:
                ax.axvline(ep - 0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=0)
            top = metric == "loss"
            ax.text(marks[0] - 0.35, 0.98 if top else 0.03, "learning rate cut", fontsize=8.5,
                    color=MUTED, va="top" if top else "bottom", transform=ax.get_xaxis_transform())
        if metric == "cat_acc":
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xticks(e)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        if marks:
            ax.legend(loc="lower left" if metric == "cat_acc" else "center right")
        else:
            ax.legend(loc="lower right" if metric == "cat_acc" else "upper right")
        tidy(ax)
        save(fig, f"{prefix}_{fname}.png")


# 5. Confusion matrix (row-normalised, held-out test set)
CM_CMAP = LinearSegmentedColormap.from_list("cm", ["#ffffff", "#c9d4e2", SLATE, "#2d3e55"])


def confusion(y_true, y_pred, fname):
    cm = np.zeros((5, 5))
    np.add.at(cm, (y_true, y_pred), 1)
    cm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.imshow(cm, cmap=CM_CMAP, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            v = cm[i, j]
            ax.text(j, i, f"{v:.0%}" if v >= 0.005 else "·", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.55 else (INK if v >= 0.005 else MUTED),
                    fontweight="semibold" if i == j else "normal")
    ax.set_xticks(range(5), NAMES); ax.set_yticks(range(5), NAMES)
    ax.set_xlabel("Predicted class", labelpad=8); ax.set_ylabel("True class", labelpad=8)
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.grid(False); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, fname)


# 6. First 20 test images with the final model's prediction
def example_predictions(X, test_idx, y_true, y_pred):
    fig, axes = plt.subplots(4, 5, figsize=(10, 9.2), gridspec_kw=dict(wspace=0.08, hspace=0.32))
    for k, ax in enumerate(axes.ravel()):
        show_plane(ax, planes(X, test_idx[k])[0])
        ok = y_true[k] == y_pred[k]
        t = NAMES[y_true[k]] if ok else f"{NAMES[y_true[k]]}, called {NAMES[y_pred[k]]}"
        ax.set_title(t, fontsize=9.5, color=INK if ok else RUST,
                     fontweight="normal" if ok else "semibold", pad=5)
        if not ok:
            for s in ax.spines.values():
                s.set_color(RUST); s.set_linewidth(2)
    save(fig, "cnn_example_predictions.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", help="npz with test_idx, y_true, initial_model, lr_1e-3_baseline")
    args = ap.parse_args()
    apply()

    X, truth, train_idx, test_idx = load_data()
    particle_types(X, truth, train_idx)
    truth_arrays(truth, train_idx)
    architecture()
    curves(INIT_HISTORY, "init_cnn")
    final = pd.read_csv(SRC / "models/lr_1e-3_baseline_history.csv")
    lr = final["learning_rate"].to_numpy()
    cuts = [int(final["epoch"][k]) for k in range(1, len(lr)) if lr[k] < lr[k - 1]]
    curves(final, "final_cnn", marks=cuts)

    if args.preds:
        p = np.load(args.preds)
        assert np.array_equal(p["test_idx"], test_idx)
        y = p["y_true"]
        for key, fname in [("initial_model", "init_cnn_confusion_matrix.png"),
                           ("lr_1e-3_baseline", "final_cnn_confusion_matrix.png")]:
            yp = p[key].argmax(1)
            print(key, "test accuracy", round(float((yp == y).mean()), 4))
            confusion(y, yp, fname)
        example_predictions(X, test_idx, y, p["lr_1e-3_baseline"].argmax(1))


if __name__ == "__main__":
    main()

"""
Cover figure for the high energy particle CNN classifier.

Shows one representative detector image per particle class, drawn from the real
dataset, so the card communicates what the model actually sees and separates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

SLATE, RUST, INK = "#3A506B", "#9E2A2B", "#1b1b1f"

# PDG codes actually present in the dataset
CLASSES = [
    (11,   "Electron", r"$e^-$"),
    (13,   "Muon",     r"$\mu^-$"),
    (22,   "Photon",   r"$\gamma$"),
    (211,  "Pion",     r"$\pi^+$"),
    (2212, "Proton",   r"$p$"),
]


def house_style(font_dir: Path) -> None:
    for face in ("Lora-Regular.ttf", "Lora-SemiBold.ttf"):
        if (font_dir / face).exists():
            fm.fontManager.addfont(str(font_dir / face))
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "font.family": "Lora" if (font_dir / "Lora-Regular.ttf").exists() else "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "semibold",
    })


def crop(img: np.ndarray, pad: int = 12) -> np.ndarray:
    """Trim to the region holding energy deposits so the track fills the cell."""
    ys, xs = np.nonzero(img)
    if len(ys) == 0:
        return img
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    h, w = y1 - y0, x1 - x0
    if h > w:  # keep cells square
        g = (h - w) // 2
        x0, x1 = max(x0 - g, 0), min(x1 + g, img.shape[1])
    else:
        g = (w - h) // 2
        y0, y1 = max(y0 - g, 0), min(y1 + g, img.shape[0])
    return img[y0:y1, x0:x1]


def pad_to(img: np.ndarray, size: int) -> np.ndarray:
    """Centre the crop in a fixed square canvas so every cell matches."""
    h, w = img.shape
    out = np.zeros((size, size), dtype=img.dtype)
    if h > size or w > size:
        y0 = max((h - size) // 2, 0); x0 = max((w - size) // 2, 0)
        img = img[y0:y0 + size, x0:x0 + size]
        h, w = img.shape
    y0 = (size - h) // 2; x0 = (size - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/oliviajackson/Documents/portfolio/projects/high-energy-particle-classifier")
    ap.add_argument("--out", default="/Users/oliviajackson/Documents/portfolio/olivia-jackson-lambert.github.io/project/cover_assets")
    ap.add_argument("--fonts", default=None)
    args = ap.parse_args()

    repo = Path(args.repo)
    house_style(Path(args.fonts) if args.fonts else repo / "assets/fonts")

    X = sp.load_npz(repo / "data/particle_images.npz")
    truth = np.load(repo / "data/particle_truth_array.npy")
    pdg = truth[:, 0].astype(int)

    # Dark ground with a cool-to-warm deposit ramp, the conventional way
    # detector images are shown. Sparse hits are invisible on white.
    GROUND = "#0d0f14"
    cmap = LinearSegmentedColormap.from_list(
        "dep", [GROUND, "#2f4258", SLATE, "#8d6b74", RUST, "#f2e4d8"])

    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.9), facecolor=GROUND)
    axes = axes.ravel()

    picked = []
    for code, name, sym in CLASSES:
        idx = np.flatnonzero(pdg == code)
        # Hit counts vary hugely by class (proton tracks are short and dense,
        # electron showers are broad), so choose relative to each class rather
        # than against one global range.
        cands = []
        for i in idx[:300]:
            img3 = X.getrow(int(i)).toarray().reshape(3, 256, 256)
            pr = int(np.argmax([(q != 0).sum() for q in img3]))
            cands.append((int((img3[pr] != 0).sum()), int(i), pr))
        cands.sort()
        n, i, pr = cands[int(len(cands) * 0.75)]   # busier than typical, still representative
        picked.append((name, sym, crop(X.getrow(i).toarray().reshape(3, 256, 256)[pr])))
        print(f"  {name:<9} hits={n}")

    size = max(max(im.shape) for _, _, im in picked)

    for ax, (name, sym, img) in zip(axes, picked):
        img = pad_to(img, size)
        nz = img[img > 0]
        vmax = np.percentile(nz, 97) if nz.size else 1.0
        ax.set_facecolor(GROUND)
        ax.imshow(img, cmap=cmap, norm=PowerNorm(0.30, vmin=0, vmax=vmax), interpolation="nearest")
        ax.set_title(f"{name}  {sym}", fontsize=11, pad=7, color="#e8e8ee")
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_color("#2a2f3a")

    # sixth cell carries the result rather than a sixth track
    ax = axes[5]
    ax.axis("off")
    ax.set_facecolor(GROUND)
    ax.text(0.5, 0.62, "5 classes", ha="center", va="center", fontsize=16,
            fontweight="semibold", color="#f2e4d8", transform=ax.transAxes)
    ax.text(0.5, 0.40, "89.1% validation accuracy", ha="center", va="center",
            fontsize=10.5, color="#c98b86", transform=ax.transAxes)

    fig.suptitle("Particle Signatures in a Liquid Argon Detector",
                 fontsize=13, fontweight="semibold", color="#f2f2f6", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=1.4)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "particle_signatures.png", dpi=220, bbox_inches="tight",
                pad_inches=0.06, facecolor=GROUND)
    print("saved:", out / "particle_signatures.png")


if __name__ == "__main__":
    main()

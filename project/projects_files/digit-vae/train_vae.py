"""
MNIST variational autoencoder: retrain and regenerate figures in house style.

A faithful PyTorch re-implementation of the original Keras notebook, which no
longer runs (it targets TF 1.x-era `tensorflow.python.keras` imports on Python
3.7). Architecture, loss and hyperparameters are unchanged:

  encoder  Conv 32 s1 -> Conv 64 s2 -> Conv 128 s2 -> Dense 256 -> (mu, logvar)
  latent   2 dimensions
  decoder  Dense 256 -> Dense 256 -> Dense 784 (sigmoid)
  loss     per-pixel BCE summed over the image, plus KL to a unit Gaussian
  training 50 epochs, batch 128, Adam

Usage:  python train_vae.py
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SLATE, RUST, INK, GRID = "#3A506B", "#9E2A2B", "#1b1b1f", "#e6e6ec"
TITLE_PAD = 16
LATENT, EPOCHS, BATCH, SEED = 2, 50, 128, 42


def house_style(font_dir: Path) -> None:
    for face in ("Lora-Regular.ttf", "Lora-SemiBold.ttf"):
        if (font_dir / face).exists():
            fm.fontManager.addfont(str(font_dir / face))
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
        "font.family": "Lora" if (font_dir / "Lora-Regular.ttf").exists() else "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10,
    })


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9d2")
    ax.tick_params(colors=INK, labelsize=9)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.c2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.c3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.fc = nn.Linear(128 * 7 * 7, 256)
        self.mu = nn.Linear(256, LATENT)
        self.lv = nn.Linear(256, LATENT)
        self.d1 = nn.Linear(LATENT, 256)
        self.d2 = nn.Linear(256, 256)
        self.d3 = nn.Linear(256, 784)

    def encode(self, x):
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h)); h = F.relu(self.c3(h))
        h = F.relu(self.fc(h.flatten(1)))
        return self.mu(h), self.lv(h)

    def decode(self, z):
        h = F.relu(self.d1(z)); h = F.relu(self.d2(h))
        return torch.sigmoid(self.d3(h)).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.decode(z), mu, lv


def losses(recon, x, mu, lv):
    reco = F.binary_cross_entropy(recon, x, reduction="none").flatten(1).sum(1)
    kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)
    return reco.mean(), kl.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets")
    ap.add_argument("--artifacts", default="../../../../projects/digit-vae-artifacts",
                    help="where weights, latent codes and history go; kept out of the site repo")
    ap.add_argument("--mnist", default=str(Path.home() / ".keras/datasets/mnist.npz"))
    ap.add_argument("--fonts", default="../../../../projects/high-energy-particle-classifier/assets/fonts")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts); art.mkdir(parents=True, exist_ok=True)
    house_style(Path(args.fonts))
    torch.manual_seed(SEED); np.random.seed(SEED)

    with np.load(args.mnist) as d:
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    print(f"train {xtr.shape} | test {xte.shape}")

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Xtr = torch.tensor(xtr, dtype=torch.float32).unsqueeze(1) / 255.0
    Xte = torch.tensor(xte, dtype=torch.float32).unsqueeze(1) / 255.0

    model = VAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tr = np.zeros(3)
        for i in range(0, len(Xtr), BATCH):
            xb = Xtr[perm[i:i + BATCH]].to(dev)
            recon, mu, lv = model(xb)
            r, k = losses(recon, xb, mu, lv)
            loss = r + k
            opt.zero_grad(); loss.backward(); opt.step()
            n = len(xb); tr += np.array([loss.item(), r.item(), k.item()]) * n
        tr /= len(Xtr)

        model.eval(); va = np.zeros(3)
        with torch.no_grad():
            for i in range(0, len(Xte), 512):
                xb = Xte[i:i + 512].to(dev)
                recon, mu, lv = model(xb)
                r, k = losses(recon, xb, mu, lv)
                n = len(xb); va += np.array([(r + k).item(), r.item(), k.item()]) * n
        va /= len(Xte)

        hist.append({"epoch": ep, "loss": tr[0], "reco": tr[1], "kl": tr[2],
                     "val_loss": va[0], "val_reco": va[1], "val_kl": va[2]})
        if ep == 1 or ep % 10 == 0 or ep == EPOCHS:
            print(f"  epoch {ep:>2}  loss {tr[0]:7.2f}  reco {tr[1]:7.2f}  kl {tr[2]:5.2f}  val {va[0]:7.2f}")

    json.dump(hist, open(art / "history.json", "w"), indent=1)
    torch.save(model.state_dict(), art / "vae_weights.pt")

    # ---------------- figures ----------------
    H = {k: np.array([h[k] for h in hist]) for k in hist[0]}

    # 1. example digits
    fig, axes = plt.subplots(3, 6, figsize=(6.4, 3.4))
    for ax, idx in zip(axes.ravel(), range(18)):
        ax.imshow(xtr[idx], cmap="gray_r"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(str(ytr[idx]), fontsize=9, pad=3, color=INK)
    fig.suptitle("Examples from the MNIST Dataset", fontsize=12, fontweight="semibold", color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "mnist_examples.png", dpi=220, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

    # 2. loss curves
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(H["epoch"], H["loss"], color=SLATE, lw=2, label="Train total")
    ax.plot(H["epoch"], H["val_loss"], color=RUST, lw=2, ls="--", label="Validation total")
    ax.plot(H["epoch"], H["reco"], color=SLATE, lw=1.2, alpha=.45, label="Train reconstruction")
    ax.plot(H["epoch"], H["kl"] * 10, color=RUST, lw=1.2, alpha=.45, label=r"Train KL ($\times$10)")
    ax.set_title("Training and Validation Loss", pad=TITLE_PAD, color=INK)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss per image")
    ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9); tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "vae_loss.png", dpi=220, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

    # latent codes for the test set
    model.eval()
    with torch.no_grad():
        Z = torch.cat([model.encode(Xte[i:i + 512].to(dev))[0].cpu() for i in range(0, len(Xte), 512)]).numpy()

    np.savez_compressed(art / "latent_codes.npz", z=Z, y=yte)

    # 3. latent clustering
    cmap10 = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    for d in range(10):
        m = yte == d
        ax.scatter(Z[m, 0], Z[m, 1], s=3, alpha=.45, color=cmap10(d), linewidths=0, label=str(d), rasterized=True)
    ax.set_title("Digit Clustering in the Latent Space", pad=TITLE_PAD, color=INK)
    ax.set_xlabel("$z_0$"); ax.set_ylabel("$z_1$")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
    leg = ax.legend(ncol=5, frameon=False, fontsize=9, markerscale=4,
                    loc="upper center", bbox_to_anchor=(.5, -.10))
    tidy(ax); fig.tight_layout()
    fig.savefig(out / "latent_clustering.png", dpi=220, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

    # 4. generated grid across the latent space
    n, span = 18, 2.6
    gx = np.linspace(-span, span, n); gy = np.linspace(span, -span, n)
    grid = np.zeros((n * 28, n * 28))
    with torch.no_grad():
        for r, yv in enumerate(gy):
            zs = torch.tensor(np.stack([gx, np.full(n, yv)], 1), dtype=torch.float32).to(dev)
            imgs = model.decode(zs).cpu().numpy().reshape(n, 28, 28)
            for c in range(n):
                grid[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = imgs[c]
    fig, ax = plt.subplots(figsize=(6.0, 6.2))
    ax.imshow(grid, cmap="gray_r")
    ax.set_title("Digits Generated Across the Latent Space", pad=TITLE_PAD, color=INK)
    ax.set_xticks([0, n * 28 / 2, n * 28]); ax.set_xticklabels([f"{-span:g}", "0", f"{span:g}"])
    ax.set_yticks([0, n * 28 / 2, n * 28]); ax.set_yticklabels([f"{span:g}", "0", f"{-span:g}"])
    ax.set_xlabel("$z_0$"); ax.set_ylabel("$z_1$")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    fig.tight_layout()
    fig.savefig(out / "latent_grid.png", dpi=220, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

    # 5. reconstructions: what a 2D bottleneck actually costs
    with torch.no_grad():
        xb = Xte[:10].to(dev)
        rec = model(xb)[0].cpu().numpy().reshape(-1, 28, 28)
    fig, axes = plt.subplots(2, 10, figsize=(9.0, 2.2))
    for i in range(10):
        axes[0, i].imshow(xte[i], cmap="gray_r")
        axes[1, i].imshow(rec[i], cmap="gray_r")
        for r in (0, 1):
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])
            for sp in axes[r, i].spines.values(): sp.set_visible(False)
    axes[0, 0].set_ylabel("Input", fontsize=9, color=INK)
    axes[1, 0].set_ylabel("Rebuilt", fontsize=9, color=INK)
    fig.suptitle("Reconstruction Through a Two-Dimensional Bottleneck",
                 fontsize=12, fontweight="semibold", color=INK, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "reconstructions.png", dpi=220, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

    f = hist[-1]
    print(f"\nFinal: loss {f['loss']:.1f} (reco {f['reco']:.1f}, kl {f['kl']:.1f}) | val {f['val_loss']:.1f}")
    print("Wrote figures to", out.resolve())


if __name__ == "__main__":
    main()

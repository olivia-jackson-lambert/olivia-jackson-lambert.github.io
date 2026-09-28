"""
MNIST variational autoencoder: retrain, or regenerate figures in the site style.

A faithful PyTorch re-implementation of the original Keras notebook, which no
longer runs (it targets TF 1.x-era `tensorflow.python.keras` imports on Python
3.7). Architecture, loss and hyperparameters are unchanged:

  encoder  Conv 32 s1 -> Conv 64 s2 -> Conv 128 s2 -> Dense 256 -> (mu, logvar)
  latent   2 dimensions
  decoder  Dense 256 -> Dense 256 -> Dense 784 (sigmoid)
  loss     per-pixel BCE summed over the image, plus KL to a unit Gaussian
  training 50 epochs, batch 128, Adam

Usage:
  python train_vae.py            # figures only, from saved weights and history
  python train_vae.py --retrain  # train from scratch, save artifacts, then figures

Weights, training history and latent codes live outside the site repo.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from figstyle import apply, tidy, INK, SLATE, RUST, DIM, MUTED  # noqa: E402

LATENT, EPOCHS, BATCH, SEED = 2, 50, 128, 42

# Ten classes: site colours first, then muted tones. Rust marks 4 and dim marks 9,
# the pair the article is about.
DIGIT_COLOURS = [
    SLATE,      # 0
    "#a9b8ca",  # 1 light slate
    "#6f8f7a",  # 2 sage
    "#e3a397",  # 3 light rust
    RUST,       # 4
    "#8c6d8f",  # 5 plum
    "#b08b5a",  # 6 ochre
    MUTED,      # 7
    "#4f7c82",  # 8 teal
    DIM,        # 9
]


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


def train(Xtr, Xte, dev, art: Path):
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

    model.eval()
    with torch.no_grad():
        Z = torch.cat([model.encode(Xte[i:i + 512].to(dev))[0].cpu()
                       for i in range(0, len(Xte), 512)]).numpy()
    return model, hist, Z


def place_labels(ax, centroids, colours, pts):
    """Greedy label placement. Each label takes the cheapest free spot around its
    centroid (short moves into sparse areas), never overlapping a label already
    placed or sitting on another centroid, with a leader line when it moves."""
    ax.figure.canvas.draw()
    to_px = ax.transData.transform
    to_data = ax.transData.inverted().transform
    cpx, ppx = to_px(centroids), to_px(pts)
    w = h = 24                                     # label box in pixels
    placed = []
    # most crowded centroids choose first
    crowd = [np.sort(np.hypot(*(cpx - c).T))[1] for c in cpx]
    angles = np.deg2rad(np.arange(0, 360, 15))
    for d in np.argsort(crowd):
        cx, cy = cpx[d]
        others = np.delete(cpx, d, axis=0)
        best, best_score = None, np.inf
        for r in (0, 34, 48, 62, 78, 95, 115, 140):
            for a in (angles if r else [0.0]):
                x, y = cx + r * np.cos(a), cy + r * np.sin(a)
                box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                if any(not (box[2] + 4 < b[0] or box[0] - 4 > b[2] or
                            box[3] + 4 < b[1] or box[1] - 4 > b[3]) for b in placed):
                    continue
                if np.min(np.hypot(*(others - [x, y]).T)) < 20:
                    continue
                density = np.sum(np.hypot(*(ppx - [x, y]).T) < 14)
                score = r + 0.1 * density
                if score < best_score:
                    best, best_score = (x, y, box), score
        x, y, box = best
        placed.append(box)
        lx, ly = to_data([x, y])
        if np.hypot(x - cx, y - cy) > 1:
            vx, vy = x - cx, y - cy
            ex, ey = to_data([x - vx / np.hypot(vx, vy) * 12, y - vy / np.hypot(vx, vy) * 12])
            ax.plot([centroids[d, 0], ex], [centroids[d, 1], ey], color=INK, lw=0.7, zorder=4)
            ax.plot(*centroids[d], "o", ms=3.4, color=INK, mec="white", mew=0.6, zorder=5)
        ax.text(lx, ly, str(d), ha="center", va="center", fontsize=10.5, fontweight="semibold",
                color="white" if d not in (1, 3) else INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", fc=colours[d], ec="white", lw=0.9))


def figures(model, hist, Z, xtr, ytr, xte, yte, Xte, dev, out: Path):
    apply()
    H = {k: np.array([h[k] for h in hist]) for k in hist[0]}
    save = dict(dpi=220, bbox_inches="tight", pad_inches=0.03, facecolor="white")

    def bare(ax):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values(): sp.set_visible(False)

    # 1. example digits
    fig, axes = plt.subplots(3, 6, figsize=(6.4, 3.3))
    for ax, idx in zip(axes.ravel(), range(18)):
        ax.imshow(xtr[idx], cmap="gray_r"); bare(ax)
        ax.set_title(str(ytr[idx]), fontsize=9.5, pad=3, color=INK)
    fig.tight_layout()
    fig.savefig(out / "mnist_examples.png", **save); plt.close(fig)

    # 2. loss curves
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(H["epoch"], H["loss"], color=SLATE, label="Train total")
    ax.plot(H["epoch"], H["val_loss"], color=RUST, label="Validation total")
    ax.plot(H["epoch"], H["reco"], color=SLATE, lw=1.2, alpha=.5, label="Train reconstruction")
    ax.plot(H["epoch"], H["kl"] * 10, color=DIM, lw=1.2, alpha=.6, label=r"Train KL ($\times$10)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss per image")
    ax.set_xlim(1, len(hist))
    ax.legend(loc="upper right"); tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "vae_loss.png", **save); plt.close(fig)

    # 3. latent clustering, labelled at each class centroid
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    order = np.random.default_rng(SEED).permutation(len(Z))   # no class drawn wholly on top
    ax.scatter(Z[order, 0], Z[order, 1], s=3, alpha=.55, linewidths=0,
               c=[DIGIT_COLOURS[k] for k in yte[order]], rasterized=True)
    ax.set_xlabel("$z_0$"); ax.set_ylabel("$z_1$")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    tidy(ax, grid_axis="both")
    C = np.array([Z[yte == d].mean(0) for d in range(10)])
    place_labels(ax, C, DIGIT_COLOURS, Z[(np.abs(Z) < 4).all(1)])
    fig.tight_layout()
    fig.savefig(out / "latent_clustering.png", **save); plt.close(fig)

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
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.imshow(grid, cmap="gray_r"); ax.grid(False)
    # ticks at z = -2, 0, 2, mapped to the centre of the cell that value falls in
    vals = np.array([-2, 0, 2])
    px = (vals + span) / (2 * span) * (n - 1) * 28 + 14
    ax.set_xticks(px); ax.set_xticklabels([f"{v:g}" for v in vals])
    ax.set_yticks(px); ax.set_yticklabels([f"{v:g}" for v in vals[::-1]])
    ax.set_xlabel("$z_0$"); ax.set_ylabel("$z_1$")
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "latent_grid.png", **save); plt.close(fig)

    # 5. reconstructions of the first ten test digits, decoded from the encoder mean
    with torch.no_grad():
        rec = model.decode(model.encode(Xte[:10].to(dev))[0]).cpu().numpy().reshape(-1, 28, 28)
    fig, axes = plt.subplots(2, 10, figsize=(9.0, 2.1))
    for i in range(10):
        axes[0, i].imshow(xte[i], cmap="gray_r"); bare(axes[0, i])
        axes[1, i].imshow(rec[i], cmap="gray_r"); bare(axes[1, i])
    axes[0, 0].set_ylabel("Input", fontsize=9.5, color=INK)
    axes[1, 0].set_ylabel("Rebuilt", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "reconstructions.png", **save); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "assets"))
    ap.add_argument("--artifacts", default=str(REPO.parent / "projects/digit-vae-artifacts"),
                    help="where weights, latent codes and history live; kept out of the site repo")
    ap.add_argument("--mnist", default=str(Path.home() / ".keras/datasets/mnist.npz"))
    ap.add_argument("--retrain", action="store_true", help="train from scratch before plotting")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    art = Path(args.artifacts); art.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    with np.load(args.mnist) as d:
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    print(f"train {xtr.shape} | test {xte.shape}")

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Xtr = torch.tensor(xtr, dtype=torch.float32).unsqueeze(1) / 255.0
    Xte = torch.tensor(xte, dtype=torch.float32).unsqueeze(1) / 255.0

    if args.retrain:
        model, hist, Z = train(Xtr, Xte, dev, art)
        np.savez_compressed(art / "latent_codes.npz", z=Z, y=yte)
    else:
        model = VAE().to(dev)
        model.load_state_dict(torch.load(art / "vae_weights.pt", map_location=dev))
        model.eval()
        hist = json.load(open(art / "history.json"))
        Z = np.load(art / "latent_codes.npz")["z"]

    figures(model, hist, Z, xtr, ytr, xte, yte, Xte, dev, out)

    C = np.array([Z[yte == d].mean(0) for d in range(10)])
    acc = (np.argmin(((Z[:, None] - C[None]) ** 2).sum(-1), 1) == yte).mean()
    f = hist[-1]
    print(f"Final: loss {f['loss']:.1f} (reco {f['reco']:.1f}, kl {f['kl']:.1f}) | val {f['val_loss']:.1f}")
    print(f"Nearest-centroid accuracy in the latent space: {acc:.1%}")
    print("Wrote figures to", out.resolve())


if __name__ == "__main__":
    main()

"""
Card thumbnails for the Work page: one design-led series, drawn from each
project's real data. No text, no axes. 1500 x 1000, shared dark ground.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm
from PIL import Image, ImageOps

SITE = Path("/Users/oliviajackson/Documents/portfolio/olivia-jackson-lambert.github.io")
OUT = SITE / "project/cover_assets/thumbs"
GROUND, LIGHT, DIM, RUST, SLATE = "#0f1115", "#ece9e4", "#3a3f48", "#c9503f", "#7189a6"
W, H, DPI = 15, 10, 100          # 1500 x 1000 px
RNG = np.random.default_rng(7)
GLOW = LinearSegmentedColormap.from_list("glow", [GROUND, "#1f2a38", SLATE, "#b8765f", RUST, "#f4e6d6"])


def canvas():
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=GROUND)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(GROUND); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, facecolor=GROUND)
    plt.close(fig); print("  wrote", name)


# 1. Flight delays: 1,000 flights in the real tier split; the severe 2.2% in rust.
def flight():
    fig, ax = canvas()
    shares = [(816, DIM, 70), (125, SLATE, 70), (37, "#b8765f", 90), (22, RUST, 150)]
    cols, rows = 50, 20
    kinds = np.concatenate([np.full(n, i) for i, (n, _, _) in enumerate(shares)])
    RNG.shuffle(kinds)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    for i, (_, c, s) in enumerate(shares):
        m = kinds == i
        ax.scatter(xs.ravel()[m], ys.ravel()[m], s=s, color=c, linewidths=0)
    ax.set_xlim(-3, cols + 2); ax.set_ylim(-3.2, rows + 2.2)
    save(fig, "flight-delay")


# 2. NBA: every home and away city pair as a travel arc; west-to-east in rust.
def nba():
    df = pd.read_csv("/Users/oliviajackson/Documents/berkeley/203/lab_2/Basketball-Study/data/processed/games_clean_model.csv",
                     usecols=["lat_home", "lon_home", "lat_away", "lon_away", "away_travel_high"])
    g = (df.groupby(["lat_home", "lon_home", "lat_away", "lon_away"])
           .agg(n=("away_travel_high", "size"),
                d=("away_travel_high", lambda s: s.mode().iat[0])).reset_index())
    fig, ax = canvas()
    t = np.linspace(0, 1, 60)[:, None]
    for _, r in g.sort_values("d").iterrows():
        a = np.array([r.lon_away, r.lat_away]); b = np.array([r.lon_home, r.lat_home])
        mid = (a + b) / 2 + np.array([0, np.linalg.norm(b - a) * .18])
        pts = (1 - t) ** 2 * a + 2 * (1 - t) * t * mid + t ** 2 * b
        we = r.d == "West to East"
        ax.plot(pts[:, 0], pts[:, 1], color=RUST if we else LIGHT,
                alpha=min(.08 + r.n / 900, .55 if we else .32), lw=1.3 if we else .8)
    cities = pd.concat([g[["lon_home", "lat_home"]].set_axis(["x", "y"], axis=1),
                        g[["lon_away", "lat_away"]].set_axis(["x", "y"], axis=1)]).drop_duplicates()
    ax.scatter(cities.x, cities.y, s=16, color=LIGHT, zorder=3, linewidths=0)
    ax.set_xlim(-128, -63); ax.set_ylim(22, 57); ax.set_aspect("auto")
    save(fig, "nba-travel")


# 3. Mars: a mosaic of real Curiosity frames, toned into the series palette.
def mars():
    src = Path("/Users/oliviajackson/Downloads/207_SWEOP_Project/Code")
    pred = pd.read_csv(src / "model_outputs/stage_3_iter5_test_predictions.csv")
    picks = []
    for c in sorted(pred.y_true.unique()):
        got = 0
        for fn in pred[pred.y_true == c].filename:
            p = src / "msl-images" / fn
            if p.exists():
                picks.append(p); got += 1
                if got == 2: break
    RNG.shuffle(picks)
    picks = picks[:24]
    cols, rows, gap = 6, 4, 10
    cw = (1500 - gap * (cols + 1)) // cols; ch = (1000 - gap * (rows + 1)) // rows
    sheet = Image.new("RGB", (1500, 1000), GROUND)
    for i, p in enumerate(picks):
        im = ImageOps.fit(Image.open(p).convert("L"), (cw, ch), Image.LANCZOS)
        im = ImageOps.colorize(ImageOps.autocontrast(im, cutoff=1), black=GROUND, white="#f1d9c4", mid="#8c5a48")
        sheet.paste(im, (gap + (i % cols) * (cw + gap), gap + (i // cols) * (ch + gap)))
    sheet.save(OUT / "mars-terrain.png"); print("  wrote mars-terrain")


# 4. Particle detector: many real events stacked into one luminous field.
def particle():
    import scipy.sparse as sp
    repo = Path("/Users/oliviajackson/Documents/portfolio/projects/high-energy-particle-classifier")
    X = sp.load_npz(repo / "data/particle_images.npz")
    pdg = np.load(repo / "data/particle_truth_array.npy")[:, 0].astype(int)
    acc = np.zeros((256, 256))
    for code, k in ((13, 4), (2212, 2), (211, 1)):
        for i in np.flatnonzero(pdg == code)[10:10 + k]:
            img = X.getrow(int(i)).toarray().reshape(3, 256, 256)
            acc += img[int(np.argmax([(q != 0).sum() for q in img]))]
    acc = acc[40:211, :]          # 171 x 256, close to 3:2
    fig, ax = canvas()
    ax.imshow(acc, cmap=GLOW, norm=PowerNorm(.32, vmin=0, vmax=np.percentile(acc[acc > 0], 98.5)),
              interpolation="bilinear", aspect="auto")
    save(fig, "particle")


# 5. Muon: resolution against momentum as a density field; shows both populations.
def muon():
    df = pd.read_csv("/Users/oliviajackson/Documents/portfolio/projects/mc-chic1.csv", comment="#",
                     names=["i", "ep", "eta", "p", "phi", "pol", "pt", "qp", "tx", "ty", "zV"], skiprows=1)
    m = (df.p < 260) & (df.ep > .0028) & (df.ep < .0105)
    Hh, _, _ = np.histogram2d(df.ep[m], np.log(df.p[m]), bins=(260, 390))
    fig, ax = canvas()
    ax.imshow(Hh + 1, origin="lower", cmap=GLOW, norm=LogNorm(vmin=1, vmax=Hh.max()),
              interpolation="bilinear", aspect="auto")
    save(fig, "muon")


# 6. VAE: the trained decoder's own digits, generated across the latent space.
def vae():
    import torch
    sys.path.insert(0, str(SITE / "project/projects_files/digit-vae"))
    from train_vae import VAE
    model = VAE(); model.load_state_dict(torch.load(
        "/Users/oliviajackson/Documents/portfolio/projects/digit-vae-artifacts/vae_weights.pt", map_location="cpu"))
    model.eval()
    cols, rows = 21, 14
    gx = np.linspace(-2.4, 2.4, cols); gy = np.linspace(1.7, -1.7, rows)
    grid = np.zeros((rows * 28, cols * 28))
    with torch.no_grad():
        for r, y in enumerate(gy):
            z = torch.tensor(np.stack([gx, np.full(cols, y)], 1), dtype=torch.float32)
            imgs = model.decode(z).numpy().reshape(cols, 28, 28)
            for c in range(cols): grid[r*28:(r+1)*28, c*28:(c+1)*28] = imgs[c]
    fig, ax = canvas()
    ax.imshow(grid, cmap=LinearSegmentedColormap.from_list("d", [GROUND, "#566479", LIGHT]),
              interpolation="bilinear", aspect="auto")
    save(fig, "digit-vae")


# 7. Receipt NLP: real BGE-M3 sentence embeddings drawn as strips, like a till roll.
def receipt():
    d = np.load(SITE / "project/projects_files/sentence-transformer/embeddings_bge_m3.npz", allow_pickle=True)
    E = np.vstack([d["receipt_embeddings"], d["embeddings"]])
    E = E[:, np.argsort(np.abs(E).mean(0))[::-1][:160]]
    lim = np.percentile(np.abs(E), 96)
    cmap = LinearSegmentedColormap.from_list("e", [SLATE, "#e9e4dc", RUST])
    fig, ax = canvas(); ax.set_xlim(0, 1500); ax.set_ylim(1000, 0)
    x0, x1, top = 560, 940, 70
    paper = plt.Rectangle((x0 - 26, top - 30), (x1 - x0) + 52, 890, color="#e9e4dc", zorder=1)
    ax.add_patch(paper)
    n = len(E); band = 30; gap = 11
    for r in range(n):
        y = top + r * (band + gap)
        ax.imshow(np.clip(E[r:r+1] / lim, -1, 1), cmap=cmap, vmin=-1, vmax=1, aspect="auto",
                  interpolation="nearest", extent=(x0, x1, y + band, y), zorder=2)
    # torn edge along the bottom of the roll
    xs = np.linspace(x0 - 26, x1 + 26, 40)
    ys = top + 860 + np.where(np.arange(40) % 2, 0, 12)
    ax.fill_between(xs, ys, 1000, color=GROUND, zorder=3)
    save(fig, "receipt")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (flight, nba, mars, particle, muon, vae, receipt): f()

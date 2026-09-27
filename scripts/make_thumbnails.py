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
from PIL import Image, ImageOps, ImageDraw

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
    """Every city pair as an arc: west-to-east rust, east-to-west blue, same zone faint."""
    df = pd.read_csv("/Users/oliviajackson/Documents/berkeley/203/lab_2/Basketball-Study/data/processed/games_clean_model.csv",
                     usecols=["lat_home", "lon_home", "lat_away", "lon_away", "away_travel_high"])
    g = (df.groupby(["lat_home", "lon_home", "lat_away", "lon_away"])
           .agg(n=("away_travel_high", "size"), d=("away_travel_high", lambda s: s.mode().iat[0])).reset_index())
    style = {"Neither": (LIGHT, .05, .6), "East to West": ("#8fa9cc", .22, 1.2), "West to East": (RUST, .18, 1.2)}
    fig, ax = canvas()
    t = np.linspace(0, 1, 80)[:, None]
    order = {"Neither": 0, "East to West": 1, "West to East": 2}
    for _, r in g.sort_values("d", key=lambda s: s.map(order)).iterrows():
        a = np.array([r.lon_away, r.lat_away]); b = np.array([r.lon_home, r.lat_home])
        # reciprocal trips share endpoints, so bend each direction its own way
        bend = {"West to East": .2, "East to West": -.2, "Neither": .08}[r.d]
        mid = (a + b) / 2 + np.array([0, np.linalg.norm(b - a) * bend])
        pts = (1 - t) ** 2 * a + 2 * (1 - t) * t * mid + t ** 2 * b
        c, al, lw = style[r.d]
        w = min(1 + r.n / 250, 3)
        if r.d != "Neither":   # soft glow under the directional arcs
            ax.plot(pts[:, 0], pts[:, 1], color=c, alpha=al * .25, lw=lw * w * 4, solid_capstyle="round")
        ax.plot(pts[:, 0], pts[:, 1], color=c, alpha=min(al * w, .85), lw=lw * w * .8, solid_capstyle="round")
    cities = pd.concat([g[["lon_home", "lat_home"]].set_axis(["x", "y"], axis=1),
                        g[["lon_away", "lat_away"]].set_axis(["x", "y"], axis=1)]).drop_duplicates()
    ax.scatter(cities.x, cities.y, s=26, color=LIGHT, zorder=4, linewidths=0)
    xs_all, ys_all = [], []
    for ln in ax.get_lines():
        x_, y_ = ln.get_data(); xs_all.append(np.asarray(x_)); ys_all.append(np.asarray(y_))
    xs_all = np.concatenate(xs_all); ys_all = np.concatenate(ys_all)
    cxm, cym = (xs_all.min() + xs_all.max()) / 2, (ys_all.min() + ys_all.max()) / 2
    hw = (xs_all.max() - xs_all.min()) / 2 * 1.14; hh = (ys_all.max() - ys_all.min()) / 2 * 1.2
    hw = max(hw, hh * 1.5 * .72); hh = max(hh, hw / 1.5 / .72)   # keep the frame's proportions
    ax.set_xlim(cxm - hw, cxm + hw); ax.set_ylim(cym - hh, cym + hh)
    save(fig, "nba-travel")


# 3. Mars: a mosaic of real Curiosity frames, toned into the series palette.
def mars():
    """B&W mosaic; calibration targets tinted blue, the easily confused instruments rust."""
    src = Path("/Users/oliviajackson/Downloads/207_SWEOP_Project/Code")
    pred = pd.read_csv(src / "model_outputs/stage_3_iter5_test_predictions.csv")
    CAL, CONFUSED = {1, 2, 12, 14}, {4, 6, 7, 10, 21}
    picks = []
    for c in sorted(pred.y_true.unique()):
        got = 0
        for fn in pred[pred.y_true == c].filename:
            p = src / "msl-images" / fn
            if p.exists():
                picks.append((p, c)); got += 1
                if got == 2: break
    RNG.shuffle(picks)
    picks = picks[:24]
    cols, rows, gap = 6, 4, 10
    cw = (1500 - gap * (cols + 1)) // cols; ch = (1000 - gap * (rows + 1)) // rows
    sheet = Image.new("RGB", (1500, 1000), GROUND)
    for i, (p, c) in enumerate(picks):
        im = ImageOps.autocontrast(ImageOps.fit(Image.open(p).convert("L"), (cw, ch), Image.LANCZOS), cutoff=1)
        if c in CAL:        im = ImageOps.colorize(im, black=GROUND, white="#dfe8f3", mid=SLATE)
        elif c in CONFUSED: im = ImageOps.colorize(im, black=GROUND, white="#f6e1d8", mid=RUST)
        else:               im = ImageOps.colorize(im, black=GROUND, white="#e6e4e0", mid="#6f7278")
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
    """Three till rolls standing side by side, each row a real BGE-M3 sentence embedding."""
    d = np.load(SITE / "project/projects_files/sentence-transformer/embeddings_bge_m3.npz", allow_pickle=True)
    E = np.vstack([d["receipt_embeddings"], d["embeddings"]])
    E = E[:, np.argsort(np.abs(E).mean(0))[::-1][:96]]
    lim = np.percentile(np.abs(E), 96)
    cmap = LinearSegmentedColormap.from_list("e", [SLATE, "#e9e4dc", RUST])
    PAPER = (0xe9, 0xe4, 0xdc)
    groups = [E[0:6], E[6:15], E[15:21]]            # three receipts of different lengths
    rw, band, gap, pad = 400, 56, 14, 30
    heights = [pad * 2 + len(g) * band + (len(g) - 1) * gap for g in groups]
    base = Image.new("RGB", (1500, 1000), GROUND)
    total = 3 * rw + 2 * 60; x0 = (1500 - total) // 2
    floor = (1000 + max(heights)) // 2                # bottom-aligned, group centred vertically
    for gi, (g, h) in enumerate(zip(groups, heights)):
        roll = Image.new("RGB", (rw, h), PAPER); px = roll.load()
        for r in range(len(g)):
            col = (np.asarray(cmap(np.clip(g[r] / lim, -1, 1) * .5 + .5))[:, :3] * 255).astype(np.uint8)
            strip = Image.fromarray(col[None, :, :]).resize((rw - 2 * pad, band), Image.NEAREST)
            roll.paste(strip, (pad, pad + r * (band + gap)))
        mask = Image.new("L", (rw, h), 255); dr = ImageDraw.Draw(mask)
        teeth = [(i * rw / 20, 0 if i % 2 == 0 else 9) for i in range(21)]
        dr.polygon([(0, 0)] + teeth + [(rw, 0)], fill=0)        # torn top edge
        base.paste(roll, (x0 + gi * (rw + 60), floor - h), mask)
    base.save(OUT / "receipt.png"); print("  wrote receipt")


# 8. MLOps essay: 100 projects leave research; about 15 reach production (the essay's 85% figure).
def essay():
    fig, ax = canvas()
    n = 100; ys = np.linspace(.07, .93, n)
    ok = np.zeros(n, bool); ok[RNG.choice(n, 15, replace=False)] = True
    for y, good in zip(ys, ok):
        if good:
            ax.plot([.06, .95], [y, y], color=RUST, lw=2.1, solid_capstyle="round")
            ax.scatter([.95], [y], s=22, color=RUST, zorder=3, linewidths=0)
        else:
            end = RNG.uniform(.14, .78)
            ax.plot([.06, end], [y, y], color=LIGHT, alpha=.34, lw=1.3, solid_capstyle="round")
            ax.scatter([end], [y], s=10, color=LIGHT, alpha=.55, zorder=3, linewidths=0)
    ax.plot([.06, .06], [.05, .95], color=LIGHT, alpha=.5, lw=1)
    ax.plot([.95, .95], [.05, .95], color=RUST, alpha=.5, lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "essay")


# 9. Nature Astronomy paper: hundreds of samples, every one of them empty.
def paper():
    fig, ax = canvas()
    cols, rows = 30, 20
    xs, ys = np.meshgrid(np.linspace(.05, .95, cols), np.linspace(.08, .92, rows))
    dist = np.hypot((xs - .5) * 1.5, ys - .5)
    shade = np.clip(1 - dist / .75, 0, 1)
    for x, y, s_ in zip(xs.ravel(), ys.ravel(), shade.ravel()):
        c = RUST if s_ > .55 else ("#b8765f" if s_ > .3 else LIGHT)
        ax.scatter([x], [y], s=210, facecolors="none", edgecolors=c, linewidths=1.8, alpha=.55 + .45 * s_)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "paper")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (flight, nba, mars, particle, muon, vae, receipt, essay, paper): f()

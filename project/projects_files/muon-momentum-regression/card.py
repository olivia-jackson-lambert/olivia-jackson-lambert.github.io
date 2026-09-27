"""
Cover animation for the muon momentum regression project.

Schematic, not measured data. A charged track in a magnetic field bends with a
radius proportional to its momentum, so high momentum tracks are nearly straight.
That is precisely why their momentum is hardest to measure: the resolution the
model predicts degrades as the curvature signal shrinks.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.animation import FuncAnimation, PillowWriter

FONTS = "/Users/oliviajackson/Documents/portfolio/projects/high-energy-particle-classifier/assets/fonts"
fm.fontManager.addfont(f"{FONTS}/Lora-Regular.ttf")
fm.fontManager.addfont(f"{FONTS}/Lora-SemiBold.ttf")
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11, "font.family": "Lora",
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.labelsize": 10,
})

SLATE, RUST, INK, GRID = "#3A506B", "#9E2A2B", "#1b1b1f", "#e6e6ec"

MOMENTA = np.array([5, 11, 25, 60, 150, 340])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("pm", [RUST, SLATE])
norm = matplotlib.colors.LogNorm(vmin=MOMENTA.min(), vmax=MOMENTA.max())

X_MAX, Y_MAX = 10.0, 4.3
N_FRAMES, N_PTS = 76, 600

def full_track(p):
    """Arc for momentum p, clipped to the frame."""
    R = 1.15 * p
    th = np.linspace(0, np.pi / 2, N_PTS)
    x, y = R * np.sin(th), R * (1 - np.cos(th))
    keep = (x <= X_MAX) & (y <= Y_MAX - 0.62)
    if not keep.any():
        keep[:2] = True
    return x[keep], y[keep]

TRACKS = [full_track(p) for p in MOMENTA]

fig, ax = plt.subplots(figsize=(6.4, 4.25))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

for x in np.arange(1.6, X_MAX + 0.01, 1.6):
    ax.axvline(x, color=GRID, lw=1.1, zorder=0)

lines, heads, tags = [], [], []
for p, (tx, ty) in zip(MOMENTA, TRACKS):
    c = cmap(norm(p))
    ln, = ax.plot([], [], lw=2.1, color=c, solid_capstyle="round", zorder=3)
    hd, = ax.plot([], [], "o", ms=4.2, color=c, zorder=4)
    tg = ax.text(tx[-1] + 0.16, ty[-1], f"{p} GeV/c", fontsize=8.6, color=c,
                 va="center", ha="left", zorder=5, alpha=0.0)
    lines.append(ln); heads.append(hd); tags.append(tg)

ax.plot([0], [0], "o", ms=7, color=INK, zorder=6)
ax.text(0.0, -0.34, "interaction point", fontsize=8.4, color=INK, alpha=.7, ha="left")

ax.set_title("Track Curvature Falls as Momentum Rises", pad=22, color=INK)
ax.text(0.5, 1.035, "the curvature signal shrinks exactly where resolution matters most",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.8, color=INK, alpha=.62)

ax.set_xlim(-0.45, X_MAX + 1.55); ax.set_ylim(-0.75, Y_MAX)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
fig.subplots_adjust(left=0.035, right=0.985, top=0.80, bottom=0.055)

def draw(s):
    for (tx, ty), ln, hd, tg in zip(TRACKS, lines, heads, tags):
        n = max(2, int(len(tx) * s))
        ln.set_data(tx[:n], ty[:n]); hd.set_data([tx[n-1]], [ty[n-1]])
        tg.set_position((tx[n-1] + 0.16, ty[n-1]))
        tg.set_alpha(float(np.clip((s - 0.80) / 0.15, 0, 1)))
    return lines + heads + tags

def update(f):
    t = min(f / (N_FRAMES * 0.70), 1.0)
    return draw(t * t * (3 - 2 * t))

anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=55, blit=True)
base = "/Users/oliviajackson/Documents/portfolio/olivia-jackson-lambert.github.io/project/projects_files/muon-momentum-regression/assets/track_curvature"
anim.save(base + ".gif", writer=PillowWriter(fps=18))
draw(1.0)
fig.savefig(base + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("saved gif + png")

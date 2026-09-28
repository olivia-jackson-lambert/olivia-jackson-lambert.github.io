"""
Cover animation for the muon momentum regression project.

Schematic, not measured data. A charged track in a magnetic field bends with a
radius proportional to its momentum, so high momentum tracks are nearly straight.
That is precisely why their momentum is hardest to measure: the resolution the
model predicts degrades as the curvature signal shrinks.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import apply, INK, SLATE, RUST, GRID, MUTED  # noqa: E402

apply()

MOMENTA = np.array([5, 11, 25, 60, 150, 340])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("pm", [RUST, SLATE])
norm = matplotlib.colors.LogNorm(vmin=MOMENTA.min(), vmax=MOMENTA.max())

X_MAX, Y_MAX = 10.0, 4.3
N_FRAMES, N_PTS = 76, 600

def full_track(p):
    """Arc for momentum p, sampled finely enough to reach the frame edge."""
    R = 1.15 * p
    th_max = min(np.arcsin(min(1.0, X_MAX / R)), np.arccos(max(-1.0, 1 - (Y_MAX - 0.62) / R)))
    th = np.linspace(0, th_max, N_PTS)
    return R * np.sin(th), R * (1 - np.cos(th))

TRACKS = [full_track(p) for p in MOMENTA]

# Labels sit at each track's end; push crowded ones apart vertically.
MIN_GAP = 0.30
ends = [ty[-1] for _, ty in TRACKS]
at_edge = [i for i, (tx, _) in enumerate(TRACKS) if tx[-1] > X_MAX - 0.2]
placed, LABEL_DY = -np.inf, np.zeros(len(TRACKS))
for i in sorted(at_edge, key=lambda i: ends[i]):
    y = max(ends[i], placed + MIN_GAP)
    LABEL_DY[i] = y - ends[i]
    placed = y

fig, ax = plt.subplots(figsize=(6.4, 3.7))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

for x in np.arange(1.6, X_MAX + 0.01, 1.6):
    ax.axvline(x, color=GRID, lw=1.1, zorder=0)

lines, heads, tags = [], [], []
for p, (tx, ty), dy in zip(MOMENTA, TRACKS, LABEL_DY):
    c = cmap(norm(p))
    ln, = ax.plot([], [], lw=2.1, color=c, solid_capstyle="round", zorder=3)
    hd, = ax.plot([], [], "o", ms=4.2, color=c, zorder=4)
    tg = ax.text(tx[-1] + 0.16, ty[-1] + dy, f"{p} GeV/c", fontsize=8.6, color=c,
                 va="center", ha="left", zorder=5, alpha=0.0)
    lines.append(ln); heads.append(hd); tags.append(tg)

ax.plot([0], [0], "o", ms=7, color=INK, zorder=6)
ax.text(0.0, -0.34, "interaction point", fontsize=8.4, color=MUTED, ha="left")

ax.set_xlim(-0.45, X_MAX + 1.55); ax.set_ylim(-0.75, Y_MAX)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
fig.subplots_adjust(left=0.035, right=0.985, top=0.97, bottom=0.055)

def draw(s):
    for (tx, ty), dy, ln, hd, tg in zip(TRACKS, LABEL_DY, lines, heads, tags):
        n = max(2, int(len(tx) * s))
        ln.set_data(tx[:n], ty[:n]); hd.set_data([tx[n-1]], [ty[n-1]])
        tg.set_position((tx[n-1] + 0.16, ty[n-1] + dy * np.clip((s - 0.80) / 0.15, 0, 1)))
        tg.set_alpha(float(np.clip((s - 0.80) / 0.15, 0, 1)))
    return lines + heads + tags

def update(f):
    t = min(f / (N_FRAMES * 0.70), 1.0)
    return draw(t * t * (3 - 2 * t))

anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=55, blit=True)
base = str(HERE / "assets" / "track_curvature")
anim.save(base + ".gif", writer=PillowWriter(fps=18))
draw(1.0)
fig.savefig(base + ".png", dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
print("saved gif + png")

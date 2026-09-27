"""
Figures for the NBA travel and point differential study.

Everything is recomputed in Python from the team's own cleaned modelling
dataset (games_clean_model.csv, 46,903 games) and reproduces the coefficients
reported in the R write-up.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SRC = Path("/Users/oliviajackson/Documents/berkeley/203/lab_2/Basketball-Study")
FONTS = Path("/Users/oliviajackson/Documents/portfolio/projects/high-energy-particle-classifier/assets/fonts")
OUT = Path("assets")
SLATE, RUST, INK, GRID = "#3A506B", "#9E2A2B", "#1b1b1f", "#e6e6ec"
TITLE_PAD = 16


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
    df = pd.read_csv(SRC / "data/processed/games_clean_model.csv")
    df["dir"] = pd.Categorical(df.away_travel_high,
                               categories=["Neither", "West to East", "East to West"])
    m = smf.ols("point_diff ~ dist_miles + C(dir) + attendance + year",
                data=df).fit(cov_type="HC1")
    sd = df.point_diff.std()

    # ---------- 1. the null result ----------
    bins = np.quantile(df.dist_miles, np.linspace(0, 1, 21))
    bins = np.unique(bins)
    idx = np.digitize(df.dist_miles, bins[1:-1])
    xs, ys, los, his = [], [], [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() < 50: continue
        v = df.point_diff[sel]
        se = v.std() / np.sqrt(len(v))
        xs.append(df.dist_miles[sel].mean()); ys.append(v.mean())
        los.append(v.mean() - 1.96 * se); his.append(v.mean() + 1.96 * se)
    xs, ys = np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.fill_between(xs, los, his, color=SLATE, alpha=.18, label="95% confidence interval")
    ax.plot(xs, ys, "o-", color=SLATE, lw=2, ms=5, label="Mean point differential")
    b0, b1 = m.params["Intercept"], m.params["dist_miles"]
    ax.axhline(df.point_diff.mean(), color=INK, lw=1, ls=":", alpha=.6)
    ax.text(xs.max(), df.point_diff.mean() + .12, "overall mean", ha="right",
            fontsize=8.4, color=INK, alpha=.7)
    ax.set_xlabel("Away team travel distance (miles)")
    ax.set_ylabel("Point differential (away minus home)")
    ax.set_title("Travel Distance Does Not Move the Result", pad=TITLE_PAD, color=INK)
    ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9); tidy(ax)
    ax.text(0.99, 0.05, f"distance coefficient {m.params['dist_miles']:+.5f} per mile, "
                        f"p = {m.pvalues['dist_miles']:.2f}",
            transform=ax.transAxes, ha="right", fontsize=8.6, color="#8a8f98", style="italic")
    fig.tight_layout()
    save(fig, "distance_null.png")

    # ---------- 2. direction is what shows up ----------
    order = ["East to West", "Neither", "West to East"]
    means, errs, ns = [], [], []
    for d in order:
        v = df.point_diff[df.away_travel_high == d]
        means.append(v.mean()); errs.append(1.96 * v.std() / np.sqrt(len(v))); ns.append(len(v))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cols = [SLATE, "#9aa4b1", RUST]
    ax.bar(order, means, yerr=errs, color=cols, width=.55, capsize=6,
           error_kw=dict(ecolor=INK, lw=1.2))
    for i, (mn, n) in enumerate(zip(means, ns)):
        ax.text(i, mn - 0.55, f"n = {n:,}", ha="center", fontsize=8.6, color="white")
    ax.set_ylabel("Mean point differential (away minus home)")
    ax.set_title("Direction of Travel Is Where the Signal Sits", pad=TITLE_PAD, color=INK)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True); tidy(ax)
    ax.text(0.99, 0.04, f"West to East coefficient {m.params['C(dir)[T.West to East]']:+.3f} points, "
                        f"p = {m.pvalues['C(dir)[T.West to East]']:.4f}",
            transform=ax.transAxes, ha="right", fontsize=8.6, color="#8a8f98", style="italic")
    fig.tight_layout()
    save(fig, "direction_effect.png")

    # ---------- 3. effect sizes against the noise ----------
    effects = [
        ("Longest trip\n(2,805 miles)", m.params["dist_miles"] * df.dist_miles.max()),
        ("West to East\ntravel", m.params["C(dir)[T.West to East]"]),
        ("A full extra\n10,000 fans", m.params["attendance"] * 10000),
        ("Ten years\nof drift", m.params["year"] * 10),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    names = [e[0] for e in effects]; vals = [e[1] for e in effects]
    bars = ax.bar(names, vals, color=[RUST if v < 0 else SLATE for v in vals], width=.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (.10 if v >= 0 else -.28),
                f"{v:+.2f}", ha="center", fontsize=10, color=INK)
    ax.axhspan(-sd, sd, color="#c8ccd4", alpha=.32, zorder=0)
    ax.axhline(0, color=INK, lw=1)
    ax.text(3.45, sd * .82, f"one standard deviation of\npoint differential  ({sd:.1f} points)",
            ha="right", fontsize=8.8, color="#6c7480", style="italic")
    ax.set_ylim(-sd * 1.12, sd * 1.12)
    ax.set_ylabel("Effect on point differential (points)")
    ax.set_title("Every Effect Is Small Against Game-to-Game Variation", pad=TITLE_PAD, color=INK)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True); tidy(ax)
    fig.tight_layout()
    save(fig, "effect_sizes.png")

    # ---------- 4. the data ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.1))
    a1.hist(df.dist_miles, bins=60, color=SLATE, edgecolor="none")
    a1.set_xlabel("Travel distance (miles)"); a1.set_ylabel("Games")
    a1.set_title("Distance Travelled", pad=TITLE_PAD, color=INK)
    per_year = df.groupby("year").size()
    a2.plot(per_year.index, per_year.values, color=SLATE, lw=2)
    a2.set_xlabel("Season"); a2.set_ylabel("Games in the dataset")
    a2.set_title("Coverage, 1947 to 2025", pad=TITLE_PAD, color=INK)
    for ax in (a1, a2):
        ax.grid(color=GRID, lw=.8); ax.set_axisbelow(True); tidy(ax)
    fig.tight_layout(w_pad=2.4)
    save(fig, "data_overview.png")

    print(f"\n  n={int(m.nobs):,}  adj R2={m.rsquared_adj:.5f}  SD(point_diff)={sd:.2f}")


if __name__ == "__main__":
    main()

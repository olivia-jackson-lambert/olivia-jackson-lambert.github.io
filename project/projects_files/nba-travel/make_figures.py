"""
Figures for the NBA travel and point differential study.

Everything is recomputed in Python from the team's own cleaned modelling
dataset (games_clean_model.csv, 46,903 games) and reproduces the coefficients
reported in the R write-up. The data stays outside the repo.

Style comes from the site's scripts/figstyle.py: shared palette only, Title Case
on all figure text, and layouts sized for a 620px scrolling column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import (apply, tidy, note, title_case, INK, SLATE, RUST, DIM,  # noqa: E402
                      MUTED, LIGHT, SLATE_TINT, PAPER)

SRC = Path("/Users/oliviajackson/Documents/berkeley/203/lab_2/Basketball-Study")
OUT = HERE / "assets"
COMMA = StrMethodFormatter("{x:,.0f}")
T = title_case


def mn(text):
    """Typographic minus signs, matching the axis tick labels."""
    return text.replace("-", "−")


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print("  wrote", name)


def main():
    apply()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC / "data/processed/games_clean_model.csv", low_memory=False)
    df["dir"] = pd.Categorical(df.away_travel_high,
                               categories=["Neither", "West to East", "East to West"])
    m = smf.ols("point_diff ~ dist_miles + C(dir) + attendance + year",
                data=df).fit(cov_type="HC1")
    sd = df.point_diff.std()
    overall = df.point_diff.mean()
    wte = "C(dir)[T.West to East]"
    y_diff = "Point Differential (Points, Away Minus Home)"

    # ---------- 1. the null result ----------
    bins = np.unique(np.quantile(df.dist_miles, np.linspace(0, 1, 21)))
    idx = np.digitize(df.dist_miles, bins[1:-1])
    xs, ys, los, his = [], [], [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() < 50:
            continue
        v = df.point_diff[sel]
        se = v.std() / np.sqrt(len(v))
        xs.append(df.dist_miles[sel].mean()); ys.append(v.mean())
        los.append(v.mean() - 1.96 * se); his.append(v.mean() + 1.96 * se)
    xs, ys = np.array(xs), np.array(ys)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.fill_between(xs, los, his, color=SLATE_TINT, lw=0, label=T("95% Confidence Interval"))
    ax.plot(xs, ys, "o-", color=SLATE, lw=2, ms=4.5, label=T("Mean Point Differential in Bin"))
    ax.axhline(overall, color=DIM, lw=1, ls=(0, (2, 2)))
    ax.text(xs.max(), min(los) - .15, mn(f"Dashed Line: Overall Mean ({overall:.2f})"),
            ha="right", va="top", fontsize=9, color=DIM)
    ax.set_xlabel("Away Team Travel Distance (Miles)")
    ax.set_ylabel(y_diff)
    ax.xaxis.set_major_formatter(COMMA)
    ax.set_ylim(min(los) - .8, max(his) + 1.0)
    ax.legend(loc="upper left", ncol=2)
    tidy(ax)
    note(ax, mn(f"Twenty Distance Bins with Equal Game Counts. Distance Coefficient "
                f"{m.params['dist_miles']:+.4f} Points per Mile, p = {m.pvalues['dist_miles']:.2f}"),
         y=-0.15)
    save(fig, "distance_null.png")

    # ---------- 2. direction is what shows up ----------
    order = ["East to West", "Neither", "West to East"]
    means, errs, ns = [], [], []
    for d in order:
        v = df.point_diff[df.away_travel_high == d]
        means.append(v.mean()); errs.append(1.96 * v.std() / np.sqrt(len(v))); ns.append(len(v))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(order, means, yerr=errs, color=[SLATE, LIGHT, RUST], width=.55, capsize=6,
           error_kw=dict(ecolor=INK, lw=1.1))
    for i, (mu, n) in enumerate(zip(means, ns)):
        ax.text(i, mu - errs[i] - 0.15, mn(f"{mu:.2f}"), ha="center", va="top",
                fontsize=10, color=INK, fontweight="semibold")
        ax.text(i, 0.1, f"n = {n:,}", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    ax.axhline(0, color=INK, lw=.8)
    ax.set_ylim(min(means) - max(errs) - 0.9, 0.6)
    ax.set_ylabel("Mean " + y_diff)
    ax.set_xlabel("Away Team Direction of Travel")
    tidy(ax)
    note(ax, f"Error Bars Are 95% Confidence Intervals. Model Coefficient for West to East "
             f"+{m.params[wte]:.2f} Points, p = {m.pvalues[wte]:.4f}", y=-0.17)
    save(fig, "direction_effect.png")

    # ---------- 3. effect sizes against the noise ----------
    effects = [
        ("Longest Trip\n(2,805 Miles)", m.params["dist_miles"] * df.dist_miles.max()),
        ("West to East\nTravel", m.params[wte]),
        ("10,000 More\nSpectators", m.params["attendance"] * 10000),
        ("Ten Years\nof Drift", m.params["year"] * 10),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axhspan(-sd, sd, color=PAPER, lw=0, zorder=0)
    names = [e[0] for e in effects]; vals = [e[1] for e in effects]
    bars = ax.bar(names, vals, color=[RUST if v < 0 else SLATE for v in vals], width=.5, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (.35 if v >= 0 else -.35),
                mn(f"{v:+.2f}"), ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, color=INK, fontweight="semibold")
    ax.axhline(0, color=INK, lw=.8, zorder=3)
    ax.text(3.45, sd - .5, f"Shaded Band: ±1 Standard Deviation\nof Point Differential ({sd:.1f} Points)",
            ha="right", va="top", fontsize=9, color=DIM)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-sd * 1.1, sd * 1.1)
    ax.set_ylabel("Effect on Point Differential (Points)")
    tidy(ax)
    save(fig, "effect_sizes.png")

    # ---------- 4. the data, stacked for a narrow column ----------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 8.0))
    a1.hist(df.dist_miles, bins=60, color=SLATE, edgecolor="white", linewidth=.3)
    a1.set_xlabel("Away Team Travel Distance (Miles)")
    a1.set_ylabel("Games")
    a1.xaxis.set_major_formatter(COMMA); a1.yaxis.set_major_formatter(COMMA)
    per_year = df.groupby("year").size()
    a2.plot(per_year.index, per_year.values, color=SLATE, lw=2)
    a2.set_xlabel("Season")
    a2.set_ylabel("Games in the Dataset")
    a2.yaxis.set_major_formatter(COMMA)
    a2.set_ylim(0, per_year.max() * 1.08)
    for ax, lab in ((a1, "A  Distance Travelled"), (a2, "B  Games per Season")):
        tidy(ax)
        ax.set_title(lab, loc="left", color=INK, pad=10)
    fig.align_ylabels((a1, a2))
    fig.tight_layout(h_pad=2.6)
    save(fig, "data_overview.png")

    print(f"\n  n={int(m.nobs):,}  adj R2={m.rsquared_adj:.5f}  SD(point_diff)={sd:.2f}")


if __name__ == "__main__":
    main()

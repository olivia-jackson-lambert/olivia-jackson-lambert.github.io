"""
Exploratory figures for the muon momentum resolution project.

Regenerates the four EDA figures from the source data in the site's shared
figure style (scripts/figstyle.py). The raw CSV stays outside the repo.

Usage:  python eda.py --data /path/to/mc-chic1.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import apply, tidy, SLATE, RUST  # noqa: E402


def save(fig, out: Path) -> None:
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print("  wrote", out.name)


def hist(series, xlabel, xlim, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(series, bins=240, range=xlim, color=SLATE, edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tracks per Bin (Log Scale)")
    tidy(ax)
    ax.tick_params(which="both", length=0)
    save(fig, out)


def band(x, y, xlabel, ylabel, out: Path) -> None:
    """Resolution band: dense scatter with a binned median overlaid."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.scatter(x, y, s=2, alpha=0.10, color=SLATE, linewidths=0, rasterized=True)

    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    edges = np.geomspace(max(x[m].min(), 1e-3), x[m].max(), 40)
    idx = np.digitize(x[m], edges)
    cx, cy = [], []
    for b in range(1, len(edges)):
        sel = idx == b
        if sel.sum() > 40:
            cx.append(np.median(x[m][sel]))
            cy.append(np.median(y[m][sel]))
    ax.plot(cx, cy, color=RUST, lw=2.0, label="Median")

    ax.set_xlim(0, 400)
    ax.set_ylim(0.002, 0.014)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    tidy(ax)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="muon_data.csv")
    ap.add_argument("--out", default=str(HERE / "assets"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    apply()
    # Math labels in the same serif face as the rest of the figure.
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Lora",
                         "mathtext.it": "Lora", "mathtext.bf": "Lora:semibold"})

    cols = ["Index", "ep", "eta", "p", "phi", "pol", "pt", "qp", "tx", "ty", "zV"]
    df = pd.read_csv(args.data, comment="#", names=cols, skiprows=1)
    df["pz"] = np.sqrt(np.clip(df["p"] ** 2 - df["pt"] ** 2, 0, None))
    df["epz"] = df["ep"] * df["p"] / df["pz"].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["p", "pt", "ep", "pz", "epz"])
    print(f"Loaded {len(df):,} tracks")

    hist(df["p"], "Momentum p (GeV/c)", (0, 600), out / "p_distribution.png")
    hist(df["pt"], "Transverse Momentum p$_T$ (GeV/c)", (0, 35), out / "pt_distrbution.png")

    band(df["p"].to_numpy(), df["ep"].to_numpy(),
         "Momentum p (GeV/c)", r"Momentum Resolution $\Delta p / p$",
         out / "fitted_track_momentum.png")

    band(df["pz"].to_numpy(), df["epz"].to_numpy(),
         "Longitudinal Momentum p$_Z$ (GeV/c)", r"Resolution $\Delta p_Z / p_Z$",
         out / "fitted_track_momentum_z.png")


if __name__ == "__main__":
    main()

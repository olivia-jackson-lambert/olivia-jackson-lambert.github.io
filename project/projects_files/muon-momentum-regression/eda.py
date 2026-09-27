"""
Exploratory figures for the muon momentum resolution project.

Regenerates the four EDA figures from the source data in the project's house
style (Lora, slate/rust palette), replacing the notebook's matplotlib defaults.

Usage:  python eda.py --data muon_data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SLATE, RUST, INK, GRID = "#3A506B", "#9E2A2B", "#1b1b1f", "#e6e6ec"
TITLE_PAD = 16


def house_style(font_dir: Path) -> None:
    for face in ("Lora-Regular.ttf", "Lora-SemiBold.ttf"):
        if (font_dir / face).exists():
            fm.fontManager.addfont(str(font_dir / face))
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "font.family": "Lora" if (font_dir / "Lora-Regular.ttf").exists() else "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
    })


def tidy(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9d2")
    ax.tick_params(colors=INK, labelsize=9)


def hist(series, title, xlabel, xlim, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.hist(series, bins=240, color=SLATE, edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_title(title, pad=TITLE_PAD, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency (log scale)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("  wrote", out.name)


def band(x, y, title, xlabel, ylabel, out: Path) -> None:
    """Resolution band: dense scatter with a binned median overlaid."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
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
    ax.set_title(title, pad=TITLE_PAD, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("  wrote", out.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="muon_data.csv")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--fonts", default="../../../../projects/high-energy-particle-classifier/assets/fonts")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    house_style(Path(args.fonts))

    cols = ["Index", "ep", "eta", "p", "phi", "pol", "pt", "qp", "tx", "ty", "zV"]
    df = pd.read_csv(args.data, comment="#", names=cols, skiprows=1)
    df["pz"] = np.sqrt(np.clip(df["p"] ** 2 - df["pt"] ** 2, 0, None))
    df["epz"] = df["ep"] * df["p"] / df["pz"].replace(0, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["p", "pt", "ep", "pz", "epz"])
    print(f"Loaded {len(df):,} tracks")

    hist(df["p"],  "Distribution of Muon Momentum",            "p (GeV/c)",     (0, 600), out / "p_distribution.png")
    hist(df["pt"], "Distribution of Transverse Momentum",      "p$_T$ (GeV/c)", (0, 35),  out / "pt_distrbution.png")

    band(df["p"].to_numpy(), df["ep"].to_numpy(),
         "Fitted Momentum Resolution vs. Momentum",
         "p (GeV/c)", r"$\Delta p / p$", out / "fitted_track_momentum.png")

    band(df["pz"].to_numpy(), df["epz"].to_numpy(),
         "Fitted Resolution vs. Longitudinal Momentum",
         "p$_Z$ (GeV/c)", r"$\Delta p_Z / p_Z$", out / "fitted_track_momentum_z.png")


if __name__ == "__main__":
    main()

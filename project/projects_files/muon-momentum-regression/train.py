"""
Muon track momentum resolution: upgraded training and evaluation pipeline.

Changes from the original notebook
----------------------------------
1. A genuine held-out test set. The original reported cross-validated R^2 on the
   full dataset, which measures fit quality, not expected performance on new data.
   Here the data is split 60/20/20 into train, validation and test. The test split
   is scored exactly once, at the end.
2. Scaling is fit on the training split only, so validation and test scores are
   not inflated by statistics leaked from data the model has not seen.
3. R^2 is reported alongside RMSE and MAE, and against two baselines (predicting
   the training mean, and linear regression), so the score has context.
4. Residuals are analysed in momentum bins, which is the question the physics
   actually cares about: resolution is hardest to measure where tracks are
   straightest.
5. Figures use the site's shared style (scripts/figstyle.py).

Note on the framework: the original used
`tensorflow.keras.wrappers.scikit_learn.KerasRegressor`, which was deprecated in
TF 2.6 and removed in TF 2.13, so the notebook no longer runs on current
TensorFlow. This pipeline uses scikit-learn's MLPRegressor, which expresses the
same dense architectures without the dead dependency.

Usage:  python train.py --data /path/to/mc-chic1.csv
        python train.py --data /path/to/mc-chic1.csv --replot
            Redraw the figures without the full model comparison: the error
            chart comes from the saved CSV, and only the selected model is
            refit (same seed and split) for the parity plot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["p", "tx", "ty", "eta", "phi"]
TARGET = "epz"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import apply, tidy, SLATE, RUST, DIM, INK, LIGHT, MUTED, MONO  # noqa: E402

SCALE = 1e4  # plot errors in units of 1e-4


def load(path: Path) -> pd.DataFrame:
    cols = ["Index", "ep", "eta", "p", "phi", "pol", "pt", "qp", "tx", "ty", "zV"]
    df = pd.read_csv(path, comment="#", names=cols, skiprows=1)

    # Longitudinal momentum from p^2 = pz^2 + pt^2, and the matching resolution.
    df["pz"] = np.sqrt(np.clip(df["p"] ** 2 - df["pt"] ** 2, 0, None))
    df["epz"] = df["ep"] * df["p"] / df["pz"].replace(0, np.nan)

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
    if before != len(df):
        print(f"  dropped {before - len(df):,} rows with undefined pz or epz")
    return df


def split(df: pd.DataFrame):
    """60 / 20 / 20 train, validation, test."""
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def mlp(hidden: tuple[int, ...]) -> TransformedTargetRegressor:
    """Dense net with both features and target standardised.

    The target is of order 1e-3, and MLPRegressor does not scale y. Left raw,
    the squared-error gradients are so small that Adam never converges and the
    network scores worse than predicting the mean. Standardising the target
    through TransformedTargetRegressor fixes this, and the inverse transform
    puts predictions back into physical units automatically.
    """
    inner = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            batch_size=100,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.15,
            random_state=SEED,
        )),
    ])
    return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())


def scores(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def error_table(p, y_true, y_pred) -> pd.DataFrame:
    """Where in momentum does the model actually fail? Eight equal-count bins."""
    edges = np.quantile(p, np.linspace(0, 1, 9))
    edges = np.unique(edges)
    idx = np.digitize(p, edges[1:-1])

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 2:
            continue
        rows.append({
            "p_low": edges[b], "p_high": edges[b + 1], "n": int(m.sum()),
            "bias": float(np.mean(y_pred[m] - y_true[m])),
            "rmse": float(np.sqrt(mean_squared_error(y_true[m], y_pred[m]))),
        })
    return pd.DataFrame(rows)


def error_plot(tab: pd.DataFrame, out: Path) -> None:
    """RMSE and mean bias per momentum bin, drawn as steps spanning each bin."""
    edges = np.r_[tab["p_low"].to_numpy(), tab["p_high"].iloc[-1]]
    rmse = tab["rmse"].to_numpy() * SCALE
    bias = tab["bias"].to_numpy() * SCALE

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    peak = tab["rmse"].nlargest(2).index
    lo, hi = tab.loc[peak, "p_low"].min(), tab.loc[peak, "p_high"].max()
    ax.axvspan(lo, hi, color=LIGHT, alpha=0.35, lw=0, zorder=0)
    ax.text(np.sqrt(lo * hi), rmse.max() * 1.06, f"peak error\n{lo:.0f} to {hi:.0f} GeV/c",
            ha="center", va="bottom", fontsize=9, color=RUST)

    ax.stairs(rmse, edges, color=SLATE, lw=2.2, label="RMSE", baseline=None)
    ax.stairs(bias, edges, color=DIM, lw=1.6, ls="--", label="Mean bias (predicted minus true)",
              baseline=None)
    ax.axhline(0, color=MUTED, lw=0.8)

    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ticks = [3, 5, 8, 12, 18, 31, 51, 85, 200, 739]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.6, rmse.max() * 1.32)
    ax.set_xlabel("Momentum p (GeV/c), eight equal-count bins")
    ax.set_ylabel(r"Error in $\Delta p_Z / p_Z$ ($\times 10^{-4}$)")
    ax.legend(loc="upper right")
    tidy(ax)
    ax.tick_params(which="both", length=0)
    fig.savefig(out / "test_error_by_momentum.png", dpi=200, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    plt.close(fig)


def parity_plot(y_true, y_pred, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lo, hi = np.quantile(y_true, [0.001, 0.999])
    ax.scatter(y_true * 1e3, y_pred * 1e3, s=3, alpha=0.18, color=SLATE, linewidths=0,
               rasterized=True)
    ax.plot([lo * 1e3, hi * 1e3], [lo * 1e3, hi * 1e3], color=RUST, lw=1.6,
            label="Perfect prediction")
    ax.set_xlim(lo * 1e3, hi * 1e3)
    ax.set_ylim(lo * 1e3, hi * 1e3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"True $\Delta p_Z / p_Z$ ($\times 10^{-3}$)")
    ax.set_ylabel(r"Predicted $\Delta p_Z / p_Z$ ($\times 10^{-3}$)")
    ax.legend(loc="upper left")
    tidy(ax, grid_axis="both")
    ax.text(0.98, 0.03, f"n = {len(y_true):,} test tracks", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=MUTED, fontfamily=MONO)
    fig.savefig(out / "test_parity.png", dpi=200, bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="muon_data.csv")
    ap.add_argument("--out", default=str(HERE / "assets"))
    ap.add_argument("--replot", action="store_true",
                    help="redraw figures only; refit just the selected model")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply()
    # Math labels in the same serif face as the rest of the figure.
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Lora",
                         "mathtext.it": "Lora", "mathtext.bf": "Lora:semibold"})

    print("Loading", args.data)
    df = load(Path(args.data))
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(df)
    print(f"  train {len(y_tr):,} | val {len(y_va):,} | test {len(y_te):,}")

    if args.replot:
        error_plot(pd.read_csv(out / "test_error_by_momentum.csv"), out)
        model = mlp((40, 5)).fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        saved = pd.read_csv(out / "test_metrics.csv").iloc[0]
        print(f"  refit test R2 = {r2_score(y_te, y_hat):.4f} (saved {saved['R2']:.4f})")
        parity_plot(y_te, y_hat, out)
        return

    candidates = {
        "Mean baseline":    DummyRegressor(strategy="mean"),
        "Linear regression": Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]),
        "MLP simple (5)":   mlp((5,)),
        "MLP denser (5,3)": mlp((5, 3)),
        "MLP wider (10)":   mlp((10,)),
        "MLP final (40,5)": mlp((40, 5)),
    }

    rows, fitted = [], {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
        row = {"Model": name}
        row.update({f"val_{k}": v for k, v in scores(y_va, model.predict(X_va)).items()})
        rows.append(row)
        print(f"  {name:<20} val R2 = {row['val_R2']:.4f}  RMSE = {row['val_RMSE']:.3e}")

    table = pd.DataFrame(rows).sort_values("val_R2", ascending=False)

    # Select on validation, then score the test split once.
    best_name = table.iloc[0]["Model"]
    best = fitted[best_name]
    y_hat = best.predict(X_te)
    test = scores(y_te, y_hat)

    print(f"\nSelected on validation: {best_name}")
    print(f"TEST  R2 = {test['R2']:.4f}  RMSE = {test['RMSE']:.4e}  MAE = {test['MAE']:.4e}")

    table["selected"] = table["Model"] == best_name
    table.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame([{"model": best_name, **test}]).to_csv(out / "test_metrics.csv", index=False)

    parity_plot(y_te, y_hat, out)
    p_te = X_te[:, FEATURES.index("p")]
    bins = error_table(p_te, y_te, y_hat)
    bins.to_csv(out / "test_error_by_momentum.csv", index=False)
    error_plot(bins, out)
    print(f"\nWrote figures and CSVs to {out.resolve()}")


if __name__ == "__main__":
    main()

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
5. Figures use the project's house style (Lora, slate/rust palette).

Note on the framework: the original used
`tensorflow.keras.wrappers.scikit_learn.KerasRegressor`, which was deprecated in
TF 2.6 and removed in TF 2.13, so the notebook no longer runs on current
TensorFlow. This pipeline uses scikit-learn's MLPRegressor, which expresses the
same dense architectures without the dead dependency.

Usage:  python train.py --data muon_data.csv
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

SLATE, RUST, INK, GRID = "#3A506B", "#9E2A2B", "#1b1b1f", "#e6e6ec"


def house_style(font_dir: Path) -> None:
    """Match the figure styling used across the rest of the portfolio."""
    for face in ("Lora-Regular.ttf", "Lora-SemiBold.ttf"):
        path = font_dir / face
        if path.exists():
            fm.fontManager.addfont(str(path))
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "font.family": "Lora" if (font_dir / "Lora-Regular.ttf").exists() else "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
    })


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


def residuals_by_momentum(p, y_true, y_pred, out: Path) -> pd.DataFrame:
    """Where in momentum does the model actually fail?"""
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
    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    centres = (tab["p_low"] + tab["p_high"]) / 2
    ax.plot(centres, tab["rmse"], "o-", color=SLATE, lw=2, ms=5, label="RMSE")
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(centres, tab["bias"], "s--", color=RUST, lw=1.8, ms=4.5, label="Mean bias")
    ax.set_title("Test Error by Momentum Bin", pad=16, color=INK)
    ax.set_xlabel("p (GeV/c)")
    ax.set_ylabel(r"Error in $\Delta p_Z / p_Z$")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "test_error_by_momentum.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return tab


def parity_plot(y_true, y_pred, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(y_true, y_pred, s=3, alpha=0.18, color=SLATE, linewidths=0, rasterized=True)
    lo, hi = np.quantile(y_true, [0.001, 0.999])
    ax.plot([lo, hi], [lo, hi], color=RUST, lw=1.6, label="Perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Predicted vs. True Resolution (Held-Out Test Set)", pad=16, color=INK)
    ax.set_xlabel(r"True $\Delta p_Z / p_Z$")
    ax.set_ylabel(r"Predicted $\Delta p_Z / p_Z$")
    ax.legend(frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "test_parity.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="muon_data.csv")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--fonts", default="../../../../projects/high-energy-particle-classifier/assets/fonts")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    house_style(Path(args.fonts))

    print("Loading", args.data)
    df = load(Path(args.data))
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(df)
    print(f"  train {len(y_tr):,} | val {len(y_va):,} | test {len(y_te):,}")

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
    bins = residuals_by_momentum(p_te, y_te, y_hat, out)
    bins.to_csv(out / "test_error_by_momentum.csv", index=False)
    print(f"\nWrote figures and CSVs to {out.resolve()}")


if __name__ == "__main__":
    main()

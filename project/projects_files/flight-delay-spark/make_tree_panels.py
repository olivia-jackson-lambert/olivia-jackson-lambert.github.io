"""Four-model comparison figures for the flight delay case study, in the site style.

The trained models and their outputs live on Databricks (DBFS), so the values here are
transcribed from the team's original figures:
- Confusion matrices: the row percentages printed in each cell, copied exactly.
- Feature importance: bar lengths measured from the original chart against its axis
  (precision about 0.1 percentage points). Feature order is as in the original.

Writes (same filenames the article uses):
  assets/tree_models_confusion_matrices_4panel.png
  assets/tree_models_feature_importance_4panel.png

Run: /Users/oliviajackson/Documents/portfolio/.venv/bin/python make_tree_panels.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from figstyle import apply, tidy, INK, SLATE, RUST, DIM, LIGHT, MUTED, CMAP  # noqa: E402

apply()
OUT = HERE / "assets"

MODELS = [
    "XGBoost, Severe Weight 1.0",
    "Random Forest, Severe Weight 1.0",
    "Random Forest, Severe Weight 1.2",
    "Random Forest, Severe Weight 1.5 (Selected)",
]
TIERS = ["On Time", "15–59 min", "1–2 hr", "2+ hr"]

# Row-normalised blind-test (2019) confusion matrices, percent; rows = true tier.
CM = [
    [[78.2, 14.9, 3.1, 3.9], [34.9, 45.4, 13.0, 6.7], [21.8, 23.0, 41.0, 14.2], [21.3, 15.1, 17.2, 46.4]],
    [[78.7, 15.0, 2.8, 3.5], [38.3, 40.8, 14.4, 6.5], [24.6, 21.6, 39.8, 14.0], [24.4, 15.2, 16.7, 43.7]],
    [[77.1, 13.8, 1.3, 7.9], [37.1, 37.9, 11.4, 13.5], [23.5, 19.1, 34.7, 22.7], [22.6, 12.8, 11.8, 52.9]],
    [[72.5, 10.2, 0.6, 16.6], [33.8, 31.5, 8.5, 26.1], [20.6, 13.9, 29.5, 36.0], [19.0, 8.5, 8.1, 64.4]],
]

# Top 15 features per model, importance in percent (measured from the original bars).
LABEL = {
    "inbound_delay_imp": "Inbound Delay",
    "inbound_delay_missing_imp": "Inbound Delay (Missing Flag)",
    "tail_cum_delay_imp": "Aircraft Cumulative Delay",
    "tail_cum_delay_missing_imp": "Aircraft Cumulative Delay (Missing Flag)",
    "tail_cum_legs_imp": "Aircraft Cumulative Legs",
    "origin_cum_delay_rate_imp": "Origin Cumulative Delay Rate",
    "origin_delay_rate_imp": "Origin Delay Rate",
    "origin_pagerank_imp": "Origin PageRank",
    "carrier_code_sidx": "Carrier",
    "carrier_freq_imp": "Carrier Frequency",
    "carrier_delay_rate_imp": "Carrier Delay Rate",
    "route_delay_te_imp": "Route Delay (Target Encoded)",
    "dest_delay_te_imp": "Destination Delay (Target Encoded)",
    "dep_hour_imp": "Departure Hour",
    "dep_hour_sin_imp": "Departure Hour (Sine)",
    "dep_time_block_sidx": "Departure Time Block",
    "time_of_day_sidx": "Time of Day",
    "is_redeye_imp": "Red-Eye Flight",
    "wx_is_precip_imp": "Precipitation",
    "wx_thunder_imp": "Thunderstorms",
    "wx_freezing_imp": "Freezing Conditions",
    "wx_snow_imp": "Snow",
}
FI = [
    [("tail_cum_delay_missing_imp", 14.57), ("inbound_delay_imp", 9.21), ("inbound_delay_missing_imp", 4.93),
     ("wx_is_precip_imp", 4.57), ("wx_thunder_imp", 4.36), ("tail_cum_delay_imp", 2.93), ("tail_cum_legs_imp", 2.86),
     ("carrier_code_sidx", 2.50), ("origin_cum_delay_rate_imp", 2.21), ("carrier_freq_imp", 1.79),
     ("time_of_day_sidx", 1.71), ("wx_freezing_imp", 1.57), ("is_redeye_imp", 1.50), ("origin_pagerank_imp", 1.50),
     ("wx_snow_imp", 1.50)],
    [("inbound_delay_imp", 42.07), ("tail_cum_delay_imp", 22.29), ("origin_cum_delay_rate_imp", 8.14),
     ("dep_hour_imp", 1.57), ("dep_hour_sin_imp", 1.57), ("origin_delay_rate_imp", 1.57), ("route_delay_te_imp", 1.50),
     ("carrier_delay_rate_imp", 1.36), ("tail_cum_legs_imp", 1.36), ("carrier_code_sidx", 1.36),
     ("dest_delay_te_imp", 1.21), ("time_of_day_sidx", 1.14), ("dep_time_block_sidx", 1.14),
     ("tail_cum_delay_missing_imp", 1.00), ("carrier_freq_imp", 0.86)],
    [("inbound_delay_imp", 41.50), ("tail_cum_delay_imp", 22.64), ("origin_cum_delay_rate_imp", 8.29),
     ("dep_hour_imp", 1.57), ("dep_hour_sin_imp", 1.50), ("carrier_delay_rate_imp", 1.50), ("origin_delay_rate_imp", 1.50),
     ("route_delay_te_imp", 1.50), ("carrier_code_sidx", 1.43), ("tail_cum_legs_imp", 1.36),
     ("dest_delay_te_imp", 1.21), ("time_of_day_sidx", 1.14), ("dep_time_block_sidx", 1.07),
     ("tail_cum_delay_missing_imp", 1.00), ("carrier_freq_imp", 0.86)],
    [("inbound_delay_imp", 41.21), ("tail_cum_delay_imp", 22.36), ("origin_cum_delay_rate_imp", 8.43),
     ("carrier_code_sidx", 1.57), ("dep_hour_sin_imp", 1.50), ("carrier_delay_rate_imp", 1.50), ("dep_hour_imp", 1.50),
     ("origin_delay_rate_imp", 1.43), ("tail_cum_legs_imp", 1.36), ("route_delay_te_imp", 1.36),
     ("dest_delay_te_imp", 1.21), ("time_of_day_sidx", 1.14), ("tail_cum_delay_missing_imp", 1.00),
     ("dep_time_block_sidx", 0.86), ("carrier_freq_imp", 0.86)],
]


def confusion():
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6))
    for k, ax in enumerate(axes.flat):
        m = CM[k]
        ax.imshow(m, cmap=CMAP, vmin=0, vmax=100)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(4):
            for j in range(4):
                v = m[i][j]
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=9.5,
                        color="white" if v >= 45 else INK,
                        fontweight="semibold" if i == j else "normal")
        # the severe-tier recall cell is the number the article is about
        ax.add_patch(Rectangle((2.5, 2.5), 1, 1, fill=False, edgecolor=RUST, linewidth=2.2))
        selected = k == 3
        ax.set_title(MODELS[k], fontsize=10.5, color=RUST if selected else INK,
                     fontweight="semibold", pad=10)
        ax.set_xticks(range(4), TIERS, fontsize=8.8)
        ax.set_yticks(range(4), TIERS, fontsize=8.8)
        ax.tick_params(length=0)
        if k >= 2:
            ax.set_xlabel("Predicted Tier")
        if k % 2 == 0:
            ax.set_ylabel("True Tier")
    fig.tight_layout(h_pad=2.6, w_pad=2.4)
    fig.savefig(OUT / "tree_models_confusion_matrices_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def importance():
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 10.2), sharex=True)
    for k, ax in enumerate(axes.flat):
        rows = FI[k][::-1]
        names = [LABEL[n] for n, _ in rows]
        vals = [v for _, v in rows]
        top = max(vals)
        colours = [RUST if v == top else SLATE for v in vals]
        ax.barh(range(len(vals)), vals, color=colours, height=0.68)
        ax.set_yticks(range(len(vals)), names, fontsize=8.6)
        for y, v in enumerate(vals):
            if v >= 5:
                ax.text(v + 0.6, y, f"{v:.1f}%", va="center", fontsize=8.4, color=DIM)
        tidy(ax, grid_axis="x")
        ax.set_xlim(0, 46)
        short = MODELS[k].replace("Random Forest", "Forest").replace("Severe Weight", "Weight")
        ax.set_title(short, fontsize=10.5, color=RUST if k == 3 else INK, fontweight="semibold", pad=8)
        if k >= 2:
            ax.set_xlabel("Importance (%)")
    fig.tight_layout(h_pad=2.4, w_pad=2.0)
    fig.savefig(OUT / "tree_models_feature_importance_4panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    confusion()
    importance()
    print("wrote tree_models_confusion_matrices_4panel.png and tree_models_feature_importance_4panel.png")

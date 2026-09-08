from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from paper_style import CN, WESTERN_FONT, apply_tick_fonts

ROOT = Path(__file__).resolve().parents[1]
STATION_MAP = {
    1011: "F01", 1009: "F02", 2213: "S01", 1686: "S02", 2255: "S03",
    1120: "S04", 1566: "S05", 1307: "S06", 1599: "M01", 1349: "M02",
}
ORDER = ["F01", "F02", "S01", "S02", "S03", "S04", "S05", "S06", "M01", "M02"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rolling scalar coefficient distributions for paper Fig. 6.")
    parser.add_argument("--output", type=Path, default=ROOT / "figures_generated/fig6_scalar_coefficients.png")
    parser.add_argument("--svg-output", type=Path, default=ROOT / "figures_generated/fig6_scalar_coefficients.svg")
    parser.add_argument("--data-output", type=Path, default=ROOT / "figures_generated/fig6_scalar_coefficients_data.csv")
    args = parser.parse_args()

    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = WESTERN_FONT
    mpl.rcParams["mathtext.it"] = f"{WESTERN_FONT}:italic"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["svg.fonttype"] = "none"

    data = pd.read_csv(ROOT / "results_reproduced/oof_predictions.csv.gz", parse_dates=["prediction_date"])
    daily = data.groupby(["station_id", "prediction_date"], as_index=False)["local_coefficient"].first()
    daily["station"] = daily["station_id"].map(STATION_MAP)
    daily = daily.dropna(subset=["station"]).copy()
    values = [daily.loc[daily["station"] == station, "local_coefficient"].to_numpy() for station in ORDER]

    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    daily[["station", "prediction_date", "local_coefficient"]].to_csv(args.data_output, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(3.35, 2.10), dpi=300)
    ax.boxplot(values, tick_labels=ORDER, widths=0.5, showfliers=False)
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True, length=3.0)
    apply_tick_fonts(ax, 7.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("目标站", fontsize=9, fontproperties=CN)
    ax.set_ylabel(r"滚动标量系数 $\alpha_{j,d}$", fontsize=9, fontproperties=CN)
    ax.axvline(2.5, color="black", linestyle="--", linewidth=0.6)
    ax.axvline(8.5, color="black", linestyle="--", linewidth=0.6)
    ax.text(1.5, 1.01, "快充站", ha="center", va="bottom", fontsize=8, fontproperties=CN)
    ax.text(5.5, 1.01, "慢充站", ha="center", va="bottom", fontsize=8, fontproperties=CN)
    ax.text(9.5, 1.01, "混合站", ha="center", va="bottom", fontsize=8, fontproperties=CN)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(args.svg_output, bbox_inches="tight", facecolor="white")
    print(args.output)
    print(args.svg_output)
    print(args.data_output)


if __name__ == "__main__":
    main()

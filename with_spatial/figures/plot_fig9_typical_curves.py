from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_style import CN, EN, apply_tick_fonts

ROOT = Path(__file__).resolve().parents[1]
STATION_MAP = {1011: "F01", 1566: "S05", 1599: "M01"}
TITLE = {"F01": ("快充站", "F01"), "S05": ("慢充站", "S05"), "M01": ("混合站", "M01")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot representative three-day forecasts for F01, S05 and M01.")
    parser.add_argument("--start-date", default="2023-02-15")
    parser.add_argument("--output", type=Path, default=ROOT / "figures_generated/fig9_typical_curves.png")
    parser.add_argument("--svg-output", type=Path, default=ROOT / "figures_generated/fig9_typical_curves.svg")
    parser.add_argument("--data-output", type=Path, default=ROOT / "figures_generated/fig9_typical_curves_data.csv")
    args = parser.parse_args()

    start = pd.Timestamp(args.start_date)
    end = start + pd.Timedelta(days=2)
    data = pd.read_csv(ROOT / "results_reproduced/oof_predictions.csv.gz", parse_dates=["prediction_date"])
    data = data[data["station_id"].isin(STATION_MAP) & data["prediction_date"].between(start, end)].copy()
    data["station"] = data["station_id"].map(STATION_MAP)
    data = data.sort_values(["station", "prediction_date", "slot"])
    data["time_h"] = data.groupby("station").cumcount() / 4.0

    expected = 3 * 96
    counts = data.groupby("station").size().to_dict()
    missing = {station: counts.get(station, 0) for station in STATION_MAP.values() if counts.get(station, 0) != expected}
    if missing:
        raise ValueError(f"Incomplete three-day representative window: {missing}")

    export = data[[
        "station", "prediction_date", "slot", "time_h", "y_true",
        "previous_day_baseline", "proposed_prediction",
    ]].rename(columns={
        "station": "target_station",
        "y_true": "actual_load_kw",
        "previous_day_baseline": "baseline_prediction_kw",
        "proposed_prediction": "proposed_prediction_kw",
    })
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(args.data_output, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.25), dpi=300, sharex=True)
    handles = None
    for ax, station in zip(axes, ["F01", "S05", "M01"]):
        g = data[data["station"] == station]
        fill = ax.fill_between(g["time_h"], 0.0, g["y_true"], color="0.85", linewidth=0.0, label="实际负荷")
        baseline, = ax.plot(
            g["time_h"], g["previous_day_baseline"],
            color="#d87a1c", linestyle="--", linewidth=1.15, label="基线方法",
        )
        proposed, = ax.plot(
            g["time_h"], g["proposed_prediction"],
            color="#3b70b7", linewidth=1.25, label="本文方法",
        )
        handles = [fill, baseline, proposed]
        ax.grid(False)
        ax.tick_params(direction="in", top=True, right=True, length=4)
        apply_tick_fonts(ax, 9)
        ax.set_ylabel("负荷 / kW", fontsize=10, fontproperties=CN)
        cn, sid = TITLE[station]
        ax.text(0.00, 1.035, cn, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontproperties=CN)
        ax.text(0.155, 1.035, sid, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", fontproperties=EN)
        y_max = max(float(g["y_true"].max()), float(g["previous_day_baseline"].max()), float(g["proposed_prediction"].max()))
        ax.set_ylim(0, y_max * 1.08)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    axes[-1].set_xlim(0, 72)
    axes[-1].set_xticks(np.arange(0, 73, 12))
    axes[-1].set_xlabel("时间 / h", fontsize=10, fontproperties=CN)
    fig.legend(
        handles, ["实际负荷", "基线方法", "本文方法"],
        loc="upper center", ncol=3, frameon=False, fontsize=10, prop=CN,
        bbox_to_anchor=(0.55, 0.995), handlelength=2.8, columnspacing=2.0,
    )
    fig.subplots_adjust(top=0.91, hspace=0.34, left=0.13, right=0.98, bottom=0.08)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(args.svg_output, bbox_inches="tight", facecolor="white")
    print(args.output)
    print(args.svg_output)
    print(args.data_output)


if __name__ == "__main__":
    main()

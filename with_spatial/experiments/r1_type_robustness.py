from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluation import (
    circular_moving_block_bootstrap_mean,
    newey_west_mean_test,
    station_wape_table,
    type_aggregate_wape_table,
)

OUTPUT_DIR = ROOT / "results_revision_r1"
TYPE_LABEL = {"pure_fast": "快充", "pure_slow": "慢充", "mixed_ac_dc": "混合"}


def _type_daily(frame: pd.DataFrame, station_type: str, prediction_column: str) -> pd.Series:
    group = frame[frame["type"].eq(station_type)].copy()
    fold_slot = group.groupby(["prediction_date", "fold", "slot"], as_index=False).agg(
        y_true=("y_true", "sum"),
        prediction=(prediction_column, "sum"),
    )
    fold_slot["absolute_error"] = np.abs(fold_slot["y_true"] - fold_slot["prediction"])
    fold_slot["actual_total"] = np.abs(fold_slot["y_true"])
    daily = fold_slot.groupby("prediction_date", sort=True).agg(
        absolute_error=("absolute_error", "sum"),
        actual_total=("actual_total", "sum"),
    )
    return daily["absolute_error"] / daily["actual_total"].clip(lower=1e-12)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "results_reproduced" / "oof_predictions.csv.gz", compression="gzip")
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"]).dt.normalize()

    aggregate = type_aggregate_wape_table(
        frame,
        "proposed_prediction",
        "previous_day_baseline",
    )
    baseline_station = station_wape_table(frame, "previous_day_baseline").rename(
        columns={"WAPE": "baseline_WAPE"}
    )
    method_station = station_wape_table(frame, "proposed_prediction").rename(
        columns={"WAPE": "method_WAPE"}
    )
    station = baseline_station.merge(
        method_station,
        on=["station_id", "type"],
        validate="one_to_one",
    )
    station["relative_change"] = station["method_WAPE"] / station["baseline_WAPE"] - 1.0

    rows = []
    inference = {}
    for station_type in sorted(frame["type"].unique()):
        station_group = station[station["type"].eq(station_type)]
        proposed = _type_daily(frame, station_type, "proposed_prediction")
        baseline = _type_daily(frame, station_type, "previous_day_baseline")
        proposed, baseline = proposed.align(baseline, join="inner")
        difference = (proposed - baseline).to_numpy(float)
        bootstrap = circular_moving_block_bootstrap_mean(
            difference,
            block_length=7,
            repetitions=50_000,
            seed=20260716,
        )
        nw = newey_west_mean_test(difference, max_lag=7)
        aggregate_row = aggregate[aggregate["type"].eq(station_type)].iloc[0]
        row = {
            "type": station_type,
            "type_cn": TYPE_LABEL.get(station_type, station_type),
            "station_count": int(len(station_group)),
            "baseline_WAPE": float(aggregate_row["baseline_WAPE"]),
            "method_WAPE": float(aggregate_row["method_WAPE"]),
            "relative_improvement_percent": float(aggregate_row["relative_improvement_percent"]),
            "improved_stations": int((station_group["relative_change"] < 0).sum()),
            "max_station_relative_degradation": float(
                max(0.0, station_group["relative_change"].max())
            ),
            "mean_daily_WAPE_difference": float(difference.mean()),
            "circular_mbb_95_low": float(np.quantile(bootstrap, 0.025)),
            "circular_mbb_95_high": float(np.quantile(bootstrap, 0.975)),
            "newey_west_p_two_sided": float(nw["p_two_sided"]),
        }
        rows.append(row)
        inference[station_type] = row

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "r1_type_robustness.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "r1_type_robustness.json").write_text(
        json.dumps(inference, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_revision_r1"

from src.evaluation import paired_daily_inference, station_wape_table, type_aggregate_wape_table

TYPE_CN = {"pure_fast": "快充", "pure_slow": "慢充", "mixed_ac_dc": "混合"}


def deployment_group_wape(group: pd.DataFrame, column: str) -> float:
    agg = group.groupby(["prediction_date", "slot"], as_index=False).agg(
        y_true=("y_true", "sum"), prediction=(column, "sum")
    )
    denom = max(float(np.abs(agg["y_true"]).sum()), 1e-12)
    return float(np.abs(agg["y_true"] - agg["prediction"]).sum() / denom)


def main() -> None:
    path = ROOT / "results_reproduced/oof_predictions.csv.gz"
    data = pd.read_csv(path)
    data["prediction_date"] = pd.to_datetime(data["prediction_date"]).dt.normalize()

    type_base = type_aggregate_wape_table(data, "proposed_prediction", "previous_day_baseline")
    station_base = station_wape_table(data, "previous_day_baseline").rename(columns={"WAPE": "baseline_WAPE"})
    station_method = station_wape_table(data, "proposed_prediction").rename(columns={"WAPE": "method_WAPE"})
    station = station_base.merge(station_method, on=["station_id", "type"], validate="one_to_one")
    station["relative_change"] = station["method_WAPE"] / station["baseline_WAPE"] - 1.0

    type_rows = []
    for _, row in type_base.iterrows():
        station_type = row["type"]
        subset = station[station["type"] == station_type]
        inference = paired_daily_inference(
            data[data["type"] == station_type],
            proposed_column="proposed_prediction",
            baseline_column="previous_day_baseline",
        )
        type_rows.append({
            "type": station_type,
            "type_cn": TYPE_CN.get(station_type, station_type),
            "station_count": int(len(subset)),
            "baseline_WAPE": float(row["baseline_WAPE"]),
            "method_WAPE": float(row["method_WAPE"]),
            "relative_improvement_percent": float(row["relative_improvement_percent"]),
            "improved_stations": int((subset["relative_change"] < 0).sum()),
            "max_station_relative_degradation": float(max(0.0, subset["relative_change"].max())),
            "mean_daily_WAPE_difference": float(inference["mean_daily_WAPE_difference"]),
            "circular_mbb_95_low": float(inference["circular_mbb_95_low"]),
            "circular_mbb_95_high": float(inference["circular_mbb_95_high"]),
            "newey_west_p_two_sided": float(inference["newey_west_p_two_sided"]),
            "inference_scope_note": "time-axis inference for the fixed target stations of this type; not a population-level station-sampling inference",
        })
    type_summary = pd.DataFrame(type_rows)

    fold_rows = []
    for fold, group in data.groupby("fold", sort=True):
        baseline_wape = deployment_group_wape(group, "previous_day_baseline")
        method_wape = deployment_group_wape(group, "proposed_prediction")
        ids = sorted(int(v) for v in group["station_id"].unique())
        station_subset = station[station["station_id"].isin(ids)].sort_values("station_id")
        status = ";".join(
            f"{int(r.station_id)}:{'improve' if r.relative_change < 0 else 'degrade'}({r.relative_change*100:+.2f}%)"
            for r in station_subset.itertuples(index=False)
        )
        inference = paired_daily_inference(
            group,
            proposed_column="proposed_prediction",
            baseline_column="previous_day_baseline",
        )
        fold_rows.append({
            "fold": fold,
            "station_ids": ",".join(str(v) for v in ids),
            "baseline_WAPE": baseline_wape,
            "method_WAPE": method_wape,
            "relative_improvement_percent": (baseline_wape - method_wape) / baseline_wape * 100.0,
            "improved_stations": int((station_subset["relative_change"] < 0).sum()),
            "max_station_relative_degradation": float(max(0.0, station_subset["relative_change"].max())),
            "mean_daily_WAPE_difference": float(inference["mean_daily_WAPE_difference"]),
            "circular_mbb_95_low": float(inference["circular_mbb_95_low"]),
            "circular_mbb_95_high": float(inference["circular_mbb_95_high"]),
            "newey_west_p_two_sided": float(inference["newey_west_p_two_sided"]),
            "station_status": status,
            "inference_scope_note": "44-day time-axis inference for this fixed two-station deployment group",
        })
    fold_summary = pd.DataFrame(fold_rows)

    deployment_meta = pd.DataFrame([{
        "deployment_groups": len(fold_summary),
        "groups_with_positive_aggregate_improvement": int((fold_summary["relative_improvement_percent"] > 0).sum()),
        "groups_with_aggregate_degradation": int((fold_summary["relative_improvement_percent"] <= 0).sum()),
        "min_group_relative_improvement_percent": float(fold_summary["relative_improvement_percent"].min()),
        "max_group_relative_improvement_percent": float(fold_summary["relative_improvement_percent"].max()),
        "interpretation": "deployment-group heterogeneity check; do not claim universal improvement across all folds/stations",
    }])

    OUT.mkdir(parents=True, exist_ok=True)
    type_summary.to_csv(OUT / "r1_type_robustness_summary.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(OUT / "r1_deployment_group_robustness.csv", index=False, encoding="utf-8-sig")
    station.to_csv(OUT / "r1_station_robustness_detail.csv", index=False, encoding="utf-8-sig")
    deployment_meta.to_csv(OUT / "r1_deployment_group_robustness_meta.csv", index=False, encoding="utf-8-sig")

    print("TYPE SUMMARY")
    print(type_summary.to_string(index=False))
    print("\nDEPLOYMENT-GROUP (FOLD) ROBUSTNESS")
    print(fold_summary.to_string(index=False))
    print("\nDEPLOYMENT-GROUP META")
    print(deployment_meta.to_string(index=False))


if __name__ == "__main__":
    main()

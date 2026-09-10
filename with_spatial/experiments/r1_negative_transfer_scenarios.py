from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_revision_r1"
OOF = ROOT / "results_reproduced/oof_predictions.csv.gz"

STAGES = ["station_candidate", "group_candidate", "proposed_prediction"]


def station_day_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["fold", "station_id", "type", "prediction_date"]
    for (fold, station_id, station_type, prediction_date), group in frame.groupby(keys, sort=True):
        y = group["y_true"].to_numpy(dtype=float)
        baseline = group["previous_day_baseline"].to_numpy(dtype=float)
        denom = max(float(np.abs(y).sum()), 1e-12)
        row: dict[str, object] = {
            "fold": fold,
            "station_id": int(station_id),
            "type": station_type,
            "prediction_date": pd.Timestamp(prediction_date),
            "actual_energy": float(y.sum()),
            "previous_day_energy": float(baseline.sum()),
            "baseline_AE": float(np.abs(y - baseline).sum()),
            "baseline_WAPE": float(np.abs(y - baseline).sum() / denom),
            "recent_enabled": int(group["recent_monitor_enabled"].iloc[0]),
            "final_enabled": int(group["final_enabled"].iloc[0]),
        }
        for stage in STAGES:
            p = group[stage].to_numpy(dtype=float)
            row[f"{stage}_AE"] = float(np.abs(y - p).sum())
            row[f"{stage}_WAPE"] = float(np.abs(y - p).sum() / denom)
        rows.append(row)
    out = pd.DataFrame(rows)
    for stage in STAGES:
        # The denominator is identical within a station-day, so WAPE and absolute-error
        # comparisons give the same negative-transfer label.
        out[f"{stage}_nt"] = out[f"{stage}_AE"] > out["baseline_AE"] + 1e-12
    return out


def load_calendar() -> pd.DataFrame:
    pieces = []
    cols = ["date", "is_weekend", "is_holiday", "is_public_holiday_cn", "is_spring_festival"]
    for split in ("train", "validation", "test"):
        path = ROOT / "data/source_station_dataset" / split / "station_daily_features.csv"
        table = pd.read_csv(path, usecols=cols)
        table["date"] = pd.to_datetime(table["date"]).dt.normalize()
        pieces.append(table.drop_duplicates())
    table = pd.concat(pieces, ignore_index=True).drop_duplicates()
    # Calendar labels must be consistent across station/split copies of the same date.
    counts = table.groupby("date").size()
    if counts.max() != 1:
        bad = counts[counts > 1]
        raise AssertionError(f"calendar flags inconsistent across source splits: {bad.to_dict()}")
    table["operating_condition"] = np.where(
        table["is_spring_festival"].astype(int).eq(1),
        "Spring_Festival",
        np.where(table["is_weekend"].astype(int).eq(1), "Weekend", "Regular_weekday"),
    )
    return table


def daily_wape_margin_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic sensitivity to the station-day NT definition.

    A station-day is counted as material negative transfer when the candidate's
    WAPE exceeds the previous-day baseline by more than an absolute WAPE margin.
    These thresholds are descriptive only and never enter the controller.
    """
    rows = []
    for stage, label in [
        ("station_candidate", "station_candidate"),
        ("group_candidate", "group_candidate"),
        ("proposed_prediction", "final_prediction"),
    ]:
        diff = data[f"{stage}_WAPE"] - data["baseline_WAPE"]
        for margin in (0.0, 0.01, 0.03, 0.05):
            rows.append({
                "stage": label,
                "absolute_WAPE_margin": margin,
                "station_days_above_margin": int((diff > margin + 1e-12).sum()),
                "rate": float((diff > margin + 1e-12).mean()),
            })
    return pd.DataFrame(rows)


def positive_excess_severity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, label in [
        ("station_candidate", "station_candidate"),
        ("group_candidate", "group_candidate"),
        ("proposed_prediction", "final_prediction"),
    ]:
        diff = data[f"{stage}_WAPE"] - data["baseline_WAPE"]
        pos = diff[diff > 1e-12]
        rows.append({
            "stage": label,
            "positive_station_days": int(len(pos)),
            "median_excess_WAPE": float(pos.median()) if len(pos) else 0.0,
            "p75_excess_WAPE": float(pos.quantile(0.75)) if len(pos) else 0.0,
            "p90_excess_WAPE": float(pos.quantile(0.90)) if len(pos) else 0.0,
            "max_excess_WAPE": float(pos.max()) if len(pos) else 0.0,
            "scope_note": "station-day diagnostic only; the controller is historical/cumulative and does not guarantee zero day-wise degradation",
        })
    return pd.DataFrame(rows)


def main() -> None:
    frame = pd.read_csv(OOF)
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"]).dt.normalize()
    data = station_day_table(frame)

    stage_rows = []
    for stage, label in [
        ("station_candidate", "station_candidate"),
        ("group_candidate", "group_candidate"),
        ("proposed_prediction", "final_prediction"),
    ]:
        n = int(data[f"{stage}_nt"].sum())
        stage_rows.append({
            "stage": label,
            "negative_transfer_days": n,
            "total_station_days": int(len(data)),
            "rate": n / len(data),
        })
    stage_rates = pd.DataFrame(stage_rows)

    station_nt = data["station_candidate_nt"]
    group_nt = data["group_candidate_nt"]
    conditions = [
        (~station_nt & ~group_nt),
        (~station_nt & group_nt),
        (station_nt & ~group_nt),
        (station_nt & group_nt),
    ]
    labels = [
        "both_improve",
        "group_induces_station_nt",
        "group_repairs_station_nt",
        "persistent_nt",
    ]
    data["mechanism_scenario"] = np.select(conditions, labels, default="unclassified")
    scenario_counts = (
        data.groupby("mechanism_scenario", sort=False)
        .size().rename("count").reset_index()
    )
    scenario_counts["rate"] = scenario_counts["count"] / len(data)

    guard = pd.DataFrame([{
        "recent_enabled_station_days": int(data["recent_enabled"].sum()),
        "final_enabled_station_days": int(data["final_enabled"].sum()),
        "cumulative_guard_closed_station_days": int(((data["recent_enabled"] == 1) & (data["final_enabled"] == 0)).sum()),
    }])

    calendar = load_calendar()[["date", "operating_condition"]]
    data = data.merge(calendar, left_on="prediction_date", right_on="date", how="left", validate="many_to_one")
    if data["operating_condition"].isna().any():
        raise AssertionError("missing calendar operating-condition label")
    data = data.drop(columns="date")

    operating = (
        data.groupby("operating_condition", sort=False)
        .agg(
            calendar_days=("prediction_date", "nunique"),
            station_days=("station_id", "size"),
            station_candidate_nt_rate=("station_candidate_nt", "mean"),
            group_candidate_nt_rate=("group_candidate_nt", "mean"),
            final_nt_rate=("proposed_prediction_nt", "mean"),
            recent_enabled_rate=("recent_enabled", "mean"),
            final_enabled_rate=("final_enabled", "mean"),
        )
        .reset_index()
    )

    # Post-hoc operating-severity diagnostic. This variable uses day-d true load and
    # therefore is NEVER an input to the forecast or the risk decision. It is used
    # only after forecasting to describe when negative transfer occurred.
    data["posthoc_interday_total_change"] = (
        np.abs(data["actual_energy"] - data["previous_day_energy"])
        / np.maximum(np.abs(data["previous_day_energy"]), 1e-12)
    )
    data["posthoc_change_quartile"] = pd.qcut(
        data["posthoc_interday_total_change"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    change_quartiles = (
        data.groupby("posthoc_change_quartile", observed=True)
        .agg(
            n=("station_id", "size"),
            variable_mean=("posthoc_interday_total_change", "mean"),
            station_candidate_nt_rate=("station_candidate_nt", "mean"),
            group_candidate_nt_rate=("group_candidate_nt", "mean"),
            final_nt_rate=("proposed_prediction_nt", "mean"),
        )
        .reset_index()
    )
    change_quartiles["diagnostic_only"] = True
    margin_sensitivity = daily_wape_margin_sensitivity(data)
    severity = positive_excess_severity(data)

    OUT.mkdir(parents=True, exist_ok=True)
    stage_rates.to_csv(OUT / "r1_negative_transfer_stage_rates.csv", index=False, encoding="utf-8-sig")
    scenario_counts.to_csv(OUT / "r1_negative_transfer_mechanism_scenarios.csv", index=False, encoding="utf-8-sig")
    guard.to_csv(OUT / "r1_negative_transfer_guard_summary.csv", index=False, encoding="utf-8-sig")
    operating.to_csv(OUT / "r1_negative_transfer_operating_conditions.csv", index=False, encoding="utf-8-sig")
    change_quartiles.to_csv(OUT / "r1_negative_transfer_posthoc_change_quartiles.csv", index=False, encoding="utf-8-sig")
    data.to_csv(OUT / "r1_negative_transfer_station_day_detail.csv", index=False, encoding="utf-8-sig")
    margin_sensitivity.to_csv(OUT / "r1_negative_transfer_margin_sensitivity.csv", index=False, encoding="utf-8-sig")
    severity.to_csv(OUT / "r1_negative_transfer_positive_severity.csv", index=False, encoding="utf-8-sig")

    print("STAGE RATES")
    print(stage_rates.to_string(index=False))
    print("\nMECHANISM SCENARIOS")
    print(scenario_counts.to_string(index=False))
    print("\nCUMULATIVE GUARD")
    print(guard.to_string(index=False))
    print("\nOPERATING CONDITIONS (mutually exclusive; Spring Festival takes precedence over weekend)")
    print(operating.to_string(index=False))
    print("\nPOST-HOC CURRENT-DAY INTERDAY CHANGE QUARTILES")
    print("NOTE: this variable uses day-d true load only for after-the-fact diagnosis; it is not available to the forecast or risk controller.")
    print(change_quartiles.to_string(index=False))
    print("\nNT DEFINITION SENSITIVITY (absolute daily-WAPE excess over baseline; diagnostic only)")
    print(margin_sensitivity.to_string(index=False))
    print("\nPOSITIVE EXCESS SEVERITY (diagnostic only)")
    print(severity.to_string(index=False))
    print("\nSCOPE NOTE: the risk controller uses prior-day recent/cumulative losses; it suppresses cumulative/tail risk but is not a pointwise guarantee that every station-day improves.")


if __name__ == "__main__":
    main()

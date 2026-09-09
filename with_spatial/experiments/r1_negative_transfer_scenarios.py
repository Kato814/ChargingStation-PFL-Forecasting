from __future__ import annotations

from pathlib import Path
import sys
import pickle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

OUTPUT_DIR = ROOT / "results_revision_r1"
EPS = 1e-12


def _daily_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, sid, station_type, date), group in frame.groupby(
        ["fold", "station_id", "type", "prediction_date"], sort=True
    ):
        y = group["y_true"].to_numpy(float)
        denom = max(float(np.abs(y).sum()), EPS)

        def error(column: str) -> float:
            return float(np.abs(y - group[column].to_numpy(float)).sum())

        base = error("previous_day_baseline")
        station = error("station_candidate")
        group_error = error("group_candidate")
        final = error("proposed_prediction")
        recent = int(group["recent_monitor_enabled"].iloc[0])
        enabled = int(group["final_enabled"].iloc[0])
        rows.append(
            {
                "fold": fold,
                "station_id": int(sid),
                "type": station_type,
                "prediction_date": pd.Timestamp(date).normalize(),
                "baseline_abs_error": base,
                "station_abs_error": station,
                "group_abs_error": group_error,
                "final_abs_error": final,
                "baseline_WAPE_day": base / denom,
                "station_WAPE_day": station / denom,
                "group_WAPE_day": group_error / denom,
                "final_WAPE_day": final / denom,
                "station_negative_transfer": int(station > base + 1e-10),
                "group_negative_transfer": int(group_error > base + 1e-10),
                "final_negative_transfer": int(final > base + 1e-10),
                "recent_enabled": recent,
                "final_enabled": enabled,
                "cumulative_guard_closed": int(recent == 1 and enabled == 0),
            }
        )

    output = pd.DataFrame(rows)
    station_nt = output["station_negative_transfer"].astype(bool)
    group_nt = output["group_negative_transfer"].astype(bool)
    output["scenario"] = np.select(
        [~station_nt & ~group_nt, station_nt & ~group_nt, ~station_nt & group_nt, station_nt & group_nt],
        ["both_improve", "group_repairs_station_nt", "group_induces_station_nt", "persistent_nt"],
        default="unknown",
    )
    return output


def _add_history_context(daily: pd.DataFrame) -> pd.DataFrame:
    with (ROOT / "data" / "clients_with_spatial.pkl").open("rb") as handle:
        clients = pickle.load(handle)["clients"]

    rows = []
    for sid, group in daily.groupby("station_id", sort=False):
        client = clients[int(sid)]
        dates = pd.to_datetime(client["forecast_date"]).normalize()
        y = np.asarray(client["y_true"], dtype=float)
        anchor = np.asarray(client["baseline"], dtype=float)
        index = {pd.Timestamp(date).normalize(): i for i, date in enumerate(dates)}
        for row in group.itertuples(index=False):
            date = pd.Timestamp(row.prediction_date).normalize()
            i = index[date]
            start = max(0, i - 7)
            hist_y = y[start:i]
            hist_anchor = anchor[start:i]
            if len(hist_y) == 0:
                context = {
                    "history7_total_cv": np.nan,
                    "history7_baseline_WAPE": np.nan,
                    "history7_interday_total_change": np.nan,
                }
            else:
                totals = hist_y.sum(axis=1)
                cv = float(np.std(totals) / max(abs(float(np.mean(totals))), EPS))
                baseline_wape = float(
                    np.abs(hist_y - hist_anchor).sum() / max(np.abs(hist_y).sum(), EPS)
                )
                if len(totals) >= 2:
                    relative_change = np.abs(np.diff(totals)) / np.maximum(np.abs(totals[:-1]), 1e-9)
                    interday_change = float(np.mean(relative_change))
                else:
                    interday_change = np.nan
                context = {
                    "history7_total_cv": cv,
                    "history7_baseline_WAPE": baseline_wape,
                    "history7_interday_total_change": interday_change,
                }
            rows.append({"station_id": int(sid), "prediction_date": date, **context})

    context = pd.DataFrame(rows)
    return daily.merge(
        context,
        on=["station_id", "prediction_date"],
        how="left",
        validate="one_to_one",
    )


def _quartile_rates(daily: pd.DataFrame, variable: str) -> pd.DataFrame:
    valid = daily.dropna(subset=[variable]).copy()
    valid["quartile"] = pd.qcut(
        valid[variable].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    return (
        valid.groupby("quartile", observed=True)
        .agg(
            n=("station_id", "size"),
            variable_mean=(variable, "mean"),
            station_nt_rate=("station_negative_transfer", "mean"),
            group_nt_rate=("group_negative_transfer", "mean"),
            final_nt_rate=("final_negative_transfer", "mean"),
        )
        .reset_index()
        .assign(variable=variable)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "results_reproduced" / "oof_predictions.csv.gz", compression="gzip")
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"]).dt.normalize()
    daily = _add_history_context(_daily_table(frame))

    stage_rates = pd.DataFrame(
        [
            {
                "stage": "station_candidate",
                "negative_transfer_days": int(daily["station_negative_transfer"].sum()),
                "total_station_days": len(daily),
                "rate": float(daily["station_negative_transfer"].mean()),
            },
            {
                "stage": "group_candidate",
                "negative_transfer_days": int(daily["group_negative_transfer"].sum()),
                "total_station_days": len(daily),
                "rate": float(daily["group_negative_transfer"].mean()),
            },
            {
                "stage": "final_prediction",
                "negative_transfer_days": int(daily["final_negative_transfer"].sum()),
                "total_station_days": len(daily),
                "rate": float(daily["final_negative_transfer"].mean()),
            },
        ]
    )
    scenarios = daily.groupby("scenario").size().rename("count").reset_index()
    scenarios["rate"] = scenarios["count"] / len(daily)
    guard = pd.DataFrame(
        [
            {
                "recent_enabled_station_days": int(daily["recent_enabled"].sum()),
                "final_enabled_station_days": int(daily["final_enabled"].sum()),
                "cumulative_guard_closed_station_days": int(daily["cumulative_guard_closed"].sum()),
            }
        ]
    )
    quartiles = pd.concat(
        [
            _quartile_rates(daily, "history7_total_cv"),
            _quartile_rates(daily, "history7_baseline_WAPE"),
            _quartile_rates(daily, "history7_interday_total_change"),
        ],
        ignore_index=True,
    )

    daily.to_csv(OUTPUT_DIR / "r1_negative_transfer_station_day.csv", index=False, encoding="utf-8-sig")
    stage_rates.to_csv(OUTPUT_DIR / "r1_negative_transfer_stage_rates.csv", index=False, encoding="utf-8-sig")
    scenarios.to_csv(OUTPUT_DIR / "r1_negative_transfer_scenarios.csv", index=False, encoding="utf-8-sig")
    guard.to_csv(OUTPUT_DIR / "r1_negative_transfer_guard_summary.csv", index=False, encoding="utf-8-sig")
    quartiles.to_csv(OUTPUT_DIR / "r1_negative_transfer_history_quartiles.csv", index=False, encoding="utf-8-sig")

    print("STAGE RATES")
    print(stage_rates.to_string(index=False))
    print("\nSCENARIOS")
    print(scenarios.to_string(index=False))
    print("\nGUARD")
    print(guard.to_string(index=False))
    print("\nHISTORY QUARTILES")
    print(quartiles.to_string(index=False))


if __name__ == "__main__":
    main()

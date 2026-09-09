from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from baselines.reference import BASELINE_COLUMN_MAP, load_locked_comparison_predictions
from src.config import load_config
from src.evaluation import aggregate_system_metrics, paired_daily_inference, station_wape_table
from src.group_total_correction import GroupCorrectionConfig, group_total_correction
from src.negative_transfer_control import (
    NegativeTransferConfig,
    cumulative_loss_guard,
    fallback_and_redistribute,
    recent_performance_eligibility,
)

OUTPUT_DIR = ROOT / "results_revision_r1"

# Methods with locked pointwise OOF evidence. The proposed method is excluded because
# this experiment asks whether the paper's post-processing also helps the baselines.
METHODS = {
    "FedAvg残差": BASELINE_COLUMN_MAP["FedAvg残差"],
    "全参数微调": BASELINE_COLUMN_MAP["全参数微调"],
    "本地MLP": BASELINE_COLUMN_MAP["本地MLP"],
    "pFedMe微调": BASELINE_COLUMN_MAP["pFedMe微调"],
}


def _fold_arrays(frame: pd.DataFrame, prediction_column: str):
    dates = sorted(pd.to_datetime(frame["prediction_date"]).dt.normalize().unique())
    stations = sorted(frame["station_id"].astype(int).unique())
    slots = sorted(frame["slot"].astype(int).unique())
    if len(stations) != 2 or slots != list(range(96)):
        raise AssertionError("expected one two-station deployment group with 96 slots")

    date_index = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    station_index = {sid: i for i, sid in enumerate(stations)}
    y = np.zeros((len(dates), len(stations), 96), dtype=np.float64)
    anchor = np.zeros_like(y)
    candidate = np.zeros_like(y)

    for row in frame.itertuples(index=False):
        di = date_index[pd.Timestamp(row.prediction_date).normalize()]
        si = station_index[int(row.station_id)]
        sl = int(row.slot)
        y[di, si, sl] = float(row.y_true)
        anchor[di, si, sl] = float(row.previous_day_baseline)
        candidate[di, si, sl] = float(getattr(row, prediction_column))
    return dates, stations, y, anchor, candidate


def _attach_stage_columns(
    frame: pd.DataFrame,
    method_slug: str,
    prediction_column: str,
    group_config: GroupCorrectionConfig,
    negative_config: NegativeTransferConfig,
) -> pd.DataFrame:
    out = frame.copy()
    group_col = f"r1_{method_slug}_group"
    final_col = f"r1_{method_slug}_group_risk"
    gamma_col = f"r1_{method_slug}_gamma"
    enabled_col = f"r1_{method_slug}_final_enabled"
    out[group_col] = np.nan
    out[final_col] = np.nan
    out[gamma_col] = np.nan
    out[enabled_col] = np.nan

    for fold, fold_frame in out.groupby("fold", sort=True):
        dates, stations, y, anchor, candidate = _fold_arrays(fold_frame, prediction_column)
        group_candidate, gamma = group_total_correction(y, anchor, candidate, group_config)
        recent = recent_performance_eligibility(y, anchor, group_candidate, negative_config)
        final_enabled = cumulative_loss_guard(y, anchor, group_candidate, recent, negative_config)
        final_prediction = fallback_and_redistribute(anchor, group_candidate, final_enabled)

        lookup = {
            (pd.Timestamp(date).normalize(), int(station), int(slot)): (
                float(group_candidate[di, si, slot]),
                float(final_prediction[di, si, slot]),
                float(gamma[di]),
                int(final_enabled[di, si]),
            )
            for di, date in enumerate(dates)
            for si, station in enumerate(stations)
            for slot in range(96)
        }
        mask = out["fold"].eq(fold)
        keys = list(
            zip(
                pd.to_datetime(out.loc[mask, "prediction_date"]).dt.normalize(),
                out.loc[mask, "station_id"].astype(int),
                out.loc[mask, "slot"].astype(int),
            )
        )
        values = [lookup[key] for key in keys]
        out.loc[mask, group_col] = [v[0] for v in values]
        out.loc[mask, final_col] = [v[1] for v in values]
        out.loc[mask, gamma_col] = [v[2] for v in values]
        out.loc[mask, enabled_col] = [v[3] for v in values]
    return out


def _stage_summary(frame: pd.DataFrame, method_name: str, column: str, stage: str) -> dict[str, object]:
    metrics = aggregate_system_metrics(frame, column)
    method_station = station_wape_table(frame, column).rename(columns={"WAPE": "method_WAPE"})
    base_station = station_wape_table(frame, "previous_day_baseline").rename(columns={"WAPE": "baseline_WAPE"})
    station = base_station.merge(method_station, on=["station_id", "type"], validate="one_to_one")
    station["relative_change"] = station["method_WAPE"] / station["baseline_WAPE"] - 1.0
    return {
        "method": method_name,
        "stage": stage,
        "WAPE": float(metrics["WAPE"]),
        "nRMSE": float(metrics["nRMSE"]),
        "pooled_WAPE": float(metrics["pooled_WAPE"]),
        "improved_stations": int((station["relative_change"] < 0).sum()),
        "max_station_relative_degradation": float(max(0.0, station["relative_change"].max())),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "paper_config.json")
    group_config = GroupCorrectionConfig(**config["group_total_correction"])
    negative_config = NegativeTransferConfig(**config["negative_transfer"])
    source = ROOT / "results_reference" / "oof_predictions_source.csv.gz"
    frame = load_locked_comparison_predictions(source)

    summaries: list[dict[str, object]] = []
    inference: dict[str, dict[str, float | int]] = {}
    enriched = frame.copy()

    for idx, (method_name, prediction_column) in enumerate(METHODS.items(), start=1):
        slug = f"m{idx}"
        enriched = _attach_stage_columns(enriched, slug, prediction_column, group_config, negative_config)
        group_col = f"r1_{slug}_group"
        final_col = f"r1_{slug}_group_risk"
        summaries.append(_stage_summary(enriched, method_name, prediction_column, "raw"))
        summaries.append(_stage_summary(enriched, method_name, group_col, "+group"))
        summaries.append(_stage_summary(enriched, method_name, final_col, "+group+risk"))
        inference[method_name] = paired_daily_inference(
            enriched,
            proposed_column="proposed_prediction",
            baseline_column=final_col,
        )

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(OUTPUT_DIR / "r1_postprocessing_fairness_summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "r1_postprocessing_fairness_daily_inference.json").write_text(
        json.dumps(inference, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stage_columns = [column for column in enriched.columns if column.startswith("r1_")]
    key_columns = [
        "fold",
        "station_id",
        "type",
        "prediction_date",
        "slot",
        "y_true",
        "previous_day_baseline",
        "proposed_prediction",
    ]
    enriched[key_columns + stage_columns].to_csv(
        OUTPUT_DIR / "r1_postprocessing_fairness_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    print(summary_frame.to_string(index=False))
    print(json.dumps(inference, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

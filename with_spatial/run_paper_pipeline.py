from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from src.config import load_config, build_folds
from src.pipeline import load_clean_clients, load_federated_statistics, run_fold, to_long_frame
from src.group_total_correction import GroupCorrectionConfig
from src.negative_transfer_control import NegativeTransferConfig
from src.evaluation import aggregate_system_metrics, station_wape_table, type_aggregate_wape_table, paired_daily_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper's proposed day-ahead method")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.output_dir or root / "results_reproduced").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(root / "paper_config.json")
    folds = build_folds(config)
    clients, feature_names = load_clean_clients(root / "data/clients_with_spatial.pkl")
    statistics = load_federated_statistics(root / "data/federated_statistics_with_spatial.pkl")

    group_config = GroupCorrectionConfig(**config["group_total_correction"])
    negative_config = NegativeTransferConfig(**config["negative_transfer"])

    fold_results = [run_fold(clients, fold, statistics, group_config, negative_config) for fold in folds]
    predictions = to_long_frame(fold_results)
    predictions.to_csv(output_dir / "oof_predictions.csv.gz", index=False, compression="gzip")

    proposed = aggregate_system_metrics(predictions, "proposed_prediction")
    baseline = aggregate_system_metrics(predictions, "previous_day_baseline")
    station_method = station_wape_table(predictions, "proposed_prediction").rename(columns={"WAPE":"method_WAPE"})
    station_base = station_wape_table(predictions, "previous_day_baseline").rename(columns={"WAPE":"baseline_WAPE"})
    station = station_base.merge(station_method, on=["station_id","type"], validate="one_to_one")
    station["relative_change"] = (station["method_WAPE"] - station["baseline_WAPE"]) / station["baseline_WAPE"]
    station.to_csv(output_dir / "station_results.csv", index=False)

    type_table = type_aggregate_wape_table(predictions, "proposed_prediction", "previous_day_baseline")
    type_table.to_csv(output_dir / "type_aggregate_results.csv", index=False)

    paired_daily = paired_daily_inference(predictions)
    (output_dir / "paired_daily_statistics.json").write_text(json.dumps(paired_daily, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "feature_dimension": len(feature_names),
        "record_count": len(predictions),
        "proposed": proposed,
        "previous_day_baseline": baseline,
        "relative_system_WAPE_reduction": (baseline["WAPE"] - proposed["WAPE"]) / baseline["WAPE"],
        "relative_pooled_WAPE_reduction": (baseline["pooled_WAPE"] - proposed["pooled_WAPE"]) / baseline["pooled_WAPE"],
        "improved_stations": int((station["relative_change"] < 0).sum()),
        "max_station_relative_degradation": float(station["relative_change"].max()),
        "enable_rate": float(predictions[["fold","station_id","prediction_date","final_enabled"]].drop_duplicates()["final_enabled"].mean()),
        "paired_daily_statistics": paired_daily,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(type_table.to_string(index=False))


if __name__ == "__main__":
    main()

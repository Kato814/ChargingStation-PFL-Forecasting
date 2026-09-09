from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import build_folds, load_config
from src.evaluation import aggregate_system_metrics, station_wape_table
from src.group_total_correction import GroupCorrectionConfig
from src.negative_transfer_control import NegativeTransferConfig, fallback_and_redistribute
from src.pipeline import load_clean_clients, load_federated_statistics, run_fold


def long_frame_for_prediction(results: list[dict], prediction_key: str, recent_only: bool = False) -> pd.DataFrame:
    rows = []
    for result in results:
        prediction = (
            fallback_and_redistribute(result["anchor"], result["group_candidate"], result["recent_eligibility"])
            if recent_only
            else result[prediction_key]
        )
        for day_index, date in enumerate(result["dates"]):
            for station_index, station_id in enumerate(result["station_ids"]):
                for slot in range(result["y_true"].shape[2]):
                    rows.append({
                        "fold": result["fold"],
                        "station_id": station_id,
                        "type": result["types"][station_index],
                        "prediction_date": date,
                        "slot": slot,
                        "y_true": result["y_true"][day_index, station_index, slot],
                        "previous_day_baseline": result["anchor"][day_index, station_index, slot],
                        "prediction": prediction[day_index, station_index, slot],
                    })
    return pd.DataFrame(rows)


def summarize(name: str, frame: pd.DataFrame) -> dict:
    metrics = aggregate_system_metrics(frame, "prediction")
    base = station_wape_table(frame, "previous_day_baseline").rename(columns={"WAPE": "baseline_WAPE"})
    method = station_wape_table(frame, "prediction").rename(columns={"WAPE": "method_WAPE"})
    station = base.merge(method, on=["station_id", "type"], validate="one_to_one")
    station["relative_change"] = (station["method_WAPE"] - station["baseline_WAPE"]) / station["baseline_WAPE"]
    return {
        "构型": name,
        "聚合WAPE": metrics["WAPE"],
        "全样本WAPE": metrics["pooled_WAPE"],
        "改善站点数": int((station["relative_change"] < 0).sum()),
        "最坏站点相对退化率/%": float(station["relative_change"].max() * 100.0),
    }


def main() -> None:
    config = load_config(ROOT / "paper_config.json")
    clients, _ = load_clean_clients(ROOT / "data/clients_with_spatial.pkl")
    statistics = load_federated_statistics(ROOT / "data/federated_statistics_with_spatial.pkl")
    gc = GroupCorrectionConfig(**config["group_total_correction"])
    nt = NegativeTransferConfig(**config["negative_transfer"])
    results = [run_fold(clients, fold, statistics, gc, nt) for fold in build_folds(config)]

    variants = [
        ("前日负荷基准", "anchor", False),
        ("标量个性化", "station_candidate", False),
        ("＋站群总量校正", "group_candidate", False),
        ("＋近期表现约束", "group_candidate", True),
        ("＋累计损失约束", "final_prediction", False),
    ]
    rows = [summarize(name, long_frame_for_prediction(results, key, recent_only)) for name, key, recent_only in variants]
    output = ROOT / "results_reproduced/table4_component_ablation.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

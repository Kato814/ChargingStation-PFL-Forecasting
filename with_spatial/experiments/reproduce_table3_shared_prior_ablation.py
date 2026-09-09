from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FoldConfig, build_folds, load_config
from src.evaluation import aggregate_system_metrics, station_wape_table
from src.federated_residual_model import FederatedRidgeResidualModel
from src.group_total_correction import GroupCorrectionConfig, group_total_correction
from src.negative_transfer_control import (
    NegativeTransferConfig,
    cumulative_loss_guard,
    fallback_and_redistribute,
    recent_performance_eligibility,
)
from src.pipeline import STATION_TYPES, fold_arrays, load_clean_clients, load_federated_statistics, station_raw_prediction, to_long_frame
from src.target_scalar_adaptation import StationScalarConfig, combine_shared_residual, rolling_scalar_adaptation


def run_variant(
    clients: dict[int, dict],
    statistics: dict[int, dict],
    folds: tuple[FoldConfig, ...],
    gc: GroupCorrectionConfig,
    nt: NegativeTransferConfig,
    fixed_type_weight: float | None,
    disable_scalar: bool,
) -> pd.DataFrame:
    fold_results = []
    for fold in folds:
        type_weight = fold.type_weight if fixed_type_weight is None else float(fixed_type_weight)
        current_fold = replace(fold, type_weight=type_weight)
        test_ids = set(current_fold.test)
        validation_ids = set(current_fold.validation)
        train_ids = set(clients) - test_ids - validation_ids

        global_model = FederatedRidgeResidualModel.fit_from_statistics(
            {sid: statistics[sid] for sid in train_ids}, alpha=current_fold.ridge_alpha
        )
        type_models = {}
        for station_type in STATION_TYPES:
            type_ids = [sid for sid in train_ids if clients[sid]["station_type"] == station_type]
            type_models[station_type] = FederatedRidgeResidualModel.fit_from_statistics(
                {sid: statistics[sid] for sid in type_ids}, alpha=current_fold.ridge_alpha
            )

        raw = []
        scalar_config = StationScalarConfig(
            type_weight=type_weight,
            history_days=current_fold.history_days,
            regularization=current_fold.regularization,
            coefficient_max=1.0,
            min_history_days=7,
        )
        for station_id in sorted(test_ids):
            item = station_raw_prediction(
                clients[station_id], global_model, type_models[clients[station_id]["station_type"]]
            )
            if disable_scalar:
                shared = combine_shared_residual(item["global_residual"], item["type_residual"], type_weight)
                item["station_candidate"] = np.maximum(0.0, item["anchor"] + shared)
                item["local_coefficient"] = np.ones(item["anchor"].shape[0], dtype=np.float64)
            else:
                item["station_candidate"], item["local_coefficient"] = rolling_scalar_adaptation(
                    item["dates"], item["y_true"], item["anchor"], item["global_residual"], item["type_residual"], scalar_config
                )
            raw.append(item)

        arrays = fold_arrays(raw)
        group_candidate, group_coefficient = group_total_correction(
            arrays["y_true"], arrays["anchor"], arrays["station_candidate"], gc
        )
        recent = recent_performance_eligibility(arrays["y_true"], arrays["anchor"], group_candidate, nt)
        final_eligibility = cumulative_loss_guard(arrays["y_true"], arrays["anchor"], group_candidate, recent, nt)
        final_prediction = fallback_and_redistribute(arrays["anchor"], group_candidate, final_eligibility)
        arrays.update({
            "group_candidate": group_candidate,
            "group_coefficient": group_coefficient,
            "recent_eligibility": recent,
            "final_eligibility": final_eligibility,
            "final_prediction": final_prediction,
            "fold": current_fold.fold,
        })
        fold_results.append(arrays)
    return to_long_frame(fold_results)


def summarize(name: str, frame: pd.DataFrame) -> dict:
    metrics = aggregate_system_metrics(frame, "proposed_prediction")
    base = station_wape_table(frame, "previous_day_baseline").rename(columns={"WAPE": "baseline_WAPE"})
    method = station_wape_table(frame, "proposed_prediction").rename(columns={"WAPE": "method_WAPE"})
    station = base.merge(method, on=["station_id", "type"], validate="one_to_one")
    station["relative_change"] = (station["method_WAPE"] - station["baseline_WAPE"]) / station["baseline_WAPE"]
    return {
        "构型": name,
        "总体WAPE": metrics["WAPE"],
        "改善站点数": int((station["relative_change"] < 0).sum()),
        "最坏站点相对退化率/%": float(station["relative_change"].max() * 100.0),
    }


def main() -> None:
    config = load_config(ROOT / "paper_config.json")
    clients, _ = load_clean_clients(ROOT / "data/clients_with_spatial.pkl")
    statistics = load_federated_statistics(ROOT / "data/federated_statistics_with_spatial.pkl")
    folds = build_folds(config)
    gc = GroupCorrectionConfig(**config["group_total_correction"])
    nt = NegativeTransferConfig(**config["negative_transfer"])

    variants = [
        ("仅全局先验", 0.0, False),
        ("仅类型先验", 1.0, False),
        ("固定等权融合", 0.5, False),
        ("验证选择或组合", None, False),
        ("无标量个性化", None, True),
    ]
    rows = []
    for name, weight, disable_scalar in variants:
        frame = run_variant(clients, statistics, folds, gc, nt, weight, disable_scalar)
        rows.append(summarize(name, frame))
    output = ROOT / "results_reproduced/table3_shared_prior_ablation.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

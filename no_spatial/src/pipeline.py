from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from .config import FoldConfig
from .federated_residual_model import FederatedRidgeResidualModel
from .target_scalar_adaptation import StationScalarConfig, rolling_scalar_adaptation
from .group_total_correction import GroupCorrectionConfig, group_total_correction
from .negative_transfer_control import (
    NegativeTransferConfig,
    recent_performance_eligibility,
    cumulative_loss_guard,
    fallback_and_redistribute,
)


TRAIN_START = pd.Timestamp("2022-09-02")
TRAIN_END = pd.Timestamp("2022-12-15")
PERSONALIZATION_START = pd.Timestamp("2022-10-17")
TEST_START = pd.Timestamp("2023-01-16")
TEST_END = pd.Timestamp("2023-02-28")
STATION_TYPES = ("pure_fast", "pure_slow", "mixed_ac_dc")


def load_clean_clients(path: Path) -> tuple[dict[int, dict], tuple[str, ...]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    clients = {int(k): v for k, v in payload["clients"].items()}
    return clients, tuple(payload["feature_names"])


def load_federated_statistics(path: Path) -> dict[int, dict]:
    with Path(path).open("rb") as handle:
        return {int(k): v for k, v in pickle.load(handle).items()}


def date_mask(client: dict, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    dates = pd.to_datetime(client["forecast_date"]).normalize()
    return np.asarray((dates >= start) & (dates <= end), dtype=bool)


def station_raw_prediction(client: dict, global_model, type_model) -> dict:
    mask = date_mask(client, PERSONALIZATION_START, TEST_END)
    dates = pd.to_datetime(np.asarray(client["forecast_date"])[mask]).normalize().tolist()
    x = np.asarray(client["x"])[mask]
    return {
        "station_id": int(client["station_id"]),
        "type": str(client["station_type"]),
        "dates": dates,
        "y_true": np.asarray(client["y_true"], dtype=np.float64)[mask],
        "anchor": np.asarray(client["baseline"], dtype=np.float64)[mask],
        "global_residual": global_model.predict_residual(x),
        "type_residual": type_model.predict_residual(x),
    }


def fold_arrays(station_predictions: list[dict]) -> dict:
    station_predictions = sorted(station_predictions, key=lambda item: item["station_id"])
    common_dates = station_predictions[0]["dates"]
    for item in station_predictions[1:]:
        if item["dates"] != common_dates:
            raise AssertionError("target stations must have aligned prediction dates")
    select = np.asarray([pd.Timestamp(d) >= TEST_START for d in common_dates], dtype=bool)
    return {
        "dates": [pd.Timestamp(d) for d, keep in zip(common_dates, select) if keep],
        "station_ids": [item["station_id"] for item in station_predictions],
        "types": [item["type"] for item in station_predictions],
        "y_true": np.stack([item["y_true"][select] for item in station_predictions], axis=1),
        "anchor": np.stack([item["anchor"][select] for item in station_predictions], axis=1),
        "global_prediction": np.stack([np.maximum(0.0, item["anchor"][select] + item["global_residual"][select]) for item in station_predictions], axis=1),
        "type_prediction": np.stack([np.maximum(0.0, item["anchor"][select] + item["type_residual"][select]) for item in station_predictions], axis=1),
        "station_candidate": np.stack([item["station_candidate"][select] for item in station_predictions], axis=1),
        "local_coefficient": np.stack([item["local_coefficient"][select] for item in station_predictions], axis=1),
    }


def run_fold(clients: dict[int, dict], fold: FoldConfig, train_statistics: dict[int, dict], group_config: GroupCorrectionConfig, negative_config: NegativeTransferConfig) -> dict:
    test_ids = set(fold.test)
    validation_ids = set(fold.validation)
    train_ids = set(clients) - test_ids - validation_ids
    global_model = FederatedRidgeResidualModel.fit_from_statistics({sid: train_statistics[sid] for sid in train_ids}, alpha=fold.ridge_alpha)
    type_models = {}
    for station_type in STATION_TYPES:
        type_ids = [sid for sid in train_ids if clients[sid]["station_type"] == station_type]
        type_models[station_type] = FederatedRidgeResidualModel.fit_from_statistics({sid: train_statistics[sid] for sid in type_ids}, alpha=fold.ridge_alpha)
    station_config = StationScalarConfig(type_weight=fold.type_weight, history_days=fold.history_days, regularization=fold.regularization, coefficient_max=1.0, min_history_days=7)
    raw = []
    for station_id in sorted(test_ids):
        item = station_raw_prediction(clients[station_id], global_model, type_models[clients[station_id]["station_type"]])
        station_candidate, local_coefficient = rolling_scalar_adaptation(item["dates"], item["y_true"], item["anchor"], item["global_residual"], item["type_residual"], station_config)
        item["station_candidate"] = station_candidate
        item["local_coefficient"] = local_coefficient
        raw.append(item)
    arrays = fold_arrays(raw)
    group_candidate, group_coefficient = group_total_correction(arrays["y_true"], arrays["anchor"], arrays["station_candidate"], group_config)
    recent_eligibility = recent_performance_eligibility(arrays["y_true"], arrays["anchor"], group_candidate, negative_config)
    final_eligibility = cumulative_loss_guard(arrays["y_true"], arrays["anchor"], group_candidate, recent_eligibility, negative_config)
    final_prediction = fallback_and_redistribute(arrays["anchor"], group_candidate, final_eligibility)
    arrays.update({"group_candidate": group_candidate, "group_coefficient": group_coefficient, "recent_eligibility": recent_eligibility, "final_eligibility": final_eligibility, "final_prediction": final_prediction, "fold": fold.fold})
    return arrays


def to_long_frame(fold_results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in fold_results:
        for day_index, date in enumerate(result["dates"]):
            for station_index, station_id in enumerate(result["station_ids"]):
                for slot in range(result["y_true"].shape[2]):
                    rows.append({
                        "fold": result["fold"], "station_id": station_id, "type": result["types"][station_index], "prediction_date": date, "slot": slot,
                        "y_true": result["y_true"][day_index, station_index, slot], "previous_day_baseline": result["anchor"][day_index, station_index, slot],
                        "global_candidate": result["global_prediction"][day_index, station_index, slot], "type_candidate": result["type_prediction"][day_index, station_index, slot],
                        "station_candidate": result["station_candidate"][day_index, station_index, slot], "group_candidate": result["group_candidate"][day_index, station_index, slot],
                        "proposed_prediction": result["final_prediction"][day_index, station_index, slot], "local_coefficient": result["local_coefficient"][day_index, station_index],
                        "group_coefficient": result["group_coefficient"][day_index], "recent_monitor_enabled": result["recent_eligibility"][day_index, station_index], "final_enabled": result["final_eligibility"][day_index, station_index],
                    })
    return pd.DataFrame(rows)

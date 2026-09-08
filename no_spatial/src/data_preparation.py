from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

TRAIN_START = pd.Timestamp("2022-09-02")
TRAIN_END = pd.Timestamp("2022-12-15")
EXPECTED_FEATURE_COUNT = 527
EXPECTED_OUTPUT_COUNT = 96
EXPECTED_STATION_COUNT = 40


def load_model_input_cache(path: Path) -> tuple[dict[int, dict], tuple[str, ...]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "clients" not in payload or "feature_names" not in payload:
        raise ValueError("model-input cache must contain 'clients' and 'feature_names'")

    feature_names = tuple(str(v) for v in payload["feature_names"])
    clients = {int(k): dict(v) for k, v in payload["clients"].items()}
    if len(clients) != EXPECTED_STATION_COUNT:
        raise ValueError(f"expected {EXPECTED_STATION_COUNT} stations, found {len(clients)}")
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"expected {EXPECTED_FEATURE_COUNT} features, found {len(feature_names)}")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature names must be unique")

    for station_id, client in clients.items():
        required = {"station_id", "station_type", "x", "y_residual", "baseline", "y_true", "forecast_date"}
        missing = required.difference(client)
        if missing:
            raise ValueError(f"station {station_id} missing fields: {sorted(missing)}")
        if int(client["station_id"]) != station_id:
            raise ValueError(f"station key/id mismatch for {station_id}")
        x = np.asarray(client["x"], dtype=np.float64)
        y_residual = np.asarray(client["y_residual"], dtype=np.float64)
        baseline = np.asarray(client["baseline"], dtype=np.float64)
        y_true = np.asarray(client["y_true"], dtype=np.float64)
        dates = pd.to_datetime(client["forecast_date"], errors="raise")
        n = len(dates)
        if x.shape != (n, EXPECTED_FEATURE_COUNT):
            raise ValueError(f"station {station_id} feature shape mismatch: {x.shape}")
        for name, arr in [("y_residual", y_residual), ("baseline", baseline), ("y_true", y_true)]:
            if arr.shape != (n, EXPECTED_OUTPUT_COUNT):
                raise ValueError(f"station {station_id} {name} shape mismatch: {arr.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"station {station_id} features contain non-finite values")
        if not np.all(np.isfinite(y_residual)) or not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(y_true)):
            raise ValueError(f"station {station_id} profile arrays contain non-finite values")
        if not np.allclose(y_residual, y_true - baseline, atol=3e-5, rtol=0.0):
            raise ValueError(f"station {station_id} residual != y_true - baseline")
        if not pd.DatetimeIndex(dates).is_monotonic_increasing:
            raise ValueError(f"station {station_id} forecast dates must be sorted")
    return clients, feature_names


def training_mask(client: dict) -> np.ndarray:
    dates = pd.to_datetime(client["forecast_date"], errors="raise").normalize()
    return np.asarray((dates >= TRAIN_START) & (dates <= TRAIN_END), dtype=bool)


def client_sufficient_statistics(client: dict) -> dict:
    mask = training_mask(client)
    x = np.asarray(client["x"], dtype=np.float64)[mask]
    y = np.asarray(client["y_residual"], dtype=np.float64)[mask]
    if x.shape[0] == 0:
        raise ValueError(f"station {client['station_id']} has no training samples")
    return {
        "n": int(x.shape[0]), "feature_width": int(x.shape[1]), "output_width": int(y.shape[1]),
        "x_sum": x.sum(axis=0), "xtx": x.T @ x, "y_sum": y.sum(axis=0), "xty": x.T @ y,
        "station_type": str(client["station_type"]), "train_start": TRAIN_START.strftime("%Y-%m-%d"), "train_end": TRAIN_END.strftime("%Y-%m-%d"),
    }


def build_federated_statistics(clients: dict[int, dict]) -> dict[int, dict]:
    return {int(station_id): client_sufficient_statistics(client) for station_id, client in sorted(clients.items())}


def save_federated_statistics(statistics: dict[int, dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)


def feature_schema_frame(feature_names: tuple[str, ...]) -> pd.DataFrame:
    def group(name: str) -> str:
        if name.startswith("prev_day_slot_"): return "previous_day_profile"
        if name.startswith("mean7_slot_"): return "recent_7day_profile"
        if name.startswith("same_dow4_slot_"): return "same_weekday_profile"
        if name.startswith("daytype14_slot_"): return "same_daytype_profile"
        if name.startswith("similar3_slot_"): return "similar_day_profile"
        if name.startswith("static_"): return "station_configuration"
        if name.startswith("dayahead_exog_"): return "calendar"
        if name.startswith("station_type_") or name.startswith("raw_station_type_"): return "station_type"
        return "historical_or_calendar_summary"

    return pd.DataFrame({"feature_index": np.arange(len(feature_names), dtype=int), "feature_name": feature_names, "feature_group": [group(name) for name in feature_names]})

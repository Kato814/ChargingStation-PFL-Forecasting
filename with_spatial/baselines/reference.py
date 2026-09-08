from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation import aggregate_system_metrics


# These column names preserve the provenance of the locked experiment outputs.
# They are intentionally not renamed to a different history-window protocol.
BASELINE_COLUMN_MAP: dict[str, str] = {
    "前日负荷基准": "previous_day_baseline",
    "FedAvg残差": "fedavg_direct_kw",
    "全参数微调": "fedavg_ft30_kw",
    "本地MLP": "local_only30_kw",
    "pFedMe微调": "pfedme_ft30_kw",
    "本文方法": "proposed_prediction",
}

TABLE2_DISPLAY_ORDER = [
    "前日负荷基准",
    "FedAvg残差",
    "全参数微调",
    "本地MLP",
    "本地岭回归",
    "pFedMe微调",
    "本文方法",
]

LOCAL_RIDGE_LOCKED_RESULT = {
    "method": "本地岭回归",
    "WAPE": 0.1025,
    "nRMSE": 0.0727,
    "source": "supplemental_locked_result",
    "raw_prediction_available": False,
}


def load_locked_comparison_predictions(path: Path) -> pd.DataFrame:
    """Load the locked OOF comparison predictions used by the paper results.

    The file contains one row per station/date/15-min slot and the neural
    comparison columns already averaged across the three deterministic seeds.
    """
    frame = pd.read_csv(Path(path), compression="infer", low_memory=False)
    required = {
        "fold",
        "station_id",
        "type",
        "prediction_date",
        "slot",
        "y_true",
        *BASELINE_COLUMN_MAP.values(),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"locked comparison predictions missing columns: {sorted(missing)}")
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"], errors="raise").dt.normalize()
    numeric_columns = ["y_true", *BASELINE_COLUMN_MAP.values()]
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite comparison values in {column}")
        frame[column] = values.astype(float)
    return frame


def reproduce_table2_from_locked_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the paper Table-2 aggregate WAPE/nRMSE rows that have raw OOF predictions.

    The local rolling ridge row is appended from its separately locked
    supplementary result because its raw point predictions are not present in
    the available final archive.
    """
    rows: list[dict[str, object]] = []
    for method, column in BASELINE_COLUMN_MAP.items():
        metrics = aggregate_system_metrics(frame, column)
        rows.append(
            {
                "method": method,
                "WAPE": metrics["WAPE"],
                "nRMSE": metrics["nRMSE"],
                "source": "locked_oof_predictions",
                "raw_prediction_available": True,
            }
        )
    rows.append(dict(LOCAL_RIDGE_LOCKED_RESULT))
    result = pd.DataFrame(rows)
    order = {name: idx for idx, name in enumerate(TABLE2_DISPLAY_ORDER)}
    result["__order"] = result["method"].map(order)
    return result.sort_values("__order").drop(columns="__order").reset_index(drop=True)

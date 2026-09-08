from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StationScalarConfig:
    type_weight: float
    history_days: int
    regularization: float
    coefficient_max: float = 1.0
    min_history_days: int = 7


def combine_shared_residual(
    global_residual: np.ndarray,
    type_residual: np.ndarray,
    type_weight: float,
) -> np.ndarray:
    """Equation (3): combine global and type residual priors."""
    return (1.0 - float(type_weight)) * np.asarray(global_residual) + float(type_weight) * np.asarray(type_residual)


def rolling_scalar_adaptation(
    dates: Sequence[pd.Timestamp],
    y_true: np.ndarray,
    baseline: np.ndarray,
    global_residual: np.ndarray,
    type_residual: np.ndarray,
    config: StationScalarConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Causally estimate one station-specific scalar for each forecast day."""
    y = np.asarray(y_true, dtype=np.float64)
    anchor = np.asarray(baseline, dtype=np.float64)
    shared = combine_shared_residual(global_residual, type_residual, config.type_weight).astype(np.float64)
    if y.shape != anchor.shape or y.shape != shared.shape:
        raise ValueError("profile arrays must have identical shape")

    prediction = anchor.copy()
    coefficients = np.zeros(y.shape[0], dtype=np.float64)
    normalized_dates = [pd.Timestamp(v).normalize() for v in dates]
    if len(normalized_dates) != y.shape[0]:
        raise ValueError("date count does not match profile count")

    for index in range(y.shape[0]):
        start = max(0, index - config.history_days)
        history_size = index - start
        if history_size < config.min_history_days:
            continue
        residual_history = shared[start:index].reshape(-1)
        target_history = (y[start:index] - anchor[start:index]).reshape(-1)
        scale = max(float(np.mean(np.square(residual_history))), 1e-8)
        denominator = (
            float(np.dot(residual_history, residual_history))
            + config.regularization * scale * residual_history.size
            + 1e-9
        )
        alpha = float(
            np.clip(
                np.dot(residual_history, target_history) / denominator,
                0.0,
                config.coefficient_max,
            )
        )
        prediction[index] = np.maximum(0.0, anchor[index] + alpha * shared[index])
        coefficients[index] = alpha
    return prediction, coefficients

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GroupCorrectionConfig:
    history_days: int = 5
    regularization: float = 0.0
    coefficient_max: float = 1.0
    min_history_days: int = 5


def group_total_correction(y_true: np.ndarray, baseline: np.ndarray, personalized_candidate: np.ndarray, config: GroupCorrectionConfig) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64)
    anchor = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(personalized_candidate, dtype=np.float64)
    if y.shape != anchor.shape or y.shape != candidate.shape or y.ndim != 3:
        raise ValueError("group arrays must share [days, stations, slots] shape")
    correction = candidate - anchor
    aggregate_correction = correction.sum(axis=1)
    aggregate_target = (y - anchor).sum(axis=1)
    output = anchor.copy()
    coefficients = np.zeros(y.shape[0], dtype=np.float64)
    station_count = y.shape[1]
    for day_index in range(y.shape[0]):
        start = max(0, day_index - config.history_days)
        if day_index - start < config.min_history_days:
            continue
        c_hist = aggregate_correction[start:day_index].reshape(-1)
        r_hist = aggregate_target[start:day_index].reshape(-1)
        scale = max(float(np.mean(np.square(c_hist))), 1e-8)
        denominator = float(np.dot(c_hist, c_hist)) + config.regularization * scale * c_hist.size + 1e-9
        gamma = float(np.clip(np.dot(c_hist, r_hist) / denominator, 0.0, config.coefficient_max))
        coefficients[day_index] = gamma
        total_difference_per_station = (gamma - 1.0) * aggregate_correction[day_index] / station_count
        output[day_index] = np.maximum(0.0, candidate[day_index] + total_difference_per_station[None, :])
    return output, coefficients

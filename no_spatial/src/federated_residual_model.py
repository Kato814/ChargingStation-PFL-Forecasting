from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class FederatedScaler:
    mean_: np.ndarray
    scale_: np.ndarray
    sample_count_: int

    def transform(self, x: np.ndarray) -> np.ndarray:
        raw = np.asarray(x, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != self.mean_.shape[0]:
            raise ValueError("feature array has incompatible shape")
        return (raw - self.mean_) / self.scale_


@dataclass(frozen=True)
class FederatedRidgeResidualModel:
    scaler: FederatedScaler
    coefficient_: np.ndarray
    intercept_: np.ndarray
    alpha: float
    participating_clients: tuple[int, ...]

    @classmethod
    def fit_from_statistics(cls, client_statistics: Mapping[int, Mapping[str, np.ndarray | int | str]], alpha: float) -> "FederatedRidgeResidualModel":
        if alpha < 0 or not client_statistics:
            raise ValueError("non-empty statistics and non-negative alpha are required")
        ordered = [(int(k), v) for k, v in client_statistics.items()]
        n = sum(int(stat["n"]) for _, stat in ordered)
        x_sum = sum((np.asarray(stat["x_sum"], dtype=np.float64) for _, stat in ordered), start=0)
        xtx = sum((np.asarray(stat["xtx"], dtype=np.float64) for _, stat in ordered), start=0)
        y_sum = sum((np.asarray(stat["y_sum"], dtype=np.float64) for _, stat in ordered), start=0)
        xty = sum((np.asarray(stat["xty"], dtype=np.float64) for _, stat in ordered), start=0)
        mean = x_sum / n
        variance = np.maximum(np.diag(xtx) / n - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < 1e-12] = 1.0
        centered_xtx = xtx - n * np.outer(mean, mean)
        ztz = centered_xtx / scale[:, None] / scale[None, :]
        centered_xty = xty - np.outer(mean, y_sum)
        zty = centered_xty / scale[:, None]
        coefficient = np.linalg.solve(ztz + float(alpha) * np.eye(ztz.shape[0]), zty)
        intercept = y_sum / n
        return cls(FederatedScaler(mean, scale, n), coefficient, intercept, float(alpha), tuple(sorted(k for k, _ in ordered)))

    def predict_residual(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x) @ self.coefficient_ + self.intercept_

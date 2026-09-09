from __future__ import annotations

import numpy as np
import pandas as pd


def metric_bundle(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    error = y - p
    denominator = max(float(np.abs(y).sum()), 1e-12)
    peak = max(float(np.max(y)), 1e-12)
    sst = float(np.square(y - y.mean()).sum())
    return {
        "WAPE": float(np.abs(error).sum() / denominator),
        "nMAE": float(np.abs(error).mean() / peak),
        "nRMSE": float(np.sqrt(np.square(error).mean()) / peak),
        "R2": float(1.0 - np.square(error).sum() / max(sst, 1e-12)),
    }


def aggregate_system_metrics(data: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    system = data.groupby(["prediction_date", "slot"], as_index=False).agg(
        y_true=("y_true", "sum"), prediction=(prediction_column, "sum")
    )
    metrics = metric_bundle(system["y_true"], system["prediction"])
    metrics["pooled_WAPE"] = float(
        np.abs(data["y_true"] - data[prediction_column]).sum() / max(np.abs(data["y_true"]).sum(), 1e-12)
    )
    return metrics


def station_wape_table(data: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    rows = []
    for (station_id, station_type), group in data.groupby(["station_id", "type"]):
        denom = max(float(np.abs(group["y_true"]).sum()), 1e-12)
        rows.append({
            "station_id": int(station_id),
            "type": station_type,
            "WAPE": float(np.abs(group["y_true"] - group[prediction_column]).sum() / denom),
        })
    return pd.DataFrame(rows)


def type_aggregate_wape_table(data: pd.DataFrame, prediction_column: str, baseline_column: str) -> pd.DataFrame:
    """Paper definition: aggregate stations of the same type at each date/slot first."""
    rows = []
    for station_type, group in data.groupby("type"):
        agg = group.groupby(["prediction_date", "slot"], as_index=False).agg(
            y_true=("y_true", "sum"), prediction=(prediction_column, "sum"), baseline=(baseline_column, "sum")
        )
        denom = max(float(np.abs(agg["y_true"]).sum()), 1e-12)
        method_wape = float(np.abs(agg["y_true"] - agg["prediction"]).sum() / denom)
        baseline_wape = float(np.abs(agg["y_true"] - agg["baseline"]).sum() / denom)
        rows.append({
            "type": station_type,
            "baseline_WAPE": baseline_wape,
            "method_WAPE": method_wape,
            "relative_improvement_percent": (baseline_wape - method_wape) / baseline_wape * 100.0,
        })
    return pd.DataFrame(rows)


def deployment_group_daily_wape(data: pd.DataFrame, prediction_column: str) -> pd.Series:
    """Daily WAPE preserving each fold's two-station deployment-group boundary."""
    required = {"prediction_date", "fold", "slot", "y_true", prediction_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing columns for deployment-group daily WAPE: {sorted(missing)}")
    group_slot = data.groupby(["prediction_date", "fold", "slot"], as_index=False).agg(
        y_true=("y_true", "sum"), prediction=(prediction_column, "sum")
    )
    group_slot["absolute_error"] = np.abs(group_slot["y_true"] - group_slot["prediction"])
    group_slot["actual_total"] = np.abs(group_slot["y_true"])
    daily = group_slot.groupby("prediction_date", sort=True).agg(
        absolute_error=("absolute_error", "sum"), actual_total=("actual_total", "sum")
    )
    return daily["absolute_error"] / daily["actual_total"].clip(lower=1e-12)


def circular_moving_block_bootstrap_mean(values: np.ndarray, block_length: int = 7, repetitions: int = 50_000, seed: int = 20260716) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("bootstrap values must be non-empty")
    if block_length <= 0 or repetitions <= 0:
        raise ValueError("block_length and repetitions must be positive")
    n = values.size
    rng = np.random.default_rng(seed)
    blocks_per_sample = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(repetitions, blocks_per_sample))
    offsets = np.arange(block_length)[None, None, :]
    indices = (starts[:, :, None] + offsets) % n
    samples = values[indices].reshape(repetitions, -1)[:, :n]
    return samples.mean(axis=1)


def newey_west_mean_test(values: np.ndarray, max_lag: int = 7) -> dict[str, float]:
    from scipy.stats import norm
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("Newey-West test requires at least two observations")
    n = values.size
    mean = float(values.mean())
    centered = values - mean
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * autocovariance
    standard_error = float(np.sqrt(max(long_run_variance / n, 0.0)))
    z_value = float(mean / standard_error) if standard_error > 0 else float("nan")
    p_two_sided = float(2.0 * norm.sf(abs(z_value))) if np.isfinite(z_value) else float("nan")
    return {"mean_difference": mean, "standard_error": standard_error, "z_value": z_value, "p_two_sided": p_two_sided}


def paired_daily_inference(data: pd.DataFrame, proposed_column: str = "proposed_prediction", baseline_column: str = "previous_day_baseline", block_length: int = 7, bootstrap_repetitions: int = 50_000, bootstrap_seed: int = 20260716, newey_west_lag: int = 7) -> dict[str, float | int]:
    proposed = deployment_group_daily_wape(data, proposed_column)
    baseline = deployment_group_daily_wape(data, baseline_column)
    proposed, baseline = proposed.align(baseline, join="inner")
    difference = (proposed - baseline).to_numpy(dtype=np.float64)
    bootstrap = circular_moving_block_bootstrap_mean(difference, block_length=block_length, repetitions=bootstrap_repetitions, seed=bootstrap_seed)
    nw = newey_west_mean_test(difference, max_lag=newey_west_lag)
    return {
        "n_prediction_days": int(difference.size),
        "mean_daily_WAPE_difference": float(difference.mean()),
        "circular_mbb_95_low": float(np.quantile(bootstrap, 0.025)),
        "circular_mbb_95_high": float(np.quantile(bootstrap, 0.975)),
        "block_length_days": int(block_length),
        "newey_west_lag": int(newey_west_lag),
        "newey_west_standard_error": float(nw["standard_error"]),
        "newey_west_z": float(nw["z_value"]),
        "newey_west_p_two_sided": float(nw["p_two_sided"]),
    }

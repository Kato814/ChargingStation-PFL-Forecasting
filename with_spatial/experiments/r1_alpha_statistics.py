from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_revision_r1"

from src.config import build_folds, load_config
from src.federated_residual_model import FederatedRidgeResidualModel
from src.pipeline import (
    STATION_TYPES,
    TEST_START,
    load_clean_clients,
    load_federated_statistics,
    station_raw_prediction,
)
from src.target_scalar_adaptation import StationScalarConfig, combine_shared_residual, rolling_scalar_adaptation

VARIABLES = ["shared_history_WAPE", "shared_history_gain", "load_total_cv", "residual_rms"]
BOOTSTRAP_SAMPLE_SIZES = (7, 14, 30, 60)
BOOTSTRAP_REPS = 300
BOOTSTRAP_SEED = 20260910


def alpha_from_day_summaries(cross: np.ndarray, ssq: np.ndarray, n_slots: int, rho: float) -> np.ndarray:
    """Vectorized alpha for bootstrap samples from day-level dot-product summaries."""
    cross_sum = np.asarray(cross, dtype=float).sum(axis=1)
    ssq_sum = np.asarray(ssq, dtype=float).sum(axis=1)
    n_days = np.asarray(cross).shape[1]
    mean_sq = ssq_sum / max(n_days * n_slots, 1)
    scale = np.maximum(mean_sq, 1e-8)
    denominator = ssq_sum + rho * scale * (n_days * n_slots) + 1e-9
    return np.clip(cross_sum / denominator, 0.0, 1.0)


def build_records():
    config = load_config(ROOT / "paper_config.json")
    folds = build_folds(config)
    clients, _ = load_clean_clients(ROOT / "data/clients_with_spatial.pkl")
    stats = load_federated_statistics(ROOT / "data/federated_statistics_with_spatial.pkl")

    records = []
    raw_cache = {}
    max_alpha_diff = 0.0

    for fold in folds:
        test_ids = set(fold.test)
        validation_ids = set(fold.validation)
        train_ids = set(clients) - test_ids - validation_ids
        global_model = FederatedRidgeResidualModel.fit_from_statistics(
            {sid: stats[sid] for sid in train_ids}, alpha=fold.ridge_alpha
        )
        type_models = {}
        for station_type in STATION_TYPES:
            type_ids = [sid for sid in train_ids if clients[sid]["station_type"] == station_type]
            type_models[station_type] = FederatedRidgeResidualModel.fit_from_statistics(
                {sid: stats[sid] for sid in type_ids}, alpha=fold.ridge_alpha
            )

        for station_id in sorted(test_ids):
            item = station_raw_prediction(
                clients[station_id], global_model, type_models[clients[station_id]["station_type"]]
            )
            dates = pd.to_datetime(item["dates"]).normalize()
            y = np.asarray(item["y_true"], dtype=float)
            anchor = np.asarray(item["anchor"], dtype=float)
            shared = combine_shared_residual(
                item["global_residual"], item["type_residual"], fold.type_weight
            ).astype(float)
            true_residual = y - anchor

            station_config = StationScalarConfig(
                type_weight=fold.type_weight,
                history_days=fold.history_days,
                regularization=fold.regularization,
                coefficient_max=1.0,
                min_history_days=7,
            )
            _, alpha = rolling_scalar_adaptation(
                item["dates"], y, anchor,
                item["global_residual"], item["type_residual"], station_config
            )

            # Compare reconstructed test alpha to the frozen OOF alpha.
            frozen = pd.read_csv(
                ROOT / "results_reproduced/oof_predictions.csv.gz",
                usecols=["fold", "station_id", "prediction_date", "local_coefficient"],
            )
            frozen["prediction_date"] = pd.to_datetime(frozen["prediction_date"]).dt.normalize()
            frozen_station = (
                frozen[(frozen["fold"] == fold.fold) & (frozen["station_id"] == station_id)]
                .drop_duplicates(["prediction_date"])
                .set_index("prediction_date")["local_coefficient"]
            )
            test_indices = np.where(dates >= TEST_START)[0]
            for idx in test_indices:
                if dates[idx] in frozen_station.index:
                    max_alpha_diff = max(max_alpha_diff, abs(float(alpha[idx]) - float(frozen_station.loc[dates[idx]])))

                start = max(0, idx - fold.history_days)
                shared_hist = shared[start:idx]
                true_res_hist = true_residual[start:idx]
                y_hist = y[start:idx]
                anchor_hist = anchor[start:idx]
                shared_pred_hist = np.maximum(0.0, anchor_hist + shared_hist)
                denom = max(float(np.abs(y_hist).sum()), 1e-12)
                shared_wape = float(np.abs(y_hist - shared_pred_hist).sum() / denom)
                baseline_wape = float(np.abs(y_hist - anchor_hist).sum() / denom)
                gain = float((baseline_wape - shared_wape) / max(baseline_wape, 1e-12))
                totals = y_hist.sum(axis=1)
                load_cv = float(totals.std() / (abs(totals.mean()) + 1e-12))
                residual_rms = float(np.sqrt(np.mean(np.square(true_res_hist))))
                records.append({
                    "fold": fold.fold,
                    "station_id": station_id,
                    "type": clients[station_id]["station_type"],
                    "prediction_date": dates[idx],
                    "alpha": float(alpha[idx]),
                    "H_main": fold.history_days,
                    "shared_history_WAPE": shared_wape,
                    "shared_history_gain": gain,
                    "load_total_cv": load_cv,
                    "residual_rms": residual_rms,
                })

            raw_cache[station_id] = {
                "fold": fold,
                "dates": dates,
                "shared": shared,
                "true_residual": true_residual,
            }

    return pd.DataFrame(records), raw_cache, max_alpha_diff


def stationwise_correlations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_id, group in data.groupby("station_id", sort=True):
        row = {
            "station_id": int(station_id),
            "type": group["type"].iloc[0],
            "n_days": len(group),
        }
        for variable in VARIABLES:
            result = spearmanr(group["alpha"], group[variable])
            row[f"rho_{variable}"] = float(result.statistic)
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_summary(data: pd.DataFrame, station_corr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows = []
    station_rows = []
    expected_direction = {
        "shared_history_WAPE": "negative",
        "shared_history_gain": "positive",
        "load_total_cv": "mixed",
        "residual_rms": "mixed",
    }
    for variable in VARIABLES:
        pooled = spearmanr(data["alpha"], data[variable])
        pooled_rows.append({
            "variable": variable,
            "rho": float(pooled.statistic),
            "p_value_descriptive_only": float(pooled.pvalue),
            "n_station_days": len(data),
            "note": "station-days are repeated measures; pooled p-value is not the primary inferential evidence",
        })
        values = station_corr[f"rho_{variable}"].to_numpy(dtype=float)
        direction = expected_direction[variable]
        # Use a two-sided station-level Wilcoxon test for all variables.  This
        # avoids post-hoc directional inference while the sign counts and
        # median station rho retain the physically expected direction.
        w = wilcoxon(values, alternative="two-sided", method="exact")
        station_rows.append({
            "variable": variable,
            "station_count": len(values),
            "median_station_rho": float(np.median(values)),
            "min_station_rho": float(np.min(values)),
            "max_station_rho": float(np.max(values)),
            "positive_station_count": int((values > 0).sum()),
            "negative_station_count": int((values < 0).sum()),
            "expected_direction_descriptive": direction,
            "wilcoxon_alternative": "two-sided",
            "wilcoxon_p": float(w.pvalue),
        })
    return pd.DataFrame(pooled_rows), pd.DataFrame(station_rows)


def fixed_pool_sample_size_stability(raw_cache: dict[int, dict]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for station_id, item in sorted(raw_cache.items()):
        fold = item["fold"]
        dates = item["dates"]
        shared = item["shared"]
        target = item["true_residual"]
        test_indices = np.where(dates >= TEST_START)[0]
        for idx in test_indices:
            pool = np.arange(idx - 60, idx)
            if pool.min() < 0 or len(pool) != 60:
                raise AssertionError(f"station {station_id} lacks a fixed 60-day preforecast pool on {dates[idx]}")
            # Day-level sufficient pieces; whole days are resampled so each sampled day retains all 96 slots.
            day_cross = np.einsum("ij,ij->i", shared[pool], target[pool])
            day_ssq = np.einsum("ij,ij->i", shared[pool], shared[pool])
            n_slots = shared.shape[1]
            full60 = alpha_from_day_summaries(day_cross[None, :], day_ssq[None, :], n_slots, fold.regularization)[0]
            for n_days in BOOTSTRAP_SAMPLE_SIZES:
                draw = rng.integers(0, 60, size=(BOOTSTRAP_REPS, n_days))
                alpha_boot = alpha_from_day_summaries(
                    day_cross[draw], day_ssq[draw], n_slots, fold.regularization
                )
                rows.append({
                    "station_id": station_id,
                    "prediction_date": dates[idx],
                    "sample_days": n_days,
                    "bootstrap_reps": BOOTSTRAP_REPS,
                    "bootstrap_alpha_sd": float(np.std(alpha_boot, ddof=1)),
                    "bootstrap_alpha_mean": float(np.mean(alpha_boot)),
                    "full60_alpha": float(full60),
                    "bootstrap_rmse_to_full60": float(np.sqrt(np.mean(np.square(alpha_boot - full60)))),
                })
    return pd.DataFrame(rows)


def main() -> None:
    data, raw_cache, max_alpha_diff = build_records()
    station_corr = stationwise_correlations(data)
    pooled, station_summary = correlation_summary(data, station_corr)

    # Type description uses station-level mean alpha, not 440 station-days as if independent.
    station_alpha = data.groupby(["station_id", "type"], as_index=False)["alpha"].mean()
    type_summary = (
        station_alpha.groupby("type")
        .agg(
            station_count=("station_id", "size"),
            mean_station_alpha=("alpha", "mean"),
            median_station_alpha=("alpha", "median"),
            min_station_alpha=("alpha", "min"),
            max_station_alpha=("alpha", "max"),
        )
        .reset_index()
    )

    stability = fixed_pool_sample_size_stability(raw_cache)
    stability_summary = (
        stability.groupby("sample_days")
        .agg(
            station_day_count=("station_id", "size"),
            mean_bootstrap_alpha_sd=("bootstrap_alpha_sd", "mean"),
            median_bootstrap_alpha_sd=("bootstrap_alpha_sd", "median"),
            mean_rmse_to_full60=("bootstrap_rmse_to_full60", "mean"),
            median_rmse_to_full60=("bootstrap_rmse_to_full60", "median"),
        )
        .reset_index()
    )

    # Check whether uncertainty decreases for every station-day as n increases.
    pivot = stability.pivot_table(
        index=["station_id", "prediction_date"], columns="sample_days", values="bootstrap_alpha_sd"
    )
    monotone = (pivot[7] > pivot[14]) & (pivot[14] > pivot[30]) & (pivot[30] > pivot[60])
    stability_meta = pd.DataFrame([{
        "fixed_history_pool_days": 60,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "station_days": len(pivot),
        "strictly_decreasing_sd_station_days": int(monotone.sum()),
        "strictly_decreasing_sd_rate": float(monotone.mean()),
        "interpretation": "whole-day bootstrap within the same 60-day preforecast pool; varies day sample count without changing history span",
    }])

    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "r1_alpha_station_day_statistics.csv", index=False, encoding="utf-8-sig")
    station_corr.to_csv(OUT / "r1_alpha_stationwise_correlations.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(OUT / "r1_alpha_pooled_correlations_descriptive.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(OUT / "r1_alpha_station_correlation_summary.csv", index=False, encoding="utf-8-sig")
    type_summary.to_csv(OUT / "r1_alpha_type_summary_station_level.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(OUT / "r1_alpha_fixed60_sample_size_bootstrap_detail.csv", index=False, encoding="utf-8-sig")
    stability_summary.to_csv(OUT / "r1_alpha_fixed60_sample_size_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    stability_meta.to_csv(OUT / "r1_alpha_fixed60_sample_size_bootstrap_meta.csv", index=False, encoding="utf-8-sig")

    print("MAX ALPHA DIFF", max_alpha_diff)
    if max_alpha_diff > 1e-10:
        raise AssertionError(f"reconstructed alpha differs from frozen OOF: {max_alpha_diff}")
    print("\nPOOLED CORRELATIONS (descriptive; repeated station-days, p-values are not primary evidence)")
    print(pooled.to_string(index=False))
    print("\nSTATION-WISE CORRELATION SUMMARY (primary relationship evidence)")
    print(station_summary.to_string(index=False))
    print("\nTYPE SUMMARY (station-level means; descriptive because fast/mixed each contain only 2 target stations)")
    print(type_summary.to_string(index=False))
    print("\nFIXED-60-DAY-POOL SAMPLE-SIZE STABILITY")
    print(stability_summary.to_string(index=False))
    print(stability_meta.to_string(index=False))


if __name__ == "__main__":
    main()

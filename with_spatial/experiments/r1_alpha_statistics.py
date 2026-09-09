from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

OUTPUT_DIR = ROOT / "results_revision_r1"
EPS = 1e-12


def _fit_models(clients, statistics, fold):
    test_ids = set(fold.test)
    validation_ids = set(fold.validation)
    train_ids = set(clients) - test_ids - validation_ids
    global_model = FederatedRidgeResidualModel.fit_from_statistics(
        {sid: statistics[sid] for sid in train_ids}, alpha=fold.ridge_alpha
    )
    type_models = {}
    for station_type in STATION_TYPES:
        ids = [sid for sid in train_ids if clients[sid]["station_type"] == station_type]
        type_models[station_type] = FederatedRidgeResidualModel.fit_from_statistics(
            {sid: statistics[sid] for sid in ids}, alpha=fold.ridge_alpha
        )
    return global_model, type_models


def _history_metrics(y, anchor, shared_residual, totals, index, history_days):
    start = max(0, index - history_days)
    hist_y = y[start:index]
    hist_anchor = anchor[start:index]
    hist_shared = shared_residual[start:index]
    if len(hist_y) == 0:
        return {
            "history_days_effective": 0,
            "history_points_effective": 0,
            "shared_history_WAPE": np.nan,
            "baseline_history_WAPE": np.nan,
            "shared_history_gain": np.nan,
            "load_total_cv": np.nan,
            "residual_rms": np.nan,
        }

    shared_prediction = np.maximum(0.0, hist_anchor + hist_shared)
    denominator = max(float(np.abs(hist_y).sum()), EPS)
    shared_loss = float(np.abs(hist_y - shared_prediction).sum())
    baseline_loss = float(np.abs(hist_y - hist_anchor).sum())
    history_totals = totals[start:index]
    cv = float(np.std(history_totals) / max(abs(float(np.mean(history_totals))), EPS))
    residual_rms = float(np.sqrt(np.mean(np.square(hist_y - hist_anchor))))
    return {
        "history_days_effective": int(len(hist_y)),
        "history_points_effective": int(len(hist_y) * hist_y.shape[1]),
        "shared_history_WAPE": shared_loss / denominator,
        "baseline_history_WAPE": baseline_loss / denominator,
        "shared_history_gain": (baseline_loss - shared_loss) / max(baseline_loss, EPS),
        "load_total_cv": cv,
        "residual_rms": residual_rms,
    }


def _correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan, int(valid.sum())
    rho, p_value = spearmanr(np.asarray(x)[valid], np.asarray(y)[valid])
    return float(rho), float(p_value), int(valid.sum())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "paper_config.json")
    folds = build_folds(config)
    clients, _ = load_clean_clients(ROOT / "data" / "clients_with_spatial.pkl")
    statistics = load_federated_statistics(ROOT / "data" / "federated_statistics_with_spatial.pkl")

    frozen = pd.read_csv(ROOT / "results_reproduced" / "oof_predictions.csv.gz", compression="gzip")
    frozen["prediction_date"] = pd.to_datetime(frozen["prediction_date"]).dt.normalize()
    frozen_alpha = frozen[["fold", "station_id", "prediction_date", "local_coefficient"]].drop_duplicates()

    rows = []
    controlled_rows = []
    for fold in folds:
        global_model, type_models = _fit_models(clients, statistics, fold)
        for station_id in sorted(fold.test):
            client = clients[station_id]
            raw = station_raw_prediction(
                client,
                global_model,
                type_models[client["station_type"]],
            )
            shared = combine_shared_residual(
                raw["global_residual"],
                raw["type_residual"],
                fold.type_weight,
            )
            totals = np.asarray(raw["y_true"], dtype=float).sum(axis=1)
            scalar_config = StationScalarConfig(
                type_weight=fold.type_weight,
                history_days=fold.history_days,
                regularization=fold.regularization,
                coefficient_max=1.0,
                min_history_days=7,
            )
            _, alpha = rolling_scalar_adaptation(
                raw["dates"],
                raw["y_true"],
                raw["anchor"],
                raw["global_residual"],
                raw["type_residual"],
                scalar_config,
            )
            for index, date in enumerate(raw["dates"]):
                prediction_date = pd.Timestamp(date).normalize()
                if prediction_date < TEST_START:
                    continue
                metrics = _history_metrics(
                    raw["y_true"],
                    raw["anchor"],
                    shared,
                    totals,
                    index,
                    fold.history_days,
                )
                rows.append(
                    {
                        "fold": fold.fold,
                        "station_id": int(station_id),
                        "type": client["station_type"],
                        "prediction_date": prediction_date,
                        "alpha": float(alpha[index]),
                        "configured_history_days": int(fold.history_days),
                        "regularization": float(fold.regularization),
                        **metrics,
                    }
                )

            # Controlled history-length analysis: only H changes.
            for history_days in (7, 14, 30, 60):
                controlled_config = StationScalarConfig(
                    type_weight=fold.type_weight,
                    history_days=history_days,
                    regularization=fold.regularization,
                    coefficient_max=1.0,
                    min_history_days=7,
                )
                _, controlled_alpha = rolling_scalar_adaptation(
                    raw["dates"],
                    raw["y_true"],
                    raw["anchor"],
                    raw["global_residual"],
                    raw["type_residual"],
                    controlled_config,
                )
                for index, date in enumerate(raw["dates"]):
                    prediction_date = pd.Timestamp(date).normalize()
                    if prediction_date < TEST_START:
                        continue
                    metrics = _history_metrics(
                        raw["y_true"],
                        raw["anchor"],
                        shared,
                        totals,
                        index,
                        history_days,
                    )
                    controlled_rows.append(
                        {
                            "fold": fold.fold,
                            "station_id": int(station_id),
                            "type": client["station_type"],
                            "prediction_date": prediction_date,
                            "H": history_days,
                            "alpha": float(controlled_alpha[index]),
                            **metrics,
                        }
                    )

    data = pd.DataFrame(rows)
    controlled = pd.DataFrame(controlled_rows)

    merged = data.merge(
        frozen_alpha,
        on=["fold", "station_id", "prediction_date"],
        how="left",
        validate="one_to_one",
    )
    max_alpha_difference = float(np.max(np.abs(merged["alpha"] - merged["local_coefficient"])))
    if max_alpha_difference > 1e-10:
        raise AssertionError(
            f"recomputed alpha differs from frozen OOF: {max_alpha_difference:.3e}"
        )

    variables = [
        "shared_history_WAPE",
        "shared_history_gain",
        "load_total_cv",
        "residual_rms",
    ]
    pooled_rows = []
    station_rows = []
    for variable in variables:
        rho, p_value, n = _correlation(data["alpha"].to_numpy(float), data[variable].to_numpy(float))
        pooled_rows.append(
            {"scope": "pooled", "variable": variable, "rho": rho, "p_value": p_value, "n": n}
        )
        for station_id, group in data.groupby("station_id", sort=True):
            rho, p_value, n = _correlation(
                group["alpha"].to_numpy(float), group[variable].to_numpy(float)
            )
            station_rows.append(
                {
                    "station_id": int(station_id),
                    "type": group["type"].iloc[0],
                    "variable": variable,
                    "rho": rho,
                    "p_value": p_value,
                    "n": n,
                }
            )

    pooled = pd.DataFrame(pooled_rows)
    by_station = pd.DataFrame(station_rows)
    type_summary = (
        data.groupby("type")
        .agg(
            n=("alpha", "size"),
            alpha_mean=("alpha", "mean"),
            alpha_median=("alpha", "median"),
            alpha_std=("alpha", "std"),
        )
        .reset_index()
    )

    history_rows = []
    for history_days, group in controlled.groupby("H", sort=True):
        station_std = group.groupby("station_id")["alpha"].std()
        rho_gain, p_gain, n_gain = _correlation(
            group["alpha"].to_numpy(float), group["shared_history_gain"].to_numpy(float)
        )
        rho_error, p_error, _ = _correlation(
            group["alpha"].to_numpy(float), group["shared_history_WAPE"].to_numpy(float)
        )
        history_rows.append(
            {
                "H": int(history_days),
                "mean_station_alpha_std": float(station_std.mean()),
                "median_station_alpha_std": float(station_std.median()),
                "pooled_rho_alpha_shared_gain": rho_gain,
                "p_alpha_shared_gain": p_gain,
                "pooled_rho_alpha_shared_WAPE": rho_error,
                "p_alpha_shared_WAPE": p_error,
                "n": n_gain,
            }
        )
    history_summary = pd.DataFrame(history_rows)

    data.to_csv(OUTPUT_DIR / "r1_alpha_station_day.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(OUTPUT_DIR / "r1_alpha_pooled_correlations.csv", index=False, encoding="utf-8-sig")
    by_station.to_csv(OUTPUT_DIR / "r1_alpha_station_correlations.csv", index=False, encoding="utf-8-sig")
    type_summary.to_csv(OUTPUT_DIR / "r1_alpha_type_summary.csv", index=False, encoding="utf-8-sig")
    controlled.to_csv(
        OUTPUT_DIR / "r1_alpha_history_length_controlled.csv.gz",
        index=False,
        compression="gzip",
    )
    history_summary.to_csv(
        OUTPUT_DIR / "r1_alpha_history_length_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "r1_alpha_reproduction_check.json").write_text(
        json.dumps({"max_abs_difference_vs_frozen_alpha": max_alpha_difference}, indent=2),
        encoding="utf-8",
    )

    print("MAX ALPHA DIFF", max_alpha_difference)
    print("\nPOOLED")
    print(pooled.to_string(index=False))
    print("\nSTATION CORR MEDIANS")
    print(by_station.groupby("variable")["rho"].agg(["median", "min", "max"]).to_string())
    print("\nTYPE SUMMARY")
    print(type_summary.to_string(index=False))
    print("\nH SUMMARY")
    print(history_summary.to_string(index=False))


if __name__ == "__main__":
    main()

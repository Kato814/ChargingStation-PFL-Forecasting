from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from baselines.reference import BASELINE_COLUMN_MAP, load_locked_comparison_predictions
from src.config import load_config
from src.evaluation import aggregate_system_metrics, paired_daily_inference, station_wape_table
from src.group_total_correction import GroupCorrectionConfig, group_total_correction
from src.negative_transfer_control import (
    NegativeTransferConfig,
    cumulative_loss_guard,
    fallback_and_redistribute,
    recent_performance_eligibility,
)

OUTPUT_DIR = ROOT / "results_revision_r1"
NEURAL_DIR = OUTPUT_DIR / "neural_baselines"

LOCKED_SOURCE = ROOT / "results_reference" / "oof_predictions_source.csv.gz"
FEDPROX_OOF = NEURAL_DIR / "fedprox_oof.csv.gz"
FEDREP_OOF = NEURAL_DIR / "fedrep_oof.csv.gz"

OLD_METHODS = {
    "FedAvg残差": BASELINE_COLUMN_MAP["FedAvg残差"],
    "全参数微调": BASELINE_COLUMN_MAP["全参数微调"],
    "本地MLP": BASELINE_COLUMN_MAP["本地MLP"],
    "pFedMe微调": BASELINE_COLUMN_MAP["pFedMe微调"],
}
NEW_METHOD_FILES = {
    "FedProx": FEDPROX_OOF,
    "FedRep": FEDREP_OOF,
}

KEYS = ["fold", "station_id", "type", "prediction_date", "slot"]
FULL_KEYS = KEYS + ["y_true", "previous_day_baseline"]
EXPECTED_RECORD_COUNT = 42240

SUMMARY_PATH = OUTPUT_DIR / "r1_postprocessing_fairness_extended_summary.csv"
INFERENCE_PATH = OUTPUT_DIR / "r1_postprocessing_fairness_extended_daily_inference.json"
METADATA_PATH = OUTPUT_DIR / "r1_postprocessing_fairness_extended_metadata.json"
OOF_PATH = OUTPUT_DIR / "r1_postprocessing_fairness_extended_oof.csv.gz"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return p.stdout.strip()
    except Exception:
        return None


def _fold_arrays(frame: pd.DataFrame, prediction_column: str):
    dates = sorted(pd.to_datetime(frame["prediction_date"]).dt.normalize().unique())
    stations = sorted(frame["station_id"].astype(int).unique())
    slots = sorted(frame["slot"].astype(int).unique())
    if len(stations) != 2 or slots != list(range(96)):
        raise AssertionError("expected one two-station deployment group with 96 slots")

    date_index = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    station_index = {sid: i for i, sid in enumerate(stations)}
    y = np.zeros((len(dates), len(stations), 96), dtype=np.float64)
    anchor = np.zeros_like(y)
    candidate = np.zeros_like(y)

    for row in frame.itertuples(index=False):
        di = date_index[pd.Timestamp(row.prediction_date).normalize()]
        si = station_index[int(row.station_id)]
        sl = int(row.slot)
        y[di, si, sl] = float(row.y_true)
        anchor[di, si, sl] = float(row.previous_day_baseline)
        candidate[di, si, sl] = float(getattr(row, prediction_column))
    return dates, stations, y, anchor, candidate


def _attach_stage_columns(
    frame: pd.DataFrame,
    method_slug: str,
    prediction_column: str,
    group_config: GroupCorrectionConfig,
    negative_config: NegativeTransferConfig,
) -> pd.DataFrame:
    out = frame.copy()
    group_col = f"r1_{method_slug}_group"
    final_col = f"r1_{method_slug}_group_risk"
    gamma_col = f"r1_{method_slug}_gamma"
    enabled_col = f"r1_{method_slug}_final_enabled"
    out[group_col] = np.nan
    out[final_col] = np.nan
    out[gamma_col] = np.nan
    out[enabled_col] = np.nan

    for fold, fold_frame in out.groupby("fold", sort=True):
        dates, stations, y, anchor, candidate = _fold_arrays(fold_frame, prediction_column)
        group_candidate, gamma = group_total_correction(y, anchor, candidate, group_config)
        recent = recent_performance_eligibility(y, anchor, group_candidate, negative_config)
        final_enabled = cumulative_loss_guard(y, anchor, group_candidate, recent, negative_config)
        final_prediction = fallback_and_redistribute(anchor, group_candidate, final_enabled)

        lookup = {
            (pd.Timestamp(date).normalize(), int(station), int(slot)): (
                float(group_candidate[di, si, slot]),
                float(final_prediction[di, si, slot]),
                float(gamma[di]),
                int(final_enabled[di, si]),
            )
            for di, date in enumerate(dates)
            for si, station in enumerate(stations)
            for slot in range(96)
        }
        mask = out["fold"].eq(fold)
        keys = list(
            zip(
                pd.to_datetime(out.loc[mask, "prediction_date"]).dt.normalize(),
                out.loc[mask, "station_id"].astype(int),
                out.loc[mask, "slot"].astype(int),
            )
        )
        values = [lookup[key] for key in keys]
        out.loc[mask, group_col] = [v[0] for v in values]
        out.loc[mask, final_col] = [v[1] for v in values]
        out.loc[mask, gamma_col] = [v[2] for v in values]
        out.loc[mask, enabled_col] = [v[3] for v in values]
    return out


def _stage_summary(frame: pd.DataFrame, method_name: str, column: str, stage: str) -> dict[str, object]:
    metrics = aggregate_system_metrics(frame, column)
    method_station = station_wape_table(frame, column).rename(columns={"WAPE": "method_WAPE"})
    base_station = station_wape_table(frame, "previous_day_baseline").rename(columns={"WAPE": "baseline_WAPE"})
    station = base_station.merge(method_station, on=["station_id", "type"], validate="one_to_one")
    station["relative_change"] = station["method_WAPE"] / station["baseline_WAPE"] - 1.0
    return {
        "method": method_name,
        "stage": stage,
        "WAPE": float(metrics["WAPE"]),
        "nRMSE": float(metrics["nRMSE"]),
        "pooled_WAPE": float(metrics["pooled_WAPE"]),
        "improved_stations": int((station["relative_change"] < 0).sum()),
        "max_station_relative_degradation": float(max(0.0, station["relative_change"].max())),
    }


def _normalize_locked() -> pd.DataFrame:
    frame = load_locked_comparison_predictions(LOCKED_SOURCE).copy()
    if len(frame) != EXPECTED_RECORD_COUNT:
        raise AssertionError(f"locked OOF record count {len(frame)} != {EXPECTED_RECORD_COUNT}")
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"]).dt.normalize()
    if frame.duplicated(KEYS).any():
        raise AssertionError("locked OOF contains duplicate keys")
    return frame


def _load_new_method(path: Path, method_name: str, locked: pd.DataFrame):
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, compression="infer", low_memory=False)
    required = set(FULL_KEYS + ["prediction"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{method_name} OOF missing columns: {sorted(missing)}")
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"], errors="raise").dt.normalize()
    frame["station_id"] = frame["station_id"].astype(int)
    frame["slot"] = frame["slot"].astype(int)
    if len(frame) != EXPECTED_RECORD_COUNT:
        raise AssertionError(f"{method_name} OOF record count {len(frame)} != {EXPECTED_RECORD_COUNT}")
    if frame.duplicated(KEYS).any():
        raise AssertionError(f"{method_name} OOF contains duplicate keys")

    reference = locked[KEYS + ["y_true", "previous_day_baseline", "proposed_prediction"]]
    merged = frame[FULL_KEYS + ["prediction"]].merge(
        reference,
        on=KEYS,
        how="inner",
        suffixes=("_new", "_locked"),
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_RECORD_COUNT:
        raise AssertionError(f"{method_name} key alignment failed: {len(merged)} rows")

    max_y = float(np.max(np.abs(merged["y_true_new"] - merged["y_true_locked"])))
    max_anchor = float(
        np.max(
            np.abs(
                merged["previous_day_baseline_new"]
                - merged["previous_day_baseline_locked"]
            )
        )
    )
    if max_y > 1e-9 or max_anchor > 1e-9:
        raise AssertionError(
            f"{method_name} locked alignment mismatch: y={max_y}, anchor={max_anchor}"
        )

    out = merged[
        KEYS
        + [
            "y_true_locked",
            "previous_day_baseline_locked",
            "proposed_prediction",
            "prediction",
        ]
    ].rename(
        columns={
            "y_true_locked": "y_true",
            "previous_day_baseline_locked": "previous_day_baseline",
        }
    )
    return out, {"max_y_true_difference": max_y, "max_baseline_difference": max_anchor}


def _old_consistency(summary: pd.DataFrame, inference: dict) -> dict:
    existing_summary_path = OUTPUT_DIR / "r1_postprocessing_fairness_summary.csv"
    existing_inference_path = OUTPUT_DIR / "r1_postprocessing_fairness_daily_inference.json"
    result = {
        "existing_summary_present": existing_summary_path.exists(),
        "existing_inference_present": existing_inference_path.exists(),
        "max_numeric_difference_summary": None,
        "max_numeric_difference_inference": None,
        "status": "NOT_CHECKED",
    }
    if not (existing_summary_path.exists() and existing_inference_path.exists()):
        return result

    old = pd.read_csv(existing_summary_path)
    cur = summary[summary["method"].isin(OLD_METHODS)].copy()
    merge_keys = ["method", "stage"]
    metrics = ["WAPE", "nRMSE", "pooled_WAPE", "improved_stations", "max_station_relative_degradation"]
    merged = old[merge_keys + metrics].merge(
        cur[merge_keys + metrics],
        on=merge_keys,
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    diffs = []
    for metric in metrics:
        diffs.extend(
            np.abs(
                pd.to_numeric(merged[f"{metric}_old"])
                - pd.to_numeric(merged[f"{metric}_new"])
            ).tolist()
        )
    max_summary = float(max(diffs, default=0.0))

    old_inf = json.loads(existing_inference_path.read_text(encoding="utf-8"))
    inf_diffs = []
    for method in OLD_METHODS:
        if method not in old_inf or method not in inference:
            raise AssertionError(f"missing old-method inference for {method}")
        for key, old_value in old_inf[method].items():
            new_value = inference[method][key]
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                inf_diffs.append(abs(float(old_value) - float(new_value)))
    max_inf = float(max(inf_diffs, default=0.0))
    result["max_numeric_difference_summary"] = max_summary
    result["max_numeric_difference_inference"] = max_inf
    result["status"] = "PASS" if max_summary <= 1e-12 and max_inf <= 1e-12 else "REVIEW"
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "paper_config.json")
    group_config = GroupCorrectionConfig(**config["group_total_correction"])
    negative_config = NegativeTransferConfig(**config["negative_transfer"])

    locked = _normalize_locked()

    summaries: list[dict[str, object]] = []
    inference: dict[str, dict[str, float | int]] = {}
    oof_parts = []
    alignment = {}

    enriched = locked.copy()
    for idx, (method_name, prediction_column) in enumerate(OLD_METHODS.items(), start=1):
        slug = f"old_m{idx}"
        enriched = _attach_stage_columns(
            enriched, slug, prediction_column, group_config, negative_config
        )
        group_col = f"r1_{slug}_group"
        final_col = f"r1_{slug}_group_risk"
        summaries.append(_stage_summary(enriched, method_name, prediction_column, "raw"))
        summaries.append(_stage_summary(enriched, method_name, group_col, "+group"))
        summaries.append(_stage_summary(enriched, method_name, final_col, "+group+risk"))
        inference[method_name] = paired_daily_inference(
            enriched,
            proposed_column="proposed_prediction",
            baseline_column=final_col,
        )

    old_stage_columns = [c for c in enriched.columns if c.startswith("r1_old_")]
    oof_parts.append(
        enriched[
            FULL_KEYS + ["proposed_prediction"] + list(OLD_METHODS.values()) + old_stage_columns
        ].assign(source_group="locked_original_baselines")
    )

    for idx, (method_name, path) in enumerate(NEW_METHOD_FILES.items(), start=1):
        frame, align = _load_new_method(path, method_name, locked)
        alignment[method_name] = align
        slug = f"new_m{idx}"
        enriched_new = _attach_stage_columns(
            frame, slug, "prediction", group_config, negative_config
        )
        group_col = f"r1_{slug}_group"
        final_col = f"r1_{slug}_group_risk"
        summaries.append(_stage_summary(enriched_new, method_name, "prediction", "raw"))
        summaries.append(_stage_summary(enriched_new, method_name, group_col, "+group"))
        summaries.append(_stage_summary(enriched_new, method_name, final_col, "+group+risk"))
        inference[method_name] = paired_daily_inference(
            enriched_new,
            proposed_column="proposed_prediction",
            baseline_column=final_col,
        )
        stage_cols = [c for c in enriched_new.columns if c.startswith(f"r1_{slug}_")]
        oof_parts.append(
            enriched_new[
                FULL_KEYS + ["proposed_prediction", "prediction"] + stage_cols
            ].assign(source_group=method_name)
        )

    summary = pd.DataFrame(summaries)
    consistency = _old_consistency(summary, inference)

    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    INFERENCE_PATH.write_text(
        json.dumps(inference, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.concat(oof_parts, ignore_index=True, sort=False).to_csv(
        OOF_PATH, index=False, compression="gzip"
    )

    metadata = {
        "experiment": "R1 fixed-postprocessing fairness extension",
        "git_commit": _git_head(),
        "locked_source_sha256": _sha256(LOCKED_SOURCE),
        "fedprox_oof_sha256": _sha256(FEDPROX_OOF),
        "fedrep_oof_sha256": _sha256(FEDREP_OOF),
        "record_count_per_method": EXPECTED_RECORD_COUNT,
        "methods": list(OLD_METHODS) + list(NEW_METHOD_FILES),
        "stages": ["raw", "+group", "+group+risk"],
        "group_total_correction": config["group_total_correction"],
        "negative_transfer": config["negative_transfer"],
        "new_method_alignment": alignment,
        "old_E2_consistency": consistency,
        "scope_note": (
            "Fixed-postprocessing control: the same frozen group-correction and "
            "negative-transfer-control parameters are applied to every baseline. "
            "No baseline-specific retuning is performed. This is not a claim that "
            "each baseline receives its own best possible calibrated postprocessor."
        ),
        "outputs": {
            "summary": str(SUMMARY_PATH),
            "daily_inference": str(INFERENCE_PATH),
            "local_large_oof": str(OOF_PATH),
        },
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(json.dumps(inference, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    if consistency["status"] != "PASS":
        raise SystemExit(
            f"Old E2 consistency check failed or unavailable: {consistency}"
        )


if __name__ == "__main__":
    main()

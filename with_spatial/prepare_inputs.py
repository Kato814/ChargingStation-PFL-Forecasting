from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.data_preparation import (
    build_federated_statistics,
    feature_schema_frame,
    load_model_input_cache,
    save_federated_statistics,
)
from src.feature_builder_with_spatial import build_model_input_cache


def _compare_clients(rebuilt: dict[int, dict], reference: dict[int, dict]) -> dict:
    max_abs = {name: 0.0 for name in ("x", "y_residual", "baseline", "y_true")}
    dates_equal = True
    station_types_equal = True
    if set(rebuilt) != set(reference):
        raise AssertionError("rebuilt/reference station sets differ")
    for station_id in sorted(reference):
        a = rebuilt[station_id]
        b = reference[station_id]
        dates_equal = dates_equal and list(a["forecast_date"]) == list(b["forecast_date"])
        station_types_equal = station_types_equal and str(a["station_type"]) == str(b["station_type"])
        for name in max_abs:
            av = np.asarray(a[name], dtype=np.float64)
            bv = np.asarray(b[name], dtype=np.float64)
            if av.shape != bv.shape:
                raise AssertionError(f"station {station_id} {name} shape differs: {av.shape} vs {bv.shape}")
            max_abs[name] = max(max_abs[name], float(np.max(np.abs(av - bv))))
    return {"station_count": len(reference), "dates_equal": bool(dates_equal), "station_types_equal": bool(station_types_equal), "max_abs_diff": max_abs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate the 539-D paper inputs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-statistics", type=Path, default=None)
    parser.add_argument("--rebuild-cache-from-raw", action="store_true", help="rebuild the 539-D cache from data/source_station_dataset and verify exact equivalence")
    parser.add_argument("--output-cache", type=Path, default=None, help="optional path for the raw-data rebuilt cache; the canonical cache is never overwritten by default")
    args = parser.parse_args()

    root = args.root.resolve()
    canonical_cache = root / "data/clients_with_spatial.pkl"
    output_statistics = args.output_statistics or (root / "data/federated_statistics_rebuilt.pkl")
    clients, feature_names = load_model_input_cache(canonical_cache)

    if args.rebuild_cache_from_raw:
        rebuilt, rebuilt_feature_names, audit = build_model_input_cache(root / "data/source_station_dataset")
        if tuple(rebuilt_feature_names) != tuple(feature_names):
            raise AssertionError("raw-data rebuilt feature names differ from canonical 539-D schema")
        equivalence = _compare_clients(rebuilt, clients)
        if any(value != 0.0 for value in equivalence["max_abs_diff"].values()):
            raise AssertionError(f"raw-data rebuilt cache is not exactly equivalent: {equivalence}")
        output_cache = args.output_cache or (root / "data/clients_with_spatial_rebuilt.pkl")
        output_cache.parent.mkdir(parents=True, exist_ok=True)
        with output_cache.open("wb") as handle:
            pickle.dump({"clients": rebuilt, "feature_names": tuple(rebuilt_feature_names)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        audit.to_csv(root / "data/raw_to_cache_audit.csv", index=False, encoding="utf-8-sig")
        (root / "data/raw_to_cache_equivalence.json").write_text(json.dumps(equivalence, ensure_ascii=False, indent=2), encoding="utf-8")

    statistics = build_federated_statistics(clients)
    save_federated_statistics(statistics, output_statistics)
    schema = feature_schema_frame(feature_names)
    schema.to_csv(root / "data/feature_schema_539.csv", index=False, encoding="utf-8-sig")
    summary = {
        "station_count": len(clients), "feature_dimension": len(feature_names), "output_dimension": 96,
        "training_sample_count": int(sum(int(v["n"]) for v in statistics.values())),
        "spatial_feature_count": int(schema["feature_group"].isin(["spatial_poi", "spatial_static"]).sum()),
        "statistics_output": str(Path(output_statistics).resolve()), "raw_cache_rebuild_requested": bool(args.rebuild_cache_from_raw),
    }
    (root / "data/input_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

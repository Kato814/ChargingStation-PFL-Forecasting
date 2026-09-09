from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from src.data_preparation import (
    build_federated_statistics,
    feature_schema_frame,
    save_federated_statistics,
)
from src.feature_builder import build_model_input_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 527-D non-spatial paper inputs from source station data")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-cache", type=Path, default=None)
    parser.add_argument("--output-statistics", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output_cache = args.output_cache or (root / "data/clients_no_spatial.pkl")
    output_statistics = args.output_statistics or (root / "data/federated_statistics_no_spatial.pkl")

    clients, feature_names, audit = build_model_input_cache(root / "data/source_station_dataset")
    if len(feature_names) != 527:
        raise AssertionError(f"expected 527 features, found {len(feature_names)}")

    output_cache.parent.mkdir(parents=True, exist_ok=True)
    with output_cache.open("wb") as handle:
        pickle.dump({"clients": clients, "feature_names": tuple(feature_names)}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    statistics = build_federated_statistics(clients)
    save_federated_statistics(statistics, output_statistics)

    audit.to_csv(root / "data/raw_to_cache_audit.csv", index=False, encoding="utf-8-sig")
    schema = feature_schema_frame(tuple(feature_names))
    schema.to_csv(root / "data/feature_schema_527.csv", index=False, encoding="utf-8-sig")

    summary = {
        "station_count": len(clients),
        "feature_dimension": len(feature_names),
        "output_dimension": 96,
        "training_sample_count": int(sum(int(v["n"]) for v in statistics.values())),
        "input_scope": "historical_load_calendar_station_configuration_only",
        "cache_output": str(Path(output_cache).resolve().relative_to(root)),
        "statistics_output": str(Path(output_statistics).resolve().relative_to(root)),
    }
    (root / "data/input_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

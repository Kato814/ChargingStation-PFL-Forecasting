from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the exact ordered 539-D feature schema from the frozen client cache.")
    parser.add_argument("--cache", type=Path, default=Path("data/clients_with_spatial.pkl"))
    parser.add_argument("--output", type=Path, default=Path("data/feature_schema_539.csv"))
    args = parser.parse_args()

    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)

    feature_names = payload.get("feature_names")
    if feature_names is None:
        raise KeyError("feature_names not found in frozen client cache")
    if len(feature_names) != 539:
        raise ValueError(f"expected 539 feature names, got {len(feature_names)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_index", "feature_name"])
        for index, name in enumerate(feature_names):
            writer.writerow([index, name])

    print(f"FEATURE_SCHEMA_EXPORTED={args.output}")
    print(f"FEATURE_DIMENSION={len(feature_names)}")


if __name__ == "__main__":
    main()

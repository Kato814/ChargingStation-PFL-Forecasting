from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    reproduced = pd.read_csv(ROOT / "results_reproduced/oof_predictions.csv.gz", parse_dates=["prediction_date"])
    reference = pd.read_csv(ROOT / "results_reference/oof_predictions_source.csv.gz", parse_dates=["prediction_date"])

    keys = ["fold", "station_id", "type", "prediction_date", "slot"]
    columns = [
        "y_true",
        "previous_day_baseline",
        "global_candidate",
        "type_candidate",
        "station_candidate",
        "group_candidate",
        "proposed_prediction",
        "local_coefficient",
        "group_coefficient",
        "final_enabled",
    ]
    left = reproduced[keys + columns].sort_values(keys).reset_index(drop=True)
    right = reference[keys + columns].sort_values(keys).reset_index(drop=True)
    if len(left) != len(right):
        raise AssertionError(f"row count mismatch: {len(left)} != {len(right)}")
    if not left[keys].equals(right[keys]):
        raise AssertionError("prediction keys differ")

    max_diffs = {}
    for col in columns:
        if col == "final_enabled":
            if not np.array_equal(left[col].to_numpy(), right[col].to_numpy()):
                raise AssertionError(f"{col} differs")
            max_diffs[col] = 0.0
        else:
            diff = np.max(np.abs(left[col].to_numpy(dtype=float) - right[col].to_numpy(dtype=float)))
            max_diffs[col] = float(diff)
            if diff > 1e-10:
                raise AssertionError(f"{col} max abs diff {diff} > 1e-10")
    print("PASS")
    for key, value in max_diffs.items():
        print(f"{key}: {value:.3e}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_preparation import build_federated_statistics, load_model_input_cache
from src.pipeline import load_federated_statistics

# Cross-BLAS/CPU reconstruction may differ at machine precision because reduction
# order can change. Structural fields remain exact; floating sufficient statistics
# use the explicit archive-maintenance tolerance documented in
# docs/ENVIRONMENT_AND_NUMERICS.md.
ATOL = 1e-9


def main() -> None:
    clients, feature_names = load_model_input_cache(ROOT / "data/clients_with_spatial.pkl")
    rebuilt = build_federated_statistics(clients)
    locked = load_federated_statistics(ROOT / "data/federated_statistics_with_spatial.pkl")
    if len(feature_names) != 539 or set(rebuilt) != set(locked):
        raise AssertionError("input/statistics structure mismatch")
    for station_id in sorted(rebuilt):
        for key in ["n", "feature_width", "output_width", "station_type", "train_start", "train_end"]:
            if rebuilt[station_id][key] != locked[station_id][key]:
                raise AssertionError(f"station {station_id} field {key} differs")
        for key in ["x_sum", "xtx", "y_sum", "xty"]:
            diff = float(np.max(np.abs(np.asarray(rebuilt[station_id][key]) - np.asarray(locked[station_id][key]))))
            if diff > ATOL:
                raise AssertionError(f"station {station_id} {key} differs: {diff} > {ATOL}")
    print(f"PASS: sufficient statistics equivalent within atol={ATOL:g}, rtol=0")


if __name__ == "__main__":
    main()

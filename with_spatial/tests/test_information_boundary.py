from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import build_folds, load_config
from src.group_total_correction import GroupCorrectionConfig
from src.negative_transfer_control import NegativeTransferConfig
from src.pipeline import load_clean_clients, load_federated_statistics, run_fold


def main() -> None:
    config = load_config(ROOT / "paper_config.json")
    clients, _ = load_clean_clients(ROOT / "data/clients_with_spatial.pkl")
    statistics = load_federated_statistics(ROOT / "data/federated_statistics_with_spatial.pkl")
    fold = build_folds(config)[0]
    gc = GroupCorrectionConfig(**config["group_total_correction"])
    nt = NegativeTransferConfig(**config["negative_transfer"])

    original = run_fold(clients, fold, statistics, gc, nt)
    perturbed_clients = deepcopy(clients)
    target_date_index = 20
    target_date = original["dates"][target_date_index]
    for station_id in fold.test:
        client = perturbed_clients[station_id]
        dates = np.asarray(client["forecast_date"], dtype="datetime64[D]")
        mask = dates >= np.datetime64(target_date.date())
        client["y_true"] = np.asarray(client["y_true"], dtype=np.float64).copy()
        client["y_residual"] = np.asarray(client["y_residual"], dtype=np.float64).copy()
        client["y_true"][mask] += 1.0e6
        client["y_residual"][mask] += 1.0e6

    perturbed = run_fold(perturbed_clients, fold, statistics, gc, nt)
    end = target_date_index + 1
    for key in ["station_candidate", "local_coefficient", "group_candidate", "group_coefficient", "recent_eligibility", "final_eligibility", "final_prediction"]:
        left = np.asarray(original[key])[:end]
        right = np.asarray(perturbed[key])[:end]
        if not np.array_equal(left, right) and not np.allclose(left, right, atol=1e-10, rtol=0.0):
            raise AssertionError(f"causal boundary violated for {key}")
    print("PASS")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import pickle
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import feature_builder_with_spatial as fb


def main() -> None:
    data_root = ROOT / "data/source_station_dataset"
    static = fb._load_static_tables(data_root)
    daily = fb._load_daily_tables(data_root)
    station_types, raw_types = fb._train_only_categories(static)
    names = fb.feature_names(station_types, raw_types)
    filtered_names, keep_mask = fb._filtered_feature_names(names)
    path = data_root / "test/station_processed_5min/1009.csv"
    client, _ = fb.build_station_client("test", path, static["test"], daily["test"], station_types, raw_types)
    assert client is not None
    with (ROOT / "data/clients_with_spatial.pkl").open("rb") as handle:
        reference = pickle.load(handle)
    ref = reference["clients"][1009]
    assert tuple(filtered_names) == tuple(reference["feature_names"])
    assert client.meta["forecast_date"].astype(str).tolist() == ref["forecast_date"]
    assert np.array_equal(np.asarray(client.x)[:, keep_mask].astype(np.float64), np.asarray(ref["x"], dtype=np.float64))
    assert np.array_equal(np.asarray(client.y_residual, dtype=np.float64), np.asarray(ref["y_residual"], dtype=np.float64))
    assert np.array_equal(np.asarray(client.baseline, dtype=np.float64), np.asarray(ref["baseline"], dtype=np.float64))
    assert np.array_equal(np.asarray(client.y_true, dtype=np.float64), np.asarray(ref["y_true"], dtype=np.float64))
    print("PASS: raw source -> 539-D cache sample equivalence")


if __name__ == "__main__":
    main()

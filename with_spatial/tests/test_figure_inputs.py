from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pred = pd.read_csv(ROOT / "results_reproduced/oof_predictions.csv.gz", parse_dates=["prediction_date"])
    required = {
        "station_id", "prediction_date", "slot", "y_true", "previous_day_baseline",
        "proposed_prediction", "local_coefficient",
    }
    missing = required.difference(pred.columns)
    assert not missing, sorted(missing)
    assert len(pred) == 42240
    daily_alpha = pred.groupby(["station_id", "prediction_date"])["local_coefficient"].nunique()
    assert int(daily_alpha.max()) == 1
    representative = pred[
        pred["station_id"].isin([1011, 1566, 1599])
        & pred["prediction_date"].between("2023-02-15", "2023-02-17")
    ]
    assert len(representative) == 3 * 3 * 96, len(representative)
    print("PASS: figure input contracts")


if __name__ == "__main__":
    main()

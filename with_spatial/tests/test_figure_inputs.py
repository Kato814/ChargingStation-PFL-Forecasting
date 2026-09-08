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

    # Fig. 6: one rolling scalar coefficient per station-day.
    daily_alpha = pred.groupby(["station_id", "prediction_date"])["local_coefficient"].nunique()
    assert int(daily_alpha.max()) == 1
    assert len(daily_alpha) == 10 * 44

    # Fig. 7: frozen risk-parameter sensitivity evidence.
    grid = pd.read_csv(ROOT / "results_reference/risk_sensitivity_grid.csv")
    delta = pd.read_csv(ROOT / "results_reference/delta_sensitivity_beta_003.csv")
    beta = pd.read_csv(ROOT / "results_reference/beta_sensitivity_delta_m002.csv")
    assert len(grid) == 25
    assert len(delta) == 5
    assert len(beta) == 5
    selected = grid[(grid["delta"] == -0.02) & (grid["beta"] == 0.03)]
    assert len(selected) == 1
    row = selected.iloc[0]
    assert abs(float(row["WAPE"]) - 0.093571) < 1e-6
    assert abs(float(row["max_relative_degradation"]) - 0.015618) < 1e-6

    # Fig. 8: three station-type aggregates.
    type_result = pd.read_csv(ROOT / "results_reproduced/type_aggregate_results.csv")
    assert set(type_result["type"]) == {"pure_fast", "pure_slow", "mixed_ac_dc"}

    # Fig. 9: complete 3-day / 3-station / 96-point window.
    representative = pred[
        pred["station_id"].isin([1011, 1566, 1599])
        & pred["prediction_date"].between("2023-02-15", "2023-02-17")
    ]
    assert len(representative) == 3 * 3 * 96, len(representative)
    print("PASS: figure input contracts")


if __name__ == "__main__":
    main()

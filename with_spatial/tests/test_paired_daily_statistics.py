from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import paired_daily_inference


def main() -> None:
    frame = pd.read_csv(
        ROOT / "results_reference/oof_predictions_source.csv.gz",
        parse_dates=["prediction_date"],
    )
    result = paired_daily_inference(frame)
    if result["n_prediction_days"] != 44:
        raise AssertionError(result)
    if abs(result["mean_daily_WAPE_difference"] - (-0.011563263932537492)) >= 1e-12:
        raise AssertionError(result)
    if abs(result["circular_mbb_95_low"] - (-0.0164907)) >= 2e-7:
        raise AssertionError(result)
    if abs(result["circular_mbb_95_high"] - (-0.0062974)) >= 2e-7:
        raise AssertionError(result)
    print("PASS")
    print(result)


if __name__ == "__main__":
    main()

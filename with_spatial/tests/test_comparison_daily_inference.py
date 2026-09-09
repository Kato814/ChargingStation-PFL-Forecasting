from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.reference import load_locked_comparison_predictions
from src.evaluation import paired_daily_inference

EXPECTED = {
    "previous_day_baseline": (-0.011563263932537492, -0.016490703504466735, -0.006297401202770892),
    "fedavg_direct_kw": (-0.012989780204685115, -0.017935458534253663, -0.007707265898986418),
    "fedavg_ft30_kw": (-0.013933482980145467, -0.02853209192903344, -0.002021486389035354),
    "local_only30_kw": (-0.01136173249825394, -0.02279673800458469, -0.0024908043344778097),
    "pfedme_ft30_kw": (-0.009711810890273499, -0.01923171161394556, -0.0014630366945719137),
}


def main() -> None:
    frame = load_locked_comparison_predictions(ROOT / "results_reference" / "oof_predictions_source.csv.gz")
    for column, expected in EXPECTED.items():
        result = paired_daily_inference(frame, proposed_column="proposed_prediction", baseline_column=column)
        actual = (
            result["mean_daily_WAPE_difference"],
            result["circular_mbb_95_low"],
            result["circular_mbb_95_high"],
        )
        assert np.allclose(actual, expected, atol=1e-12, rtol=0.0), (column, actual)
    print("PASS: paired daily deployment-group comparisons reproduced")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.reference import load_locked_comparison_predictions, reproduce_table2_from_locked_predictions

EXPECTED = {
    "前日负荷基准": (0.10042573373583695, 0.06975904123138603),
    "FedAvg残差": (0.10161459174758193, 0.07033751175461174),
    "全参数微调": (0.10873053151338084, 0.0746735993842494),
    "本地MLP": (0.10376417109992134, 0.07131901258620356),
    "本地岭回归": (0.1025, 0.0727),
    "pFedMe微调": (0.10420431053961007, 0.07152138777507201),
    "本文方法": (0.09357050219189945, 0.06608295125576832),
}


def main() -> None:
    source = ROOT / "results_reference" / "oof_predictions_source.csv.gz"
    frame = load_locked_comparison_predictions(source)
    result = reproduce_table2_from_locked_predictions(frame).set_index("method")
    assert list(result.index) == list(EXPECTED)
    for method, (expected_wape, expected_nrmse) in EXPECTED.items():
        actual_wape = float(result.loc[method, "WAPE"])
        actual_nrmse = float(result.loc[method, "nRMSE"])
        if method == "本地岭回归":
            assert actual_wape == expected_wape and actual_nrmse == expected_nrmse
        else:
            assert np.isclose(actual_wape, expected_wape, atol=1e-12, rtol=0.0), (method, actual_wape)
            assert np.isclose(actual_nrmse, expected_nrmse, atol=1e-12, rtol=0.0), (method, actual_nrmse)
    assert bool(result.loc["本地岭回归", "raw_prediction_available"]) is False
    print("PASS: paper Table-2 comparison results reproduced from locked evidence")


if __name__ == "__main__":
    main()

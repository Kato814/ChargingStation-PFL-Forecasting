from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.reference import load_locked_comparison_predictions
from src.evaluation import paired_daily_inference


COMPARISONS = {
    "前日负荷基准": "previous_day_baseline",
    "FedAvg残差": "fedavg_direct_kw",
    "全参数微调": "fedavg_ft30_kw",
    "本地MLP": "local_only30_kw",
    "pFedMe微调": "pfedme_ft30_kw",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper deployment-group paired daily inference versus comparison methods.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results_reference" / "oof_predictions_source.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_reproduced" / "baselines" / "paired_daily_comparisons.json",
    )
    args = parser.parse_args()
    frame = load_locked_comparison_predictions(args.input)
    result = {
        method: paired_daily_inference(
            frame,
            proposed_column="proposed_prediction",
            baseline_column=column,
        )
        for method, column in COMPARISONS.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

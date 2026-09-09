from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.reference import load_locked_comparison_predictions, reproduce_table2_from_locked_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce paper Table 2 from locked OOF comparison predictions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results_reference" / "oof_predictions_source.csv.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_reproduced" / "baselines" / "table2_comparison_results.csv",
    )
    args = parser.parse_args()
    frame = load_locked_comparison_predictions(args.input)
    table = reproduce_table2_from_locked_predictions(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

OUTPUT_DIR = ROOT / "results_revision_r1"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = pd.read_csv(ROOT / "data" / "feature_schema_539.csv")

    audit_rows = [
        {
            "item": "previous_day_profile",
            "dimension_or_state": 96,
            "availability": "prediction day d-1 completed load",
            "uses_day_d_true_load": False,
            "basis": "feature_builder: previous_date; only dates < forecast_date enter history",
        },
        {
            "item": "historical_profiles_and_summaries",
            "dimension_or_state": 319,
            "availability": "history strictly before prediction day d",
            "uses_day_d_true_load": False,
            "basis": "mean7/same_dow4/daytype14/similar3 are selected from history_dates < forecast_date",
        },
        {
            "item": "day_ahead_calendar",
            "dimension_or_state": 5,
            "availability": "calendar/holiday information known before day d",
            "uses_day_d_true_load": False,
            "basis": "dayofweek/weekend/holiday/public-holiday/Spring-Festival flags",
        },
        {
            "item": "station_non_spatial_static",
            "dimension_or_state": 5,
            "availability": "static station attributes known before deployment",
            "uses_day_d_true_load": False,
            "basis": "equipment counts, DC ratio and rated power",
        },
        {
            "item": "spatial_inputs",
            "dimension_or_state": 12,
            "availability": "static POI/coordinate/zone attributes known before deployment",
            "uses_day_d_true_load": False,
            "basis": "8 POI + longitude + latitude + zone charge count + zone area",
        },
        {
            "item": "station_type_indicators",
            "dimension_or_state": 6,
            "availability": "static station type known before deployment",
            "uses_day_d_true_load": False,
            "basis": "model/raw station type one-hot indicators",
        },
        {
            "item": "alpha_jd",
            "dimension_or_state": "1 scalar per station-day",
            "availability": "estimated only from dates before d",
            "uses_day_d_true_load": False,
            "basis": "rolling_scalar_adaptation uses [start:index), excluding current day",
        },
        {
            "item": "gamma_d",
            "dimension_or_state": "1 scalar per deployment-group day",
            "availability": "estimated only from previous group-history days",
            "uses_day_d_true_load": False,
            "basis": "group_total_correction uses [start:day_index), excluding current day",
        },
        {
            "item": "risk_eligibility",
            "dimension_or_state": "1 binary state per station-day",
            "availability": "recent/cumulative losses accumulated before current prediction",
            "uses_day_d_true_load": False,
            "basis": "recent screen excludes current day; cumulative guard updates losses after decision",
        },
    ]
    audit = pd.DataFrame(audit_rows)

    if len(schema) != 539:
        raise AssertionError(f"expected 539 schema rows, got {len(schema)}")
    group_counts = schema.groupby("feature_group").size().to_dict()
    expected = {
        "previous_day_profile": 96,
        "historical_summary": 319,
        "similar_day_profile": 96,
        "station_static": 5,
        "spatial_poi": 8,
        "spatial_static": 4,
        "calendar": 5,
        "station_type": 6,
    }
    if group_counts != expected:
        raise AssertionError(f"feature-group counts differ: {group_counts}")

    test = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "test_information_boundary.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError(test.stdout + "\n" + test.stderr)

    audit.to_csv(
        OUTPUT_DIR / "r1_information_boundary_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "r1_information_boundary_check.json").write_text(
        json.dumps(
            {
                "feature_dimension": 539,
                "feature_group_counts": group_counts,
                "causal_perturbation_test": test.stdout.strip(),
                "status": "PASS",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(audit.to_string(index=False))
    print("\nCAUSAL TEST:", test.stdout.strip())


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ARCHIVE_SHA = "8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e"
EXPECTED_SOURCE_SHA = "ec5618f9a979049c57659c51716929514a6897d829652ba626e8b10a1e1be8ed"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "VERSION.txt",
        ROOT / "paper_config.json",
        ROOT / "docs/ARCHIVE_RECORD.md",
        ROOT / "docs/CODE_TO_PAPER_MAPPING.md",
        ROOT / "docs/FEATURE_SCHEMA_SUMMARY.md",
        ROOT / "docs/FOLD_PROTOCOL.md",
        ROOT / "docs/METRIC_DEFINITIONS.md",
        ROOT / "docs/RESULT_PROVENANCE.md",
        ROOT / "docs/RESULTS_BASELINE.md",
        ROOT / "docs/FIGURE_REPRODUCTION.md",
        ROOT / "docs/ENVIRONMENT_AND_NUMERICS.md",
        ROOT / "docs/GITHUB_MAINTENANCE_GUIDE.md",
        ROOT / "archive/VERIFICATION.json",
        ROOT / "archive/PACKAGE_SUMMARY.json",
        ROOT / "archive/FROZEN_ARCHIVE_SHA256.txt",
        ROOT / "data/feature_group_ranges_539.csv",
        ROOT / "data/source_station_dataset/station_split.csv",
        ROOT / "data/source_station_dataset/dataset_manifest.csv",
        ROOT / "results_reference/risk_sensitivity_grid.csv",
        ROOT / "results_reference/delta_sensitivity_beta_003.csv",
        ROOT / "results_reference/beta_sensitivity_delta_m002.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"missing required repository records: {missing}")

    version = (ROOT / "VERSION.txt").read_text(encoding="utf-8")
    assert "WITH_SPATIAL_539D" in version
    assert "FROZEN_PAPER_RECORD" in version

    archive_record = (ROOT / "archive/FROZEN_ARCHIVE_SHA256.txt").read_text(encoding="utf-8")
    assert EXPECTED_ARCHIVE_SHA in archive_record
    assert EXPECTED_SOURCE_SHA in archive_record

    verification = json.loads((ROOT / "archive/VERIFICATION.json").read_text(encoding="utf-8"))
    assert verification["status"] == "PASS"
    assert int(verification["feature_dimension"]) == 539
    assert int(verification["record_count"]) == 42240

    config = json.loads((ROOT / "paper_config.json").read_text(encoding="utf-8"))
    assert len(config["target_pairs"]) == 5
    assert sorted(len(pair) for pair in config["target_pairs"]) == [2, 2, 2, 2, 2]
    assert config["negative_transfer"]["relative_gain_margin"] == -0.02
    assert config["negative_transfer"]["regret_budget"] == 0.03

    groups = read_csv(ROOT / "data/feature_group_ranges_539.csv")
    feature_count = sum(int(row["count"]) for row in groups)
    assert feature_count == 539, feature_count

    station_split = read_csv(ROOT / "data/source_station_dataset/station_split.csv")
    assert len(station_split) == 40
    assert sum(row["split"] == "train" for row in station_split) == 30
    assert sum(row["split"] == "validation" for row in station_split) == 5
    assert sum(row["split"] == "test" for row in station_split) == 5

    manifest = read_csv(ROOT / "data/source_station_dataset/dataset_manifest.csv")
    assert len(manifest) == 56, len(manifest)

    risk_grid = read_csv(ROOT / "results_reference/risk_sensitivity_grid.csv")
    assert len(risk_grid) == 25
    selected = [row for row in risk_grid if float(row["delta"]) == -0.02 and float(row["beta"]) == 0.03]
    assert len(selected) == 1
    assert abs(float(selected[0]["WAPE"]) - 0.093571) < 1e-6
    assert abs(float(selected[0]["max_relative_degradation"]) - 0.015618) < 1e-6

    print("PASS: authoritative 539-D repository metadata")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def compile_python() -> int:
    files = [p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    for path in files:
        py_compile.compile(str(path), doraise=True)
    return len(files)


def main() -> None:
    py = sys.executable
    compiled = compile_python()
    with tempfile.TemporaryDirectory(prefix="pfl_spatial_verify_") as tmp:
        rebuilt = Path(tmp) / "federated_statistics.pkl"
        run([py, "prepare_inputs.py", "--output-statistics", str(rebuilt)])
    run([py, "run_paper_pipeline.py"])
    run([py, "tests/test_input_statistics_equivalence.py"])
    run([py, "tests/test_regression_equivalence.py"])
    run([py, "tests/test_paired_daily_statistics.py"])
    run([py, "tests/test_information_boundary.py"])
    run([py, "tests/test_ablation_reproduction.py"])
    run([py, "experiments/reproduce_table2_baselines.py"])
    run([py, "experiments/reproduce_comparison_daily_inference.py"])
    run([py, "tests/test_baseline_protocol_lock.py"])
    run([py, "tests/test_table2_baselines.py"])
    run([py, "tests/test_comparison_daily_inference.py"])
    run([py, "tests/test_removed_unwanted_daily_metric.py"])
    run([py, "tests/test_source_dataset_manifest.py"])
    run([py, "tests/test_raw_feature_builder_sample.py"])
    run([py, "tests/test_figure_inputs.py"])
    run([py, "figures/plot_fig6_scalar_coefficients.py"])
    run([py, "figures/plot_fig8_type_improvement.py"])
    run([py, "figures/plot_fig9_typical_curves.py"])

    summary = json.loads((ROOT / "results_reproduced/summary.json").read_text(encoding="utf-8"))
    verification = {
        "status": "PASS",
        "variant": "WITH_SPATIAL_539D",
        "compiled_python_files": compiled,
        "feature_dimension": summary["feature_dimension"],
        "record_count": summary["record_count"],
        "system_WAPE": summary["proposed"]["WAPE"],
        "nRMSE": summary["proposed"]["nRMSE"],
        "pooled_WAPE": summary["proposed"]["pooled_WAPE"],
        "improved_stations": summary["improved_stations"],
        "max_station_relative_degradation": summary["max_station_relative_degradation"],
        "paired_daily_mean_difference": summary["paired_daily_statistics"]["mean_daily_WAPE_difference"],
        "statistics_equivalence_atol": 1e-10,
        "statistics_equivalence_rtol": 0.0,
    }
    (ROOT / "VERIFICATION.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

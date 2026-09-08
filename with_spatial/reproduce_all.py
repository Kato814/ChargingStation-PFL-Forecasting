from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(args: list[str]) -> None:
    print('+', ' '.join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce the paper main method, tables, inference summaries and paper figures.')
    parser.add_argument('--skip-figures', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Run the full package verification after reproduction.')
    args = parser.parse_args()

    py = sys.executable
    run([py, 'run_paper_pipeline.py'])
    run([py, 'experiments/reproduce_table2_baselines.py'])
    run([py, 'experiments/reproduce_comparison_daily_inference.py'])
    run([py, 'experiments/reproduce_table3_shared_prior_ablation.py'])
    run([py, 'experiments/reproduce_table4_ablation.py'])

    if not args.skip_figures:
        run([py, 'figures/plot_fig6_scalar_coefficients.py'])
        run([py, 'figures/plot_fig7_risk_sensitivity.py'])
        run([py, 'figures/plot_fig8_type_improvement.py'])
        run([py, 'figures/plot_fig9_typical_curves.py'])

    if args.verify:
        run([py, 'verify_package.py'])


if __name__ == '__main__':
    main()

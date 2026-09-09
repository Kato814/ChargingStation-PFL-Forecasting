from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paper_style import CN, apply_tick_fonts

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"pure_fast": "快充站", "pure_slow": "慢充站", "mixed_ac_dc": "混合站"}
ORDER = ["pure_fast", "pure_slow", "mixed_ac_dc"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot type-level aggregate WAPE improvements for paper Fig. 8.")
    parser.add_argument("--output", type=Path, default=ROOT / "figures_generated/fig8_type_improvement.png")
    parser.add_argument("--svg-output", type=Path, default=ROOT / "figures_generated/fig8_type_improvement.svg")
    args = parser.parse_args()
    table = pd.read_csv(ROOT / "results_reproduced/type_aggregate_results.csv").set_index("type").loc[ORDER]
    values = table["relative_improvement_percent"].to_numpy()

    fig, ax = plt.subplots(figsize=(3.35, 2.15), dpi=300)
    bars = ax.bar([LABELS[v] for v in ORDER], values, width=0.55)
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True, length=3.0)
    apply_tick_fonts(ax, 8)
    for label in ax.get_xticklabels():
        label.set_fontproperties(CN)
        label.set_fontsize(8)
    ax.set_ylabel("类型聚合WAPE相对改善 / %", fontsize=9, fontproperties=CN)
    ax.set_ylim(0, max(values) * 1.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.15, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(args.svg_output, bbox_inches="tight", facecolor="white")
    print(args.output)
    print(args.svg_output)


if __name__ == "__main__":
    main()

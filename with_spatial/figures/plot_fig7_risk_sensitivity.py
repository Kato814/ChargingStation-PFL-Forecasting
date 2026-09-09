from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_style import CN, EN, apply_tick_fonts

ROOT = Path(__file__).resolve().parents[1]


def _panel(ax, x, wape, risk_percent, selected_x, xlabel: str) -> None:
    line1, = ax.plot(x, wape, marker="o", markersize=3.2, linewidth=1.0, label="聚合WAPE")
    ax.set_xlabel(xlabel, fontsize=9, fontproperties=CN)
    ax.set_ylabel("聚合WAPE", fontsize=9, fontproperties=CN)
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=False, length=3.0)
    apply_tick_fonts(ax, 7.8)
    ax.set_ylim(0.0934, 0.0957)

    ax2 = ax.twinx()
    line2, = ax2.plot(x, risk_percent, marker="s", markersize=3.0, linewidth=1.0, linestyle="--", label="最坏站点相对退化率")
    ax2.set_ylabel("最坏站点相对退化率 / %", fontsize=9, fontproperties=CN)
    ax2.tick_params(direction="in", top=True, right=True, length=3.0)
    apply_tick_fonts(ax2, 7.8)
    ax2.set_ylim(0.8, 4.3)

    ax.axvline(selected_x, color="0.45", linestyle=":", linewidth=0.9, label="本文取值")
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.85)

    return line1, line2


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot risk-control parameter sensitivity for paper Fig. 7.")
    parser.add_argument("--output", type=Path, default=ROOT / "figures_generated/fig7_risk_sensitivity.png")
    parser.add_argument("--svg-output", type=Path, default=ROOT / "figures_generated/fig7_risk_sensitivity.svg")
    args = parser.parse_args()

    delta = pd.read_csv(ROOT / "results_reference/delta_sensitivity_beta_003.csv")
    beta = pd.read_csv(ROOT / "results_reference/beta_sensitivity_delta_m002.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.45), dpi=300)

    l1, l2 = _panel(
        axes[0],
        delta["delta"].to_numpy(),
        delta["WAPE"].to_numpy(),
        100.0 * delta["max_relative_degradation"].to_numpy(),
        -0.02,
        r"近期表现阈值 $\delta$",
    )
    axes[0].set_xticks(delta["delta"].to_numpy())

    _panel(
        axes[1],
        beta["beta"].to_numpy(),
        beta["WAPE"].to_numpy(),
        100.0 * beta["max_relative_degradation"].to_numpy(),
        0.03,
        r"累计损失预算 $\beta$",
    )
    axes[1].set_xticks(beta["beta"].to_numpy())

    # One shared legend. The vertical dotted line has the same meaning in both panels.
    proxy_selected, = axes[0].plot([], [], color="0.45", linestyle=":", linewidth=0.9, label="本文取值")
    fig.legend(
        [l1, l2, proxy_selected],
        ["聚合WAPE", "最坏站点相对退化率", "本文取值"],
        loc="upper center",
        ncol=3,
        frameon=False,
        prop=CN,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 1.02),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.09, right=0.92, bottom=0.22, top=0.78, wspace=0.48)
    fig.savefig(args.output, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(args.svg_output, bbox_inches="tight", facecolor="white")
    print(args.output)
    print(args.svg_output)


if __name__ == "__main__":
    main()

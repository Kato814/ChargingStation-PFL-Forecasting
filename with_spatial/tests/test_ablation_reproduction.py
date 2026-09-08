from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    subprocess.run([sys.executable, script], cwd=ROOT, check=True)


def main() -> None:
    run("experiments/reproduce_table3_shared_prior_ablation.py")
    run("experiments/reproduce_table4_ablation.py")

    table3 = pd.read_csv(ROOT / "results_reproduced/table3_shared_prior_ablation.csv")
    expected3 = {
        "仅全局先验": (0.09521661889785651, 7, 1.5618088958722622),
        "仅类型先验": (0.09436512452397242, 7, 1.4747178623018375),
        "固定等权融合": (0.09362815786382665, 8, 1.5763687545546097),
        "验证选择或组合": (0.09357050219189945, 7, 1.5618088958722622),
        "无标量个性化": (0.09677532999527914, 6, 2.8662148917033816),
    }
    for _, row in table3.iterrows():
        wape, improved, worst = expected3[row["构型"]]
        if not np.isclose(row["总体WAPE"], wape, atol=1e-12, rtol=0.0):
            raise AssertionError(row)
        if int(row["改善站点数"]) != improved or not np.isclose(row["最坏站点相对退化率/%"], worst, atol=1e-10, rtol=0.0):
            raise AssertionError(row)

    table4 = pd.read_csv(ROOT / "results_reproduced/table4_component_ablation.csv")
    expected4 = {
        "前日负荷基准": (0.10042573373583695, 0.22599867420833705, 0, 0.0),
        "标量个性化": (0.11013379799093936, 0.2197047890174864, 7, 20.181868951240585),
        "＋站群总量校正": (0.09460375672698168, 0.211033347994949, 8, 11.62044698956237),
        "＋近期表现约束": (0.09460375672698168, 0.2122534215433947, 7, 6.6347990258662115),
        "＋累计损失约束": (0.09357050219189945, 0.21098808555609647, 7, 1.5618088958722622),
    }
    for _, row in table4.iterrows():
        wape, pooled, improved, worst = expected4[row["构型"]]
        if not np.isclose(row["聚合WAPE"], wape, atol=1e-12, rtol=0.0):
            raise AssertionError(row)
        if not np.isclose(row["全样本WAPE"], pooled, atol=1e-12, rtol=0.0):
            raise AssertionError(row)
        if int(row["改善站点数"]) != improved:
            raise AssertionError(row)
        if not np.isclose(row["最坏站点相对退化率/%"], worst, atol=1e-10, rtol=0.0):
            raise AssertionError(row)
    print("PASS")


if __name__ == "__main__":
    main()

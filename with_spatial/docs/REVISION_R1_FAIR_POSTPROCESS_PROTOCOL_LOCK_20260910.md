# R1 固定后处理公平性扩展实验协议锁（2026-09-10）

## 1. 目的

本实验仅回应审稿人关于“本文优势是否主要来自站群校正和风险筛选”的公平性问题。
在原 R1-E2 四个已有逐点 OOF 基线基础上，加入已经按锁定协议完成的 FedProx 和 FedRep，
并对六种基线统一施加本文相同的站群总量校正与负迁移控制。

该实验不重新训练任何模型，不修改 V1.0 主方法，不改变 539D 输入、五折部署组、
正式评价口径或风险参数。

## 2. 输入

原四个基线来自冻结 OOF：

- FedAvg 残差
- 全参数微调
- 本地 MLP
- pFedMe 微调

新增基线来自：

- `results_revision_r1/neural_baselines/fedprox_oof.csv.gz`
- `results_revision_r1/neural_baselines/fedrep_oof.csv.gz`

FedProx 和 FedRep 均已按 `a790d4b` 锁定的 5 折 × 3 seed × 30 rounds 协议生成。

## 3. 固定后处理

每个方法均依次评价：

1. `raw`
2. `+group`
3. `+group+risk`

其中参数直接读取 `paper_config.json` 中冻结的：

- `group_total_correction`
- `negative_transfer`

不针对任何基线重新选择 `gamma`、`delta`、`beta` 或历史窗口。
`gamma` 与风险状态仍按各方法自身此前历史表现因果更新。

## 4. 公平性边界

该实验是固定参数的 downstream control experiment。
它用于判断站群校正和风险控制是否具有跨预测器的通用作用，并检验本文方法在统一后处理条件下的相对位置。

不得将其表述为：

- 每个基线都获得了各自最优的后处理调参；
- 本文全部性能收益均来自 PFL；
- 相同后处理后本文必然显著优于所有基线。

## 5. 一致性检查

扩展脚本会重新计算原 E2 四个方法，并与既有
`r1_postprocessing_fairness_summary.csv` 和
`r1_postprocessing_fairness_daily_inference.json`
进行数值一致性检查。

只有原 E2 结果在 `1e-12` 数值容差内保持一致，扩展实验才视为通过。

## 6. 输出

轻量正式证据：

- `r1_postprocessing_fairness_extended_summary.csv`
- `r1_postprocessing_fairness_extended_daily_inference.json`
- `r1_postprocessing_fairness_extended_metadata.json`

大体量诊断文件：

- `r1_postprocessing_fairness_extended_oof.csv.gz`

大体量 OOF 仅保留在本地/归档，不进入普通 Git 历史。

## 7. 实现锁

正式实验脚本：

`with_spatial/experiments/r1_postprocessing_fairness_extended.py`

脚本 SHA-256：

`38af7fb9402deaf91363b4b33d95e32126add1783f62d78b931c960ba5314863`

实验结果审核通过后，再将轻量结果、结论摘要及状态更新提交到 `revision/r1-202609`。

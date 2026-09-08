# 结果来源与证据层级

本文件说明 `with_spatial/` 中各类结果是如何获得的，避免后续把“重新运行结果”“冻结逐点结果”“仅有汇总值的历史对比结果”混为一谈。

## A. 本文主方法

状态：**可端到端重新运行并与冻结逐点结果比较**。

入口：

- `run_paper_pipeline.py`
- `src/pipeline.py`

主要输出：

- `results_reproduced/summary.json`
- `results_reproduced/station_results.csv`
- `results_reproduced/type_aggregate_results.csv`
- `results_reproduced/paired_daily_statistics.json`
- 完整冻结档案中的 `results_reproduced/oof_predictions.csv.gz`

V1.0 逐点结果共42240条，清洁重构主链与原最终实验逐点预测差异处于约 `1e-12` 数值量级。

## B. 表3共享先验与标量适配消融

状态：**由清洁主方法代码重新运行**。

入口：`experiments/reproduce_table3_shared_prior_ablation.py`

结果：`results_reproduced/table3_shared_prior_ablation.csv`

包括：仅全局先验、仅类型先验、固定等权融合、验证选择/组合、取消标量个性化。

## C. 表4关键组件累积消融

状态：**由清洁主方法代码重新运行**。

入口：`experiments/reproduce_table4_ablation.py`

结果：`results_reproduced/table4_component_ablation.csv`

包括：前日基准、标量个性化、站群总量校正、近期表现约束、累计损失约束。

## D. 表2神经网络和联邦对比方法

状态分两类。

### 有冻结逐点OOF预测，可直接重新评价

- 前日负荷基准；
- FedAvg残差；
- 全参数微调；
- 本地MLP；
- pFedMe微调；
- 本文方法。

入口：

- `experiments/reproduce_table2_baselines.py`
- `experiments/reproduce_comparison_daily_inference.py`

### 只有锁定汇总结果

- 本地滚动岭回归。

其论文表2锁定结果为 WAPE `0.1025`、nRMSE `0.0727`。当前归档没有该方法对应的逐点OOF预测，因此不得伪造或声称能够从当前Git轻量文件重新训练得到完全相同结果。

## E. 图件

- 图6滚动标量系数分布：来自本文主方法逐站逐日 `alpha`；
- 分类型聚合改善图：来自 `type_aggregate_results.csv`；
- 三类典型站连续三日曲线：来自冻结/重现的OOF逐点预测，典型窗口默认 `2023-02-15` 至 `2023-02-17`。

图件脚本位于 `figures/`。完整冻结档案中保留对应可复查的绘图数据和生成图。

## F. 冻结证据与Git轻量记录

普通Git历史有意不保存大型 `.pkl`、完整5 min CSV和大型逐点证据文件。完整证据由 SHA-256 匹配的 `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip` 保存。

Git中的结果CSV/JSON用于：

- 快速审计；
- 版本差异检查；
- 单元/回归测试目标；
- 避免后续维护时丢失论文口径。

若Git记录与SHA匹配的冻结V1.0压缩包冲突，以冻结包、验证报告和明确的版本变更记录进行联合审计，不得静默覆盖。

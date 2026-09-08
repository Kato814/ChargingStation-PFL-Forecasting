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

V1.0逐点结果共42240条，清洁重构主链与原最终实验逐点预测差异处于约 `1e-12` 数值量级。

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

冻结补充证据副本：`results_reference/component_ablation_source.csv`。

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

## E. 风险参数敏感性

状态：**冻结补充实验轻量证据，可直接重绘，但当前Git主流程不重新搜索整套参数网格**。

来源：论文核心问题与证据重构补充实验包中的 `delta_beta_sensitivity_grid.csv` 及两个论文切片。

Git固定副本：

- `results_reference/risk_sensitivity_grid.csv`；
- `results_reference/delta_sensitivity_beta_003.csv`；
- `results_reference/beta_sensitivity_delta_m002.csv`。

论文选取点：`delta=-0.02`、`beta=0.03`。

绘图入口：`figures/plot_fig7_risk_sensitivity.py`。

该证据与主方法V1.0参数口径一致，但不应被描述成 `reproduce_all.py` 每次都重新训练/搜索了25个参数组合；当前仓库复现的是冻结的敏感性结果图。

## F. 图件

- 图6滚动标量系数分布：来自本文主方法逐站逐日 `alpha`；
- 图7风险参数敏感性：来自上述冻结补充实验敏感性CSV；
- 图8分类型聚合改善：来自 `results_reproduced/type_aggregate_results.csv`；
- 图9三类典型站连续三日曲线：来自冻结/重现的OOF逐点预测，典型窗口默认 `2023-02-15` 至 `2023-02-17`。

图件脚本位于 `figures/`。具体数据关系与解释边界见 `FIGURE_REPRODUCTION.md`。

## G. 冻结证据与Git轻量记录

普通Git历史有意不保存大型 `.pkl`、完整5 min CSV和大型逐点证据文件。完整证据由 SHA-256 匹配的 `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip` 保存。

Git中的结果CSV/JSON用于：

- 快速审计；
- 版本差异检查；
- 单元/回归测试目标；
- 避免后续维护时丢失论文口径。

冻结档案校验值和包摘要位于 `archive/`；源站点数据文件级SHA-256位于 `data/source_station_dataset/dataset_manifest.csv`。

若Git记录与SHA匹配的冻结V1.0压缩包冲突，以冻结包、验证报告和明确的版本变更记录进行联合审计，不得静默覆盖。

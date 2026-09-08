# 论文表格、图件与代码/结果索引

本文件按当前论文正文编号记录 `with_spatial/` 539维权威版本与论文主要数值表格和结果图之间的对应关系。后续论文图表编号若调整，应同步更新本文件，但不得改变冻结结果数据。

## 表1 模型参数设置

主要来源：

- `paper_config.json`
- `src/config.py`
- `docs/FOLD_PROTOCOL.md`
- `docs/BASELINE_ALIGNMENT_AUDIT.md`

注意：折级 `ridge_alpha`、`type_weight`、标量历史窗口和正则化参数按折冻结；神经网络本地更新对比方法实际采用30 d目标站历史，本地滚动岭回归采用60 d。

## 表2 不同方法的折外总体聚合预测结果

复现入口：

```bash
python experiments/reproduce_table2_baselines.py
```

轻量结果：

- `results_reproduced/baselines/table2_comparison_results.csv`

配套日级统计：

```bash
python experiments/reproduce_comparison_daily_inference.py
```

- `results_reproduced/baselines/paired_daily_comparisons.json`

结果来源边界见 `RESULT_PROVENANCE.md`。本地滚动岭回归仅保存锁定汇总结果，没有逐点OOF证据。

## 表3 共享先验及标量个性化消融结果

复现入口：

```bash
python experiments/reproduce_table3_shared_prior_ablation.py
```

结果：

- `results_reproduced/table3_shared_prior_ablation.csv`

## 表4 关键组件累积消融

复现入口：

```bash
python experiments/reproduce_table4_ablation.py
```

结果：

- `results_reproduced/table4_component_ablation.csv`
- `results_reference/component_ablation_source.csv`（冻结补充证据副本）

## 表5 目标站基准相对误差比及预测技能得分

主方法入口：

```bash
python run_paper_pipeline.py
```

结果：

- `results_reproduced/station_results.csv`

10座论文目标站及站点ID对应关系见 `FOLD_PROTOCOL.md`。

## 图5 目标站负荷规模及日内形态

脚本：

```bash
python figures/plot_fig5_target_characteristics.py
```

数据：恢复后的 `data/source_station_dataset/`。

输出包括：

- 日均充电量柱状图和CSV；
- 三类目标站归一化日内平均曲线、类型内最小—最大范围和CSV。

精确构造规则见 `FIGURE_REPRODUCTION.md`。

## 图6 不同目标站滚动标量系数分布

脚本：

```bash
python figures/plot_fig6_scalar_coefficients.py
```

数据：`results_reproduced/oof_predictions.csv.gz` 中的逐站逐日 `local_coefficient`。

## 图7 风险参数敏感性

脚本：

```bash
python figures/plot_fig7_risk_sensitivity.py
```

冻结补充实验数据：

- `results_reference/risk_sensitivity_grid.csv`
- `results_reference/delta_sensitivity_beta_003.csv`
- `results_reference/beta_sensitivity_delta_m002.csv`

论文取值：`delta=-0.02`、`beta=0.03`。

## 图8 不同类型目标站聚合WAPE相对改善

脚本：

```bash
python figures/plot_fig8_type_improvement.py
```

数据：`results_reproduced/type_aggregate_results.csv`。

## 图9 三类典型目标站连续三日预测曲线

脚本：

```bash
python figures/plot_fig9_typical_curves.py --start-date 2023-02-15
```

数据：`results_reproduced/oof_predictions.csv.gz`。

论文当前代表站点：F01/1011、S05/1566、M01/1599；窗口为2023-02-15至2023-02-17。

## 方法图1–图4

论文图1–图4主要是概念/方法框架图，不由数值脚本自动生成：

- 图1：多类型充电站负荷特征及多主体预测场景；
- 图2：基于个性化联邦学习的负荷预测框架；
- 图3：目标站个性化适配及站群总量校正；
- 图4：负迁移判别域及修正量重分配。

它们的计算逻辑分别由 `src/`、`CODE_TO_PAPER_MAPPING.md` 和 `METHOD_IMPLEMENTATION.md` 支撑。论文排版图本身不作为代码运行产物。

## 编号与结果冻结原则

- 论文排版调整导致图表编号变化时，只更新索引和图题映射；
- 结果数据变化必须经过新实验、新版本和Changelog；
- 不允许仅为匹配论文图表而人工修改CSV中的冻结结果；
- 图表数值与论文正文发生冲突时，先核查版本、折协议、指标定义和结果来源，不能静默覆盖。

# 论文结果图复现说明

本文件只记录 `with_spatial/` 539维权威版本当前维护的数值/数据结果图。方法框架图（论文图1–图4）由论文排版文件维护，不从数值代码自动生成。

## 图5：目标站负荷规模及日内形态

脚本：

`figures/plot_fig5_target_characteristics.py`

运行：

```bash
python figures/plot_fig5_target_characteristics.py
```

图5(a)数据来源：恢复后的 `data/source_station_dataset/{train,validation,test}/station_daily_features.csv`。

处理方式：

1. 按论文目标站映射选取F01、F02、S01–S06、M01、M02；
2. 每座目标站使用2022-09-01至2023-02-28共181 d的 `daily_total_load`；
3. 计算日均充电量并采用对数纵轴展示不同站点的数量级差异；
4. 柱体按快充、慢充、混合三类着色。

图5(b)数据来源：恢复后的10座目标站完整5 min `station_processed_5min/<station_id>.csv`。

处理方式：

1. 将每3个5 min点组成一个15 min时段；
2. 对每座站点跨181 d计算96点平均日内曲线；
3. 每座站点用自身平均日内曲线最大值归一化；
4. 在快充2站、慢充6站、混合2站内分别计算同一时段的均值、最小值和最大值；
5. 类型均值作为主曲线，类型内目标站最小—最大范围作为浅色带。

该算法可复现当前论文图5(b)所展示的三类代表性平均日内形态和类型内差异范围。

默认输出：

- `figures_generated/fig5a_station_daily_energy.png`
- `figures_generated/fig5a_station_daily_energy.svg`
- `figures_generated/fig5a_station_daily_energy_data.csv`
- `figures_generated/fig5b_type_profiles.png`
- `figures_generated/fig5b_type_profiles.svg`
- `figures_generated/fig5b_type_profiles_data.csv`

## 图6：不同目标站滚动标量系数分布

脚本：

`figures/plot_fig6_scalar_coefficients.py`

运行：

```bash
python figures/plot_fig6_scalar_coefficients.py
```

数据来源：`results_reproduced/oof_predictions.csv.gz`。

处理方式：

1. 对每个 `station_id`、`prediction_date` 读取当日唯一 `local_coefficient`；
2. 映射为论文目标站 F01、F02、S01–S06、M01、M02；
3. 每站44个测试日形成箱线分布；
4. 箱线图隐藏离群点圆圈，仅用于展示中位数、四分位范围和须；
5. 快充、慢充和混合站通过竖向虚线分组。

默认输出：

- `figures_generated/fig6_scalar_coefficients.png`
- `figures_generated/fig6_scalar_coefficients.svg`
- `figures_generated/fig6_scalar_coefficients_data.csv`

解释边界：`alpha` 表示共享残差修正幅度，不是预测精度指标。不能把更大的 `alpha` 解释为更高预测精度。

## 图7：风险参数敏感性

脚本：

`figures/plot_fig7_risk_sensitivity.py`

运行：

```bash
python figures/plot_fig7_risk_sensitivity.py
```

数据来源：论文补充实验的风险参数敏感性证据，已以轻量CSV固定在：

- `results_reference/risk_sensitivity_grid.csv`：5×5完整网格；
- `results_reference/delta_sensitivity_beta_003.csv`：固定 `beta=0.03` 的 `delta` 切片；
- `results_reference/beta_sensitivity_delta_m002.csv`：固定 `delta=-0.02` 的 `beta` 切片。

网格取值：

- `delta ∈ {-0.05, -0.02, 0, 0.02, 0.05}`；
- `beta ∈ {0, 0.01, 0.03, 0.05, 0.10}`。

论文选取点为 `delta=-0.02`、`beta=0.03`。图中同时展示聚合WAPE和最坏站点相对退化率，用于说明邻近参数范围内的精度—风险变化。

默认输出：

- `figures_generated/fig7_risk_sensitivity.png`
- `figures_generated/fig7_risk_sensitivity.svg`

重要来源边界：该敏感性网格来自论文补充实验证据包，不是由当前V1.0主流程在每次 `reproduce_all.py` 中重新搜索得到。Git仓库保存原始轻量敏感性CSV和绘图脚本，以保证论文图可追踪；如果未来要重新运行完整参数网格，应新建独立实验脚本和结果版本，不能静默替换这些冻结证据。

## 图8：不同类型目标站聚合WAPE相对改善

脚本：

`figures/plot_fig8_type_improvement.py`

运行：

```bash
python figures/plot_fig8_type_improvement.py
```

数据来源：`results_reproduced/type_aggregate_results.csv`。

V1.0锁定相对改善：

- 快充站：4.6184%；
- 慢充站：10.5818%；
- 混合站：6.9191%。

默认输出：

- `figures_generated/fig8_type_improvement.png`
- `figures_generated/fig8_type_improvement.svg`

类型WAPE必须先在同类型目标站之间逐时聚合，再计算WAPE，不得改成站点WAPE简单平均。

## 图9：三类典型目标站连续三日预测曲线

脚本：

`figures/plot_fig9_typical_curves.py`

论文当前推荐窗口：

```bash
python figures/plot_fig9_typical_curves.py --start-date 2023-02-15
```

对应日期：2023-02-15 至 2023-02-17。

代表站点：

- F01：快充站，原始站点ID 1011；
- S05：慢充站，原始站点ID 1566；
- M01：混合站，原始站点ID 1599。

数据来源：`results_reproduced/oof_predictions.csv.gz`。

图形编码：

- 实际负荷：灰色填充；
- 基线方法：橙色虚线；
- 本文方法：蓝色实线；
- 横轴：连续72 h；
- 每日96点、15 min分辨率。

默认输出：

- `figures_generated/fig9_typical_curves.png`
- `figures_generated/fig9_typical_curves.svg`
- `figures_generated/fig9_typical_curves_data.csv`

该三日窗口用于展示三类站点的代表性运行形态和预测跟踪，不作为额外统计检验样本。典型案例选择不能替代全文44日、10座目标站的总体评价。

## 绘图风格

公共绘图字体/样式定义位于 `figures/paper_style.py`。当前论文结果图遵循：

- 中文宋体兼容字体；
- 西文/数字Times New Roman兼容字体；
- 无网格线；
- 坐标刻度线朝内；
- 主文图默认输出600 dpi PNG，同时保留SVG矢量版本；
- 图内不放论文图题，图题由Word正文管理。

如果仅修改配色、字号、线宽或版面，不改变绘图数据和计算口径，可视为排版维护；如果修改数据筛选、指标定义、风险参数网格或典型站点/日期的科学选择原则，应在Changelog中记录。

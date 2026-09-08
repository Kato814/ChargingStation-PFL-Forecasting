# 论文结果图复现说明

本文件只记录 `with_spatial/` 539维权威版本当前维护的结果图。方法框架图（论文图1–图4）由论文排版文件维护，不从数值代码自动生成。

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

如果仅修改配色、字号、线宽或版面，不改变绘图数据和计算口径，可视为排版维护；如果修改数据筛选、指标定义或典型站点/日期的科学选择原则，应在Changelog中记录。

# 三类型充电站缺口修复与边缘平滑数据集

- 上游数据：`站点数据集_three_type_cleaned_deepsmooth_v2_adjcap160_20260710`
- 站点数量：40
- 5 min记录数：2085120
- 时间范围：2022-09-01 至 2023-02-28
- 站点类型：`pure_fast`、`pure_slow`、`mixed_ac_dc`

## 处理说明

本数据集使用占用状态约束修复持续占用条件下的异常低负荷缺口，再执行短脉冲修复和边缘保持平滑；混合站额外采用35 min范围内的稳健时域平滑。缺口补入电量保留并单独审计，平滑阶段保持缺口修复后曲线的逐日能量。

`station_processed_5min/*.csv` 仅更新 `volume`；其他原字段保持不变。`station_daily_features.csv`、`station_selection.csv` 和 `balanced_split_mapping.csv` 已按清洗后负荷刷新。

## 完整性

- 所有站点均通过形态、峰值、日能量和相邻跳变约束。
- 完整冻结包中的 `dataset_manifest.csv` 记录发布数据文件的大小和 SHA-256。
- 详细逐点日志和分类型审计位于历史数据清洗工作区，不进入论文代码主路径。

## GitHub归档说明

普通 Git 历史不保存 40 座站点完整 5 min CSV。完整数据由 `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip` 冻结归档恢复；本目录在 Git 中仅保存数据来源、清洗摘要、站点划分和完整性元数据。

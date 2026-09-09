# 数据接口与输入契约

## 1. 本版本定位

本版本是**有空间输入特征的原结果精确复现版**。论文方法主链的规范输入为：

- 40 座充电站客户端；
- 每个站点 180 个日样本；
- 每个日样本 539 维输入；
- 输出为 96 个 15 min 时段的跨日残差/负荷曲线；
- 训练期：2022-09-02—2022-12-15；
- 验证期：2022-12-16—2023-01-15；
- 测试期：2023-01-16—2023-02-28。

`data/clients_with_spatial.pkl` 是冻结核验的规范模型输入缓存。`src/feature_builder_with_spatial.py` 可从冻结源CSV独立重建该缓存，不依赖历史Stage类或旧工程模块。

## 2. 客户端字段

每个站点包含：

- `station_id`：站点编号；
- `station_type`：`pure_fast`、`pure_slow` 或 `mixed_ac_dc`；
- `x`：`[days, 539]` 输入矩阵；
- `y_residual`：`[days, 96]` 跨日残差；
- `baseline`：`[days, 96]` 前日负荷基准；
- `y_true`：`[days, 96]` 实际负荷；
- `forecast_date`：预测日期。

## 3. 539维输入

`data/feature_schema_539.csv` 列出全部特征名称及顺序。本版本保留原最终实验实际使用的空间输入，包括：

- POI统计特征；
- 经度、纬度；
- 分区充电设施数量；
- 分区面积。

此外还包含历史负荷曲线、日期/节假日属性、站点设备与容量属性及最终实验实际使用的其他非空间特征。该539维结构只属于精确复现版；无空间版本单独维护。

## 4. 联邦充分统计量

完整冻结档案中的 `data/federated_statistics_with_spatial.pkl` 保存每座训练站的可加充分统计量：

- `n`；
- `x_sum`；
- `xtx`；
- `y_sum`；
- `xty`。

`prepare_inputs.py` 可从客户端缓存重新计算这些统计量。

## 5. 数值等价规则

结构信息（站点集合、特征名、日期、维数、站点类型）必须完全一致。浮点充分统计量在同一数值环境中可达到逐元素一致；跨BLAS/CPU环境允许机器精度级舍入差异。GitHub维护版测试采用 `rtol=0, atol=1e-10`，详见 `docs/ENVIRONMENT_AND_NUMERICS.md`。

## 6. 大文件恢复

普通Git历史不包含 `clients_with_spatial.pkl`、`federated_statistics_with_spatial.pkl` 和40站完整5 min源CSV。运行完整重建或端到端复现前，应从匹配SHA-256的V1.0冻结ZIP恢复到本文档约定路径。

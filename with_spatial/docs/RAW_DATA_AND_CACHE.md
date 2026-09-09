# 原始站点数据与539维模型输入缓存

## 1. 两层数据

冻结V1.0包同时保存：

1. `data/source_station_dataset/`：40座站点的源CSV与静态/日级辅助表；
2. `data/clients_with_spatial.pkl`：论文最终实验直接使用的539维日级模型输入缓存。

普通Git历史只保存构建代码、schema、manifest和轻量审计记录；完整5 min源CSV与大型pkl从冻结ZIP恢复。

## 2. 清洁特征构建模块

`src/feature_builder_with_spatial.py` 从源CSV直接执行：

- 5 min记录聚合为15 min；
- 构造每日96点实际负荷与前日基准；
- 构造历史曲线、日期属性、站点静态属性及空间静态特征；
- 构造跨日残差；
- 按最终实验口径处理/排除不进入最终模型输入的候选项；
- 输出40站、180日/站、539维输入。

运行：

```bash
python prepare_inputs.py --rebuild-cache-from-raw
```

默认输出重建缓存，不覆盖冻结缓存。

## 3. 等价验收

完整40站重构的冻结审计记录表明：

- 站点集合一致；
- 日期一致；
- 站点类型一致；
- 539个特征名称及顺序一致；
- `x`、`y_residual`、`baseline`、`y_true` 与冻结缓存一致。

相关轻量记录应保留在Git中：

- `data/feature_schema_539.csv`；
- `data/raw_to_cache_equivalence.json`；
- `data/raw_to_cache_audit.csv`；
- `data/source_station_dataset_manifest.csv`；
- `data/source_station_dataset_summary.json`。

## 4. 联邦统计量

`prepare_inputs.py` 可从539维客户端缓存重建训练站充分统计量。跨数值环境比较使用 `rtol=0, atol=1e-10`；该容差只用于浮点中间量，离散结构和数据清单仍要求完全一致。

## 5. 长期修改建议

若以后修改特征：

1. 不覆盖539维V1.0冻结缓存；
2. 新建版本/分支；
3. 从源CSV重新构建输入；
4. 重新生成充分统计量；
5. 完整运行五折端到端回归；
6. 记录新旧feature schema和结果差异。

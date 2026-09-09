# 40站源数据清洗记录

本文件记录冻结源站点数据集随V1.0归档保存的清洗摘要。为避免把历史本地绝对路径写入维护仓库，只保留可复现参数和数值审计信息。

## 总体规模

- 站点数：40；
- 5 min记录总行数：2,085,120；
- 快充站：7；
- 慢充站：24；
- 混合站：9；
- gap数量：424；
- gap修复点数：1,974；
- 修复填补电量：20,424.1112455552 kWh；
- 变化幅度≥0.1 kW的点：1,341,819；
- 全部清洗质量门通过：true；
- 数值核心SHA-256：`004c1bf6a76e16dabad4794cad16ec63414b2f4f10bbba64fdc795bffd6032da`。

## Gap repair参数

- `max_gap_points=72`
- `minimum_expected_kw=60.0`
- `maximum_gap_load_kw=40.0`
- `low_load_ratio=0.3`
- `recovered_ratio=0.6`
- `context_points=6`
- `minimum_context_points=2`
- `maximum_occupancy_ratio_to_context=2.0`
- `occupancy_context_tolerance=2.0`
- `shoulder_points=3`
- `minimum_scale=0.5`
- `maximum_scale=1.5`
- `interval_minutes=5.0`

## Edge-preserving despike参数

- `median_window=7`
- `mad_window=13`
- `max_pulse_points=3`
- `minimum_impulse_kw=8.0`
- `hampel_k=2.2`
- `pulse_blend=1.0`
- `return_tolerance_kw=12.0`
- `return_ratio=0.45`
- `bilateral_radius=4`
- `bilateral_time_sigma=2.0`
- `bilateral_range_floor_kw=8.0`
- `bilateral_range_scale=0.085`
- `bilateral_blend=0.84`
- `bilateral_passes=3`

## 混合站平滑参数

- `radius=3`
- `sigma=1.5`
- `blend=0.88`
- `passes=1`

## 完整文件级校验

`data/source_station_dataset_manifest.csv` 保存冻结数据集中56个文件的相对路径、大小和SHA-256。完整数据从V1.0 ZIP恢复后，可运行：

```bash
python tests/test_source_dataset_manifest.py
```

该记录用于追踪论文实验实际使用的数据版本，不代表仓库未来研究必须继续采用相同清洗策略。

# 539维输入特征结构说明

完整逐维schema保存在冻结V1.0归档的 `data/feature_schema_539.csv`。GitHub维护版以本说明和 `src/feature_builder_with_spatial.py` 作为可读结构记录；完整CSV随冻结归档恢复。

## 特征组计数

| 特征组 | 维数 |
|---|---:|
| 前日96点曲线 | 96 |
| 历史汇总/历史模板 | 319 |
| 相似日96点曲线 | 96 |
| 日历属性 | 5 |
| 站点非空间静态属性 | 5 |
| 站点类型独热编码 | 6 |
| POI空间特征 | 8 |
| 其他空间静态特征 | 4 |
| **合计** | **539** |

## 12个空间输入

这12维是有空间版本与无空间版本的核心差异：

| 索引（0-based） | 特征名 | 类别 |
|---:|---|---|
| 516 | `static_poi_1km_total` | POI |
| 517 | `static_poi_1km_business_and_residential` | POI |
| 518 | `static_poi_1km_food_and_beverage_services` | POI |
| 519 | `static_poi_1km_lifestyle_services` | POI |
| 520 | `static_poi_3km_total` | POI |
| 521 | `static_poi_3km_business_and_residential` | POI |
| 522 | `static_poi_3km_food_and_beverage_services` | POI |
| 523 | `static_poi_3km_lifestyle_services` | POI |
| 524 | `static_longitude` | 空间静态 |
| 525 | `static_latitude` | 空间静态 |
| 526 | `static_zone_charge_count` | 空间静态 |
| 527 | `static_zone_area` | 空间静态 |

## 重要边界

- `with_spatial/` 保留上述12维，因为它们存在于原最终实验的539维输入中。
- 论文正文如果不披露空间输入，不应把本版本描述为“正文输入完全一致版”；本版本的职责是**原结果精确复现**。
- `no_spatial/` 删除这12维，形成527维研究版本，两者不得混用缓存、充分统计量或结果。
- 特征的精确名称、顺序及逐维索引以冻结归档中的 `feature_schema_539.csv` 和 `feature_builder_with_spatial.py` 为准。

# ChargingStation PFL Paper Code — With Spatial Inputs

论文：**《基于个性化联邦学习的多类型充电站负荷预测及负迁移抑制方法》**。

本目录是当前仓库的**优先维护版本**，用于保存论文原最终实验的539维输入口径、完整方法逻辑和可追溯复现链。它只对应论文日前预测，不包含日内滚动预测。

## 版本定位

- 采用最终优化实验实际使用的539维输入；
- 保留经纬度、POI与空间派生静态特征；
- 作为原最终实验的精确复现档和后续代码维护基线；
- 主方法代码按论文第2章结构组织，不依赖Stage 6/7/8、patch或旧FedProx历史入口运行本文方法；
- 对应无空间研究版本为 `../no_spatial/`，两个版本不得混用缓存、充分统计量、结果和结论。

冻结归档：

- `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`
- SHA-256: `8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`
- Private Release: `https://github.com/Kato814/ChargingStation-PFL-Forecasting/releases/tag/v1.0-archive`

Release中已保存完整ZIP及配套 `.sha256` 文件；GitHub记录的ZIP asset digest与上述SHA-256一致。

原最终优化源包：

- `ThreeType_ChargingLoadForecasting_FINAL_OPTIMIZED_20260715(1).zip`
- SHA-256: `ec5618f9a979049c57659c51716929514a6897d829652ba626e8b10a1e1be8ed`

## 快速运行

先从Private Release下载并校验冻结ZIP，恢复完整 `data/` 与 `results_reference/oof_predictions_source.csv.gz`，然后：

```bash
python prepare_inputs.py
python run_paper_pipeline.py
python reproduce_all.py
python verify_package.py
```

完整恢复步骤见 `docs/RELEASE_RESTORE_CHECKLIST.md`。

## 论文代码映射

- 2.1 联邦残差预测模型 → `src/federated_residual_model.py`
- 2.2.1 站点个性化适配 → `src/target_scalar_adaptation.py`
- 2.2.2 站群总量校正 → `src/group_total_correction.py`
- 2.3 负迁移抑制 → `src/negative_transfer_control.py`
- 第3章评价 → `src/evaluation.py`
- 五折端到端流程 → `src/pipeline.py`
- 源CSV→539维输入 → `src/feature_builder_with_spatial.py`

详细映射见 `docs/CODE_TO_PAPER_MAPPING.md`。

## 输入与数据记录

- 完整输入维数：539；
- 其中12维为空间输入，索引与名称见 `docs/FEATURE_SCHEMA_SUMMARY.md`；
- 539维分组索引见 `data/feature_group_ranges_539.csv`；
- 完整逐维特征顺序保存在冻结归档，可在恢复 `data/clients_with_spatial.pkl` 后运行 `tools/export_feature_schema.py` 导出；
- 40座站点源数据的划分、清洗摘要和逐文件SHA-256记录在 `data/source_station_dataset/`；
- 源数据到539维冻结缓存的等价核验见 `data/raw_to_cache_equivalence.json` 与 `docs/RAW_DATA_AND_CACHE.md`。

## 对比方法协议

- 全参数微调、本地MLP和pFedMe微调：30 d目标站历史窗口；
- 本地滚动岭回归：60 d历史窗口；
- 本地滚动岭回归当前只有锁定汇总结果，无逐点OOF预测。

详见 `docs/BASELINE_ALIGNMENT_AUDIT.md` 与 `docs/RESULT_PROVENANCE.md`。

## 论文正式评价与结果

论文日级配对统计保留五折各自两站部署组边界。仓库只维护这一论文口径，测试会阻止已废弃的另一日级聚合口径重新进入论文代码。

核心V1.0结果：

- 折外逐点预测：42240条；
- 总体聚合WAPE：0.0935705022；
- nRMSE：0.0660829513；
- 全样本WAPE：0.2109880856；
- 7/10座目标站改善；
- 最坏站点相对退化：1.5618%；
- 日级部署组平均WAPE差：-0.0115632639。

详细指标定义见 `docs/METRIC_DEFINITIONS.md`，冻结数值见 `docs/RESULTS_BASELINE.md`。

## 论文结果图

当前仓库维护：

- `figures/plot_fig5_target_characteristics.py`：目标站日均充电量及三类目标站归一化日内形态；
- `figures/plot_fig6_scalar_coefficients.py`：不同目标站滚动标量系数分布；
- `figures/plot_fig7_risk_sensitivity.py`：近期表现阈值与累计损失预算敏感性；
- `figures/plot_fig8_type_improvement.py`：三类目标站聚合WAPE相对改善；
- `figures/plot_fig9_typical_curves.py`：三类典型目标站连续三日预测曲线，论文推荐窗口为2023-02-15至2023-02-17。

图5由冻结源站点数据构造；图7敏感性数据来自冻结补充实验证据；图6、图8和图9由主方法折外结果或类型汇总结果直接生成。详细数据来源和复现边界见 `docs/FIGURE_REPRODUCTION.md` 与 `docs/RESULT_PROVENANCE.md`。

## 重要文档

建议按以下顺序阅读：

1. `docs/ARCHIVE_RECORD.md`：冻结档案、结果和修改政策；
2. `docs/CODE_TO_PAPER_MAPPING.md`：论文章节与代码模块映射；
3. `docs/METHOD_IMPLEMENTATION.md`：方法实现细节；
4. `docs/DATA_CONTRACT.md`：数据接口和输入边界；
5. `docs/FEATURE_SCHEMA_SUMMARY.md`：539维结构和12维空间输入；
6. `docs/FOLD_PROTOCOL.md`：五折目标站轮换与折级参数；
7. `docs/METRIC_DEFINITIONS.md`：论文评价口径；
8. `docs/RESULT_PROVENANCE.md`：主方法、消融、基线和图件的结果来源；
9. `docs/PAPER_ARTIFACT_INDEX.md`：论文表1–表5、图5–图9与代码/数据对应关系；
10. `docs/RESULTS_BASELINE.md`：V1.0冻结数值基线；
11. `docs/FIGURE_REPRODUCTION.md`：论文结果图数据来源与绘图入口；
12. `docs/ENVIRONMENT_AND_NUMERICS.md`：环境与浮点容差；
13. `docs/REPRODUCTION_GUIDE.md`：端到端复现步骤；
14. `docs/GITHUB_MAINTENANCE_GUIDE.md`：后续版本和分支管理。

## 大文件与长期归档

完整40站源数据、539维客户端缓存、联邦充分统计量、逐点锁定预测和冻结ZIP不进入普通Git历史。Git中保留文件级manifest、SHA-256、恢复路径和轻量结果。

完整冻结ZIP已经保存到Private GitHub Release `v1.0-archive`。普通Git与Private Release共同构成长期归档链。详见 `../docs/LARGE_FILES.md` 与 `docs/RELEASE_RESTORE_CHECKLIST.md`。

## 修改规则

V1.0是历史结果记录，不应原地覆盖。任何改变特征、折参数、模型形式、站群分配、负迁移判定或论文评价口径的科学修改，都应新建分支并升级版本，同时记录结果差异。

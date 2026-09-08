# ChargingStation PFL Paper Code — With Spatial Inputs

论文：**《基于个性化联邦学习的多类型充电站负荷预测及负迁移抑制方法》**。

本目录是当前仓库的**优先维护版本**，用于保存论文原最终实验的539维输入口径和可追溯复现链。它不包含日内滚动预测。

## 版本定位

- 采用最终优化实验实际使用的539维输入；
- 保留经纬度、POI与空间派生静态特征；
- 作为原最终实验的精确复现档和后续代码维护基线；
- 主方法代码按论文第2章结构组织，不依赖Stage 6/7/8、patch或旧FedProx历史入口运行本文方法；
- 对应无空间研究版本为 `../no_spatial/`，两个版本不得混用缓存、充分统计量、结果和结论。

冻结归档：

- `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`
- SHA-256: `8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`

原最终优化源包：

- `ThreeType_ChargingLoadForecasting_FINAL_OPTIMIZED_20260715(1).zip`
- SHA-256: `ec5618f9a979049c57659c51716929514a6897d829652ba626e8b10a1e1be8ed`

## 快速运行

先从匹配SHA-256的冻结归档恢复完整 `data/` 与 `results_reference/oof_predictions_source.csv.gz`，然后：

```bash
python prepare_inputs.py
python run_paper_pipeline.py
python reproduce_all.py
python verify_package.py
```

## 论文代码映射

- 2.1 联邦残差预测模型 → `src/federated_residual_model.py`
- 2.2.1 站点个性化适配 → `src/target_scalar_adaptation.py`
- 2.2.2 站群总量校正 → `src/group_total_correction.py`
- 2.3 负迁移抑制 → `src/negative_transfer_control.py`
- 第3章评价 → `src/evaluation.py`
- 五折端到端流程 → `src/pipeline.py`
- 源CSV→539维输入 → `src/feature_builder_with_spatial.py`

详细映射见 `docs/CODE_TO_PAPER_MAPPING.md`。

## 对比方法协议

- 全参数微调、本地MLP和pFedMe微调：30 d目标站历史窗口；
- 本地滚动岭回归：60 d历史窗口；
- 本地滚动岭回归当前只有锁定汇总结果，无逐点OOF预测。

## 论文正式日级统计

日级配对统计保留五折各自两站部署组边界。仓库只维护这一论文口径，测试会阻止已废弃的另一日级聚合数值重新进入代码或文档。

## 重要文档

- `docs/ARCHIVE_RECORD.md`：冻结档案、结果和修改政策
- `docs/METHOD_IMPLEMENTATION.md`：论文方法与实现细节
- `docs/DATA_CONTRACT.md`：539维输入及数据接口
- `docs/FEATURE_SCHEMA_SUMMARY.md`：特征组和12维空间输入
- `docs/RAW_DATA_AND_CACHE.md`：源CSV到模型缓存
- `docs/BASELINE_ALIGNMENT_AUDIT.md`：表2基线协议
- `docs/RESULTS_BASELINE.md`：冻结数值基线
- `docs/ENVIRONMENT_AND_NUMERICS.md`：环境与浮点容差

## 大文件

完整40站源数据、539维客户端缓存、联邦充分统计量、逐点锁定预测和冻结ZIP不进入普通Git历史。校验清单与恢复说明保存在Git中；完整冻结ZIP建议作为私有GitHub Release附件保存。详见 `../docs/LARGE_FILES.md`。

## 修改规则

V1.0是历史结果记录，不应原地覆盖。任何改变特征、折参数、基线协议、站群分配或负迁移判定的科学修改，都应新建分支并升级版本，同时记录结果差异。

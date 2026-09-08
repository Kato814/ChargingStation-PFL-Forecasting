# ChargingStation PFL Paper Code — With Spatial Inputs

论文：**《基于个性化联邦学习的多类型充电站负荷预测及负迁移抑制方法》**。

本工程是论文日前预测方法的**有空间输入特征原结果精确复现版**。不包含日内滚动预测。

## 版本定位

- 采用最终优化实验实际使用的 539 维输入；
- 保留经纬度、POI 与空间派生静态特征；
- 用于精确复现原最终实验结果和后续方法修改记录；
- 主方法代码按论文第2章结构组织，不依赖 Stage 6/7/8、patch 或 FedProx 历史入口运行本文方法；
- 对应的无空间研究版本为 `../no_spatial/`，两个版本不得混用缓存、结果和说明。

## 快速运行

完整档案恢复后：

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

## 对比方法协议

- 全参数微调、本地 MLP 和 pFedMe 微调：30 d目标站历史窗口；
- 本地滚动岭回归：60 d历史窗口；
- 本地滚动岭回归当前只有锁定汇总结果，无逐点OOF预测。

## 大文件

完整源数据、539维客户端缓存和联邦充分统计量位于冻结档案 `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip` 中。正常Git历史不保存这些大型文件；详见 `../docs/LARGE_FILES.md`。

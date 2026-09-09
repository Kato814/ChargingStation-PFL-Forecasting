# 论文正文—代码模块映射

对应论文：**基于个性化联邦学习的多类型充电站负荷预测及负迁移抑制方法**。

## 2.1 联邦残差预测模型

- `src/federated_residual_model.py`
  - `FederatedScaler`
  - `FederatedRidgeResidualModel.fit_from_statistics()`
  - `FederatedRidgeResidualModel.predict_residual()`
- `src/data_preparation.py`
  - 客户端输入契约
  - 局部充分统计量重建
- `src/feature_builder_with_spatial.py`
  - 源CSV到539维模型输入的完整构建链

## 2.2.1 站点个性化适配

- `src/target_scalar_adaptation.py`
  - `combine_shared_residual()`：全局/类型残差先验组合
  - `rolling_scalar_adaptation()`：滚动估计 `alpha_{j,d}` 并形成个性化预测候选

## 2.2.2 站群总量校正

- `src/group_total_correction.py`
  - `group_total_correction()`：汇总个性化修正量、基于历史估计 `gamma_d`、等额分配总量差额

## 2.3 负迁移抑制

- `src/negative_transfer_control.py`
  - `recent_performance_eligibility()`：近期表现约束
  - `cumulative_loss_guard()`：累计超额损失约束
  - `fallback_and_redistribute()`：基准回退与退出修正量重分配

## 第3章评价与实验

- `src/evaluation.py`
  - `aggregate_system_metrics()`：总体聚合 WAPE、nRMSE 等
  - `station_wape_table()`：站点级结果
  - `type_aggregate_wape_table()`：分类型逐时聚合 WAPE
  - `paired_daily_inference()`：五折独立两站部署组边界下的日级配对统计
- `experiments/reproduce_table2_baselines.py`
  - 表2：前日基准、FedAvg残差、全参数微调、本地MLP、本地岭回归锁定结果、pFedMe微调、本文方法
- `experiments/reproduce_comparison_daily_inference.py`
  - 本文方法相对前日基准及四种神经网络对比方法的部署组日级配对统计
- `experiments/reproduce_table3_shared_prior_ablation.py`
  - 表3：共享先验构成与标量个性化消融
- `experiments/reproduce_table4_ablation.py`
  - 表4：关键组件累积消融
- `figures/plot_fig6_scalar_coefficients.py`
  - 滚动标量系数分布
- `figures/plot_fig8_type_improvement.py`
  - 分类型聚合 WAPE 相对改善
- `figures/plot_fig9_typical_curves.py`
  - F01、S05、M01 连续三日预测曲线

## 端到端主入口

- `run_paper_pipeline.py`
  - 显式五折划分
  - 全局/类型联邦残差模型
  - 目标站标量适配
  - 站群总量校正
  - 负迁移抑制
  - 42240 条折外逐点预测
  - 主指标、站点结果、类型聚合结果、部署组日级配对统计

## 数据与特征构建

- 源CSV→539维最终实验输入：`src/feature_builder_with_spatial.py`
- 输入缓存与联邦统计量检查：`prepare_inputs.py`、`src/data_preparation.py`

## 表2对比方法

- 锁定逐点结果映射：`baselines/reference.py`
- 实验协议：`baselines/protocols.py`
- 表2复算：`experiments/reproduce_table2_baselines.py`
- 神经网络本地更新窗口30 d；本地滚动岭回归60 d。

## 版本边界

该映射仅对应 `with_spatial/` 539维精确复现版。无空间版本不得作为当前论文冻结结果的替代来源。

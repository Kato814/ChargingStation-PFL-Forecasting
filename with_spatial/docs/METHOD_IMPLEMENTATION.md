# 论文方法实现说明

## 2.1 联邦残差预测模型

对应 `src/federated_residual_model.py`。

1. 前日负荷作为本地基准；
2. 训练目标为跨日残差；
3. 每座训练站仅提供充分统计量；
4. 聚合端由统计量恢复标准化参数并闭式求解多输出岭回归；
5. 每折分别建立全局残差模型与三类类型残差模型。

联邦训练主链不依赖历史 Stage/FedProx 入口。局部充分统计量由 `src/data_preparation.py` 构建，源CSV到539维输入由 `src/feature_builder_with_spatial.py` 完成。

## 2.2.1 目标站标量适配

对应 `src/target_scalar_adaptation.py`。

- `combine_shared_residual()`：按折级权重组合全局/类型残差先验；
- `rolling_scalar_adaptation()`：仅使用预测日前历史估计 `alpha_{j,d}`；
- 个性化自由度为单个滚动标量，不更新539维模型参数。

## 2.2.2 站群总量校正

对应 `src/group_total_correction.py`。

- 汇总同期两座目标站的个性化修正；
- 仅使用预测日前5 d站群历史估计 `gamma_d`；
- 总量差额在同期目标站间等额分配。

## 2.3 负迁移抑制

对应 `src/negative_transfer_control.py`。

- `recent_performance_eligibility()`：近期表现约束；
- `cumulative_loss_guard()`：累计超额损失约束；
- `fallback_and_redistribute()`：不合格站点回退至前日负荷基准，并把退出修正量等额分配至合格站点；若无合格站点则全部回退。

## 因果信息边界

日期 `d` 的：

- 共享残差预测；
- `alpha_{j,d}`；
- `gamma_d`；
- 修正资格；

均只允许使用日期 `d` 之前的信息。日期 `d` 的真实负荷只在预测完成后进入后续状态更新。

`tests/test_information_boundary.py` 对当日及未来标签施加大幅扰动，并验证当日及此前输出不变。

## 论文正式日级统计边界

日级配对统计必须保留五折各自两站部署组边界：先在每折两站部署组内逐时聚合并形成绝对误差，再跨折汇总。该口径由 `src/evaluation.py` 和对应测试固定。仓库不保留其他跨折日级聚合口径。

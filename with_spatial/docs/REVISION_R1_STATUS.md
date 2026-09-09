# R1返修实验状态（development record）

更新时间：2026-09-09

> 本文件只记录返修开发状态。云端试跑数值不是论文最终证据；正式结果必须在用户本机基于SHA匹配的539D冻结档案复跑后冻结。

## 1. 分支与冻结边界

- V1.0稳定基线：`main@860047c0c3aa7311f1ef83f2818e2fa0c7b0028c`
- 返修开发分支：`revision/r1-202609`
- 权威冻结ZIP SHA-256：`8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`
- V1.0主结果不修改，不用返修结果覆盖 `results_reproduced/` 或 `results_reference/`。

## 2. 已完成的开发项

### R1-E1 信息边界审计

脚本：`experiments/r1_information_boundary_audit.py`

云端开发检查：因果标签扰动测试PASS；539维输入分组与12维空间特征边界一致。正式本机仍需复跑。

### R1-E2 统一后处理公平性

脚本：`experiments/r1_postprocessing_fairness.py`

云端开发预检显示：将相同站群校正和负迁移控制施加于已有逐点OOF基线后，多数基线的聚合WAPE得到改善，说明后处理机制具有一定通用贡献；本文最终方法在该开发预检中仍保持最低总体聚合WAPE。日级显著性并非对所有统一后处理基线均成立，因此返修中不得预设“显著优于所有方法”。正式数值必须以本机复跑为准。

### R1-E3 负迁移工况

脚本：`experiments/r1_negative_transfer_scenarios.py`

已形成站点×预测日工况分类，包括：两阶段均改善、站群校正修复个性化负迁移、站群校正诱发单站退化、持续负迁移，以及近期筛选通过但累计损失约束关闭的场景。历史波动等解释量采用预测日前历史，避免将当日真实负荷混入机制变量。正式数值必须以本机复跑为准。

### R1-E4 alpha统计含义

脚本：`experiments/r1_alpha_statistics.py`

云端开发预检能够以机器精度重建冻结alpha；共享先验历史收益与alpha呈强正相关，共享先验历史误差与alpha呈负相关。历史窗口受控分析仅用于稳定性解释，不用于重新选择V1.0主参数。正式数值必须以本机复跑为准。

### R1-E5 三类型稳健性

脚本：`experiments/r1_type_robustness.py`

已实现快充、慢充、混合三类的类型聚合WAPE、站点改善/退化、部署组日级Bootstrap和Newey-West统计。结论边界固定为本数据集覆盖的三类目标站，不扩大为跨数据集普适性证明。

## 3. 新增主流基线

训练入口：`experiments/r1_train_fedprox_fedper.py`

新增方法：

- FedProx；
- FedPer。

共同使用V1.0的539-128-128-96残差MLP训练预算和三个锁定随机种子。FedProx的 `mu` 仅使用每折验证站从固定网格 `{0.001, 0.01, 0.1, 1.0}` 选择。FedPer采用共享两层表示、个性化96维输出层；对未参与共享训练的目标站，仅利用预测日前30 d历史更新个性化输出层，共享表示保持固定。

云端CPU smoke test已完成：FedAvg、FedProx固定网格选择流程和FedPer目标站滚动个性化均可执行并产生正确格式输出。smoke test只有1折、1种子、2轮，不是论文证据。

正式运行前必须先检查同一新训练框架的FedAvg五折三种子校准。若 `fedavg_calibration.json` 状态为 `REVIEW`，立即停止结果解释，先排查训练框架差异。

## 4. 待完成

1. 用户本机恢复SHA匹配的539D冻结档案；
2. 本机运行V1.0完整核验；
3. 本机复跑R1-E1—E5，冻结返修统计结果；
4. 本机运行FedAvg五折三种子校准；
5. 校准通过后正式运行FedProx和FedPer；
6. 审核新增结果后再决定正文、表格、Response Letter的最终修改。

在第1—5项完成前，不修改论文实验结论。

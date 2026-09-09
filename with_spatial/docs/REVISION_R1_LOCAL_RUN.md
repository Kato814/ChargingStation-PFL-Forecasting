# R1返修实验本机运行指南

## 1. 基本原则

正式返修结果只接受以下组合：

`revision/r1-202609` 代码 + SHA匹配的539D冻结档案 + 本机完整运行输出。

GitHub Actions不承担完整科研训练，云端smoke test不进入论文。

## 2. 准备

切换到返修分支：

```bash
git checkout revision/r1-202609
```

按 `docs/RELEASE_RESTORE_CHECKLIST.md` 从 `v1.0-archive` 恢复完整大文件，并验证ZIP SHA-256：

```text
8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e
```

进入：

```bash
cd with_spatial
```

## 3. 先核验V1.0

```bash
python verify_package.py
```

若V1.0核验未通过，不启动返修新增训练。

## 4. 运行无需重新训练的返修分析

```bash
python experiments/r1_information_boundary_audit.py
python experiments/r1_postprocessing_fairness.py
python experiments/r1_negative_transfer_scenarios.py
python experiments/r1_alpha_statistics.py
python experiments/r1_type_robustness.py
```

输出统一写入：

```text
results_revision_r1/
```

这些分析应先完成并保存，再启动新增神经网络基线。

## 5. 神经网络环境smoke test

首先确认PyTorch可识别GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

随后运行：

```bash
python experiments/r1_train_fedprox_fedper.py --smoke --methods fedavg fedprox fedper --device auto
```

smoke test只运行第1折、第1个随机种子、2个共享轮次，目的仅是检查环境、数据和代码路径。

## 6. 正式新增基线实验

正式运行：

```bash
python experiments/r1_train_fedprox_fedper.py --methods fedavg fedprox fedper --device auto
```

默认协议：

- 五折全部运行；
- seeds = 20260612, 20260613, 20260614；
- 共享训练30轮；
- FedProx `mu` 网格 = 0.001, 0.01, 0.1, 1.0；
- `mu` 仅由每折验证站选择；
- FedPer目标站只使用预测日前30 d历史更新个性化输出层。

正式输出目录：

```text
results_revision_r1/neural_baselines/
```

## 7. 强制校准门槛

训练器会先用相同新框架重跑FedAvg，并生成：

```text
results_revision_r1/neural_baselines/fedavg_calibration.json
```

其中：

- `PASS`：可继续审查FedProx/FedPer正式结果；
- `REVIEW`：不得把新基线写入论文，先检查标准化、训练预算、客户端聚合、随机种子和预测后处理。

当前校准阈值为总体聚合WAPE绝对差不超过0.003；该阈值只用于判断新训练框架是否与冻结神经网络基线处于同一实现口径，不意味着新结果必须位级复现V1.0随机训练输出。

## 8. 需要回传的文件

本机运行结束后，优先提供：

```text
results_revision_r1/
```

整个目录即可。若只上传关键文件，至少包括：

```text
r1_information_boundary_check.json
r1_postprocessing_fairness_summary.csv
r1_postprocessing_fairness_daily_inference.json
r1_negative_transfer_stage_rates.csv
r1_negative_transfer_scenarios.csv
r1_negative_transfer_guard_summary.csv
r1_alpha_pooled_correlations.csv
r1_alpha_station_correlations.csv
r1_alpha_type_summary.csv
r1_alpha_history_length_summary.csv
r1_type_robustness.csv
neural_baselines/run_metadata.json
neural_baselines/fedavg_calibration.json
neural_baselines/fedprox_validation_grid.csv
neural_baselines/fedprox_selected_mu.json
neural_baselines/fedprox_summary.json
neural_baselines/fedper_summary.json
neural_baselines/new_baseline_summary.csv
```

收到正式本机结果后，再锁定哪些新增实验进入正文、哪些仅用于Response Letter。

# R1返修实验本地运行指南

## 1. 当前工作目录

仓库根目录：

`C:\Users\Administrator\ChargingStation-PFL-Forecasting`

返修分支：

`revision/r1-202609`

正式539D目录：

`C:\Users\Administrator\ChargingStation-PFL-Forecasting\with_spatial`

## 2. 前置条件

正式新增基线只允许基于：
- `revision/r1-202609`
- SHA校验通过的539D V1.0冻结档案恢复环境
- 已完成的V1.0完整复现
- 已锁定的E1—E5非训练证据

正式训练前不要删除本机恢复的大型539D缓存和OOF文件。

## 3. V1.0核验

在 `with_spatial` 下：

`python verify_package.py`

如果完整核验没有PASS，不启动新增神经基线。

## 4. E1—E5

已完成并锁定。脚本：

`python experiments/r1_information_boundary_audit.py`

`python experiments/r1_postprocessing_fairness.py`

`python experiments/r1_negative_transfer_scenarios.py`

`python experiments/r1_alpha_statistics.py`

`python experiments/r1_type_robustness.py`

这些分析不需要在每次新增神经基线运行前重复。

## 5. 新增基线

正式训练脚本：

`experiments/r1_train_fedprox_fedrep.py`

方法：
- `fedavg`：用于新框架校准
- `fedprox`：新增异质性鲁棒FL基线
- `fedrep`：新增共享表示+个性化head基线

旧开发脚本 `r1_train_fedprox_fedper.py` 不再使用，应从返修分支删除。

## 6. GPU环境检查

先检查NVIDIA驱动：

`nvidia-smi`

再确认实际用于训练的Python/PyTorch：

`python -c "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"`

正式GPU训练前必须看到：
- `torch.cuda.is_available() == True`
- 正确的NVIDIA GPU名称

如果当前Microsoft Store Python没有GPU版PyTorch，不要在该环境直接正式训练，应切换到已有CUDA PyTorch Conda环境。

## 7. smoke test

进入 `with_spatial` 后运行：

`python experiments/r1_train_fedprox_fedrep.py --smoke --methods fedavg fedprox fedrep --device auto`

smoke固定：
- 1折
- 1seed
- 2通信轮
- FedProx仍走完整mu网格以检查验证选参逻辑
- FedRep执行10 head epochs + 1 representation epoch，并检查新目标站head-only适配
- 输出标记为 `DEVELOPMENT_ONLY`

smoke结果只用于程序通路、CUDA和输出检查，不用于论文。默认输出到 `results_revision_r1/neural_baselines_smoke/`，不会覆盖正式 `neural_baselines/` 目录中的FedAvg校准。

## 8. 正式训练顺序

### 8.1 先单独做FedAvg正式校准

`python experiments/r1_train_fedprox_fedrep.py --methods fedavg --device auto`

检查：

`results_revision_r1/neural_baselines/fedavg_calibration.json`

只有：

`"status": "PASS"`

才进入下一步。

若为 `REVIEW`，停止，不运行FedProx/FedRep。

### 8.2 FedProx + FedRep正式训练

FedAvg已PASS后运行：

`python experiments/r1_train_fedprox_fedrep.py --methods fedprox fedrep --device auto`

正式设置：
- 5折
- seeds：20260612、20260613、20260614
- 30轮
- FedProx mu网格：0.001、0.01、0.1、1.0
- FedRep：每轮10个head epoch + 1个representation epoch
- FedRep目标站：滚动30 d历史、固定共享表示、head-only 10 epoch

也可以在一次命令中运行全部三种方法：

`python experiments/r1_train_fedprox_fedrep.py --methods fedavg fedprox fedrep --device auto`

但推荐“FedAvg校准单独运行 → PASS后再运行FedProx/FedRep”，避免校准失败时浪费训练时间。

## 9. 正式输出目录

`results_revision_r1/neural_baselines/`

核心文件：

运行元数据：
- `run_metadata.json`（最近一次运行）
- `run_metadata_fedavg.json`（若先单独校准FedAvg）
- `run_metadata_fedprox_fedrep.json`（若后续单独运行FedProx+FedRep）

FedAvg：
- `fedavg_calibration.json`
- `fedavg_calibration_seed_metrics.csv`

FedProx：
- `fedprox_validation_grid.csv`
- `fedprox_selected_mu.json`
- `fedprox_seed_metrics.csv`
- `fedprox_oof.csv.gz`
- `fedprox_station.csv`
- `fedprox_type.csv`
- `fedprox_summary.json`

FedRep：
- `fedrep_validation.csv`
- `fedrep_seed_metrics.csv`
- `fedrep_oof.csv.gz`
- `fedrep_station.csv`
- `fedrep_type.csv`
- `fedrep_summary.json`

汇总：
- `new_baseline_summary.csv`

## 10. FedAvg校准规则

V1.0锁定FedAvg WAPE约为：

`0.101615`

新实现正式FedAvg与其WAPE绝对差：
- `<= 0.003`：PASS
- `> 0.003`：REVIEW

校准只在完整5折×3seed×30轮条件下有效。

smoke模式不会生成正式PASS。

## 11. 中断原则

如果正式运行出现：
- CUDA错误
- NaN/Inf
- FedAvg校准REVIEW
- 站点/日期/OOF数量异常
- 输出不是42,240条
- 训练站/验证站/测试站边界异常

立即停止并保留：
- 控制台输出
- `run_metadata.json`
- 已生成结果

不要临时改参数继续跑。

## 12. 正式结果之后

先审核：
- 三seed稳定性
- FedProx各折mu选择
- FedRep验证集表现
- 42,240条OOF完整性
- WAPE/nRMSE
- 10站改善数和最坏退化
- 三类型聚合表现
- 本文方法相对新增基线的日级配对统计

确认后，再将FedProx/FedRep OOF接入R1-E2的相同固定后处理公平性分析。

不得先看测试结果再修改FedRep epoch或FedProx mu网格。

# R1返修实验状态（development record）

更新日期：2026-09-10

> 本文件记录论文一审返修补充实验的当前状态。正式论文结果仍以V1.0冻结基线和已锁定返修证据为准。

## 1. 冻结基线

- V1.0正式基线：`main@860047c0c3aa7311f1ef83f2818e2fa0c7b0028c`
- 返修分支：`revision/r1-202609`
- V1.0完整冻结档案：Private Release `v1.0-archive`
- ZIP SHA-256：`8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`
- 权威论文版本：`with_spatial/`，539D
- `no_spatial/`仅作后续研究备用，不用于本轮返修正式结果

## 2. V1.0本机复现

Windows本机已完成V1.0完整核验：
- 40站
- 539维
- 96点/日
- 42,240条OOF
- WAPE = 0.0935705022
- nRMSE = 0.0660829513
- 7/10站改善
- 最坏站点相对退化 = 1.5618%
- 正式日级配对统计复现
- 信息边界、消融、源数据manifest、原始源CSV到539D样本重建、图件输入全部PASS

跨Windows/Linux的充分统计量浮点重建采用R1维护容差 `atol=1e-9, rtol=0`；预测结果和离散状态保持机器精度一致。

## 3. E1—E5非训练返修证据

锁定提交：
- `8205e89baf9aca7078108667554cd34301d5aaf6`
- commit message：`Lock R1 non-training revision evidence`

### R1-E1 信息边界
状态：完成。

内容：
- 539维输入逐组闭合
- 相似日96维单列
- 预测日实测价格/天气特征数为0
- alpha/gamma/风险状态均仅使用预测日前历史
- 共享模型、验证选参、测试站本地历史边界审计
- 因果扰动测试PASS

### R1-E2 相同固定后处理公平性
状态：完成。

结论边界：
- 站群校正和风险控制对多种预测器具有通用作用
- 本文完整方法在统一后处理比较中仍取得最低总体聚合WAPE
- 但相对全参数微调和pFedMe的日级差异并非均达到5%显著水平
- E2不代表各基线各自重新寻优后的性能上限

### R1-E3 负迁移机制与工况
状态：完成。

内容：
- 标量候选、站群候选、最终预测的负迁移率
- 站群修复/诱发/持续负迁移机制
- 春节、普通周末、常规工作日工况
- 事后跨日变化诊断
- 日WAPE退化阈值敏感性
- 正退化日幅度诊断

解释边界：
- 风险控制是基于历史近期/累计损失的风险抑制
- 不保证每个站点每天均改善

### R1-E4 alpha统计含义
状态：完成。

主证据：
- alpha与shared-history WAPE：10/10站均负相关，站内rho中位数约-0.776
- alpha与shared-history gain：10/10站均正相关，站内rho中位数约0.943
- 两项站点级双侧Wilcoxon p均为0.001953
- 固定60 d历史池中，抽样7/14/30/60 d时alpha bootstrap标准差随样本量增加持续下降

解释边界：
- pooled 440站点日p值仅作描述，不作为独立样本推断
- alpha与shared-history gain的强相关属于机制一致性验证
- 样本量实验验证估计稳定性，不直接等价于预测精度提高

### R1-E5 类型与部署组稳健性
状态：完成。

结果边界：
- 快充、慢充、混合三类均取得正向类型聚合WAPE收益
- 5个独立部署组中4组聚合改善
- fold 5聚合WAPE轻微退化约1.12%
- fold 5的44日Bootstrap CI跨0，Newey-West p约0.113
- 不宣称所有站点、所有类型内站点或所有fold均提升

## 4. 当前唯一未完成的新增训练任务

第一审稿人要求增加主流对比方法。

最终新增方法：
- FedProx
- FedRep

此前开发阶段计划的FedPer不再作为本轮新增正式基线。

正式协议：
- `docs/REVISION_R1_EXPERIMENT_PROTOCOL.md`

训练脚本：
- `experiments/r1_train_fedprox_fedrep.py`

## 5. 新增神经基线协议状态

当前状态：**PROTOCOL_LOCKED（以本文件与训练脚本所在提交为准）**

已确定：
- 539-128-128-96残差MLP
- 30通信轮
- batch size 64
- shared lr 8e-4
- weight decay 1e-4
- seeds：20260612 / 20260613 / 20260614
- FedAvg正式校准闸门
- FedProx mu网格 `{0.001, 0.01, 0.1, 1.0}`
- FedRep共享body：fc1 + ln1 + fc2
- FedRep本地head：128 -> 96
- FedRep每轮：10 head epochs + 1 representation epoch
- 新目标站：固定共享表示，滚动30 d历史，head-only 10 epochs

尚未执行：
- GPU/PyTorch环境确认
- smoke test
- FedAvg正式校准
- FedProx正式训练
- FedRep正式训练
- 新增方法的相同固定后处理公平性扩展

## 6. 下一步

1. 将本协议锁提交并推送到 `revision/r1-202609`
2. 检查本机GPU版PyTorch
3. 运行1折×1seed×2轮smoke
4. smoke通过后先完成FedAvg正式校准
5. FedAvg校准PASS后再正式训练FedProx/FedRep
6. 再扩展R1-E2统一后处理分析
7. 最后修改论文和Response Letter

## 7. 禁止事项

- 不修改V1.0主实验结果
- 不重新选择V1.0风险参数
- 不改变五折部署组
- 不从527D版本借用结果
- 不根据测试结果调整新增基线协议
- 不把smoke结果写入论文
- 不宣称本文显著优于全部比较方法

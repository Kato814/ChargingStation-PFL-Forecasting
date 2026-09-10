# R1返修补充实验协议（2026-09-10）

## 1. 目的与冻结边界

本协议仅服务论文《基于个性化联邦学习的多类型充电站负荷预测及负迁移抑制方法》一审返修。

原论文 V1.0 科学基线保持冻结：
- `main@860047c0c3aa7311f1ef83f2818e2fa0c7b0028c`
- Private Release：`v1.0-archive`
- 权威版本：`with_spatial/`，539维输入
- V1.0 主模型、风险参数、五折协议、正式评价口径不得因返修补充实验静默改变

返修分支：
- `revision/r1-202609`
- E1—E5 非训练证据锁定提交：`8205e89baf9aca7078108667554cd34301d5aaf6`

返修新增结果统一写入：
- `with_spatial/results_revision_r1/`

本协议之后的新增神经基线属于返修补充证据，不覆盖 `results_reproduced/` 或 `results_reference/` 中的 V1.0 冻结结果。

## 2. 固定数据与评价协议

### 2.1 数据

- 40座充电站
- 原始/清洗数据：5 min
- 论文建模分辨率：15 min
- 每日96点
- 正式输入维数：539
- 输出：未来一日96点跨日残差
- 最终负荷预测：前日同刻持续性基准 + 预测残差，并进行非负截断

### 2.2 时间边界

- 训练期：2022-09-02—2022-12-15
- 验证期：2022-12-16—2023-01-15
- 测试期：2023-01-16—2023-02-28
- 测试期：44 d

### 2.3 五折目标站轮换

每折固定：
- 36座训练站
- 2座验证站
- 2座测试站

对应折的2座测试站组成一个独立部署组。测试站不参与该折共享模型训练或超参数选择。

五折全部完成后形成42,240条折外逐点预测。

### 2.4 日前信息边界

所有新增基线必须满足 R1-E1 已锁定的信息边界：
- 预测日真实负荷不得进入输入、适配或风险判定
- 相似日候选负荷仅来自 `history_date < forecast_date`
- 预测日仅允许使用日前可知的日历/节假日及静态站点属性
- 目标站本地适配仅使用预测日前已经完成的历史
- 测试站标签不得参与共享模型训练和超参数选择

## 3. E1—E5非训练证据状态

以下补充分析已完成并锁定，不再因新增基线重新调参：

- R1-E1：539维输入与训练/验证/测试信息边界审计
- R1-E2：相同固定站群校正与风险控制的公平性分析
- R1-E3：负迁移机制、运行工况、阈值敏感性与退化幅度诊断
- R1-E4：滚动标量系数 alpha 的站内统计关系与固定60 d历史池样本量稳定性
- R1-E5：快充/慢充/混合类型汇总及五折部署组稳健性

E1—E5 的正式轻量证据位于 `results_revision_r1/`，并由 `REVISION_R1_EVIDENCE_LOCK_20260910.md` 记录。

## 4. R1新增主流基线的选择

新增两个方法：

1. **FedProx**
2. **FedRep**

不再使用此前开发阶段计划的 FedPer 作为本轮新增正式基线。

### 4.1 FedProx的作用

FedProx在FedAvg本地目标中加入近端项，用于降低客户端统计异质性下的本地漂移：

`F_k(w) + (mu/2) * ||w - w_global||^2`

其作用是回答：本文相对普通FedAvg的收益是否仅来自对非IID异质性的处理不足。

### 4.2 FedRep的作用

FedRep采用“共享表示 + 客户端个性化头”结构。该方法与本文“训练站学习共享知识、未参与共享训练的目标站利用少量本地历史完成个性化”的部署协议具有较强可比性。

本研究中的结构拆分固定为：
- 共享表示：`fc1 + LayerNorm + fc2`
- 个性化头：`out: 128 -> 96`

即：
`539 -> 128 -> 128` 为共享表示，
`128 -> 96` 为个性化输出头。

## 5. 共同神经网络基线协议

除算法特有部分外，FedAvg校准、FedProx和FedRep均继承现有神经基线协议：

- 网络：539-128-128-96 残差MLP
- 第一隐藏层后使用LayerNorm
- 激活：ReLU
- dropout：0.08
- 共享训练：30通信轮
- batch size：64
- shared learning rate：8e-4
- shared weight decay：1e-4
- 目标站本地历史：30 d
- target learning rate：5e-4
- target gradient clip：5
- 固定随机种子：20260612、20260613、20260614

三个seed分别训练，最终正式OOF预测按相同 `fold-station-date-slot` 对三个seed逐点取均值后计算主要指标。

同时保存每个seed单独的WAPE、nRMSE与pooled WAPE，用于稳定性检查。不得从三个seed中挑选最优结果。

## 6. R1-E6：FedAvg校准与FedProx

### 6.1 FedAvg校准闸门

在正式FedProx/FedRep训练前，必须使用当前新增基线训练框架重新训练一次FedAvg，并与V1.0锁定FedAvg OOF结果比较。

锁定参考：
- FedAvg WAPE约为0.101615

正式校准采用：
- 五折全部运行
- 3个固定seed
- 30通信轮
- 逐点seed均值

校准判据：
- 新实现FedAvg与锁定FedAvg的WAPE绝对差 `<= 0.003`：`PASS`
- 超过该阈值：`REVIEW`

若正式校准为 `REVIEW`，禁止继续正式FedProx/FedRep训练，先检查模型结构、归一化、训练轮数、学习率、随机种子和输出定义。

smoke test中的FedAvg结果只用于代码通路检查，不作为校准证据。

### 6.2 FedProx参数

FedProx只新增近端系数：

`mu in {0.001, 0.01, 0.1, 1.0}`

每折仅利用该折2座验证站选择 `mu`。选择准则：
1. 验证集聚合WAPE最小
2. 若WAPE完全相同，取更小 `mu`

测试站不得参与 `mu` 选择。

除近端项外，FedProx与FedAvg保持相同：
- 网络结构
- 初始化方式
- 共享学习率
- weight decay
- batch size
- 30轮
- 每客户端每轮1个本地epoch
- 全部36座训练站参与每一轮

## 7. R1-E7：FedRep

### 7.1 联邦训练结构

FedRep固定：
- 共享表示：`fc1`、`ln1`、`fc2`
- 个性化头：`out`

所有36座训练站每轮参与，与当前FedAvg/FedProx基线的全参与设置保持一致。

### 7.2 每轮本地更新

每个训练客户端在每个通信轮中按以下顺序更新：

1. 固定共享表示，训练个性化头10个本地epoch
2. 固定已更新的个性化头，训练共享表示1个本地epoch
3. 仅上传共享表示参数
4. 服务器按训练样本量加权聚合共享表示
5. 个性化头保留在各训练客户端本地，不参与聚合

固定：
- `FEDREP_HEAD_EPOCHS = 10`
- `FEDREP_REPRESENTATION_EPOCHS = 1`

选择“10个head epoch + 1个representation epoch”的目的：
- 保留FedRep“多次低维head更新、较少共享表示更新”的算法特征
- 共享表示每轮更新深度仍与FedAvg/FedProx的1个本地epoch保持可比
- 不针对测试结果重新选择该比例

FedRep共享训练同样运行30通信轮。

### 7.3 未见目标站部署

验证站和测试站均视为未参与共享训练的新客户端。

预测日期 `d` 时：
1. 加载训练站学习得到的共享表示
2. 共享表示完全冻结
3. 新建目标站96输出个性化head
4. head从该seed的共同初始head状态开始
5. 仅利用日期 `d` 前最近30 d目标站历史训练head
6. head训练10个epoch
7. 预测日期 `d`
8. 日期 `d` 的真实负荷仅在预测完成后成为后续日期的历史

固定：
- `FEDREP_TARGET_HEAD_EPOCHS = 10`
- 目标站共享表示不更新
- 不使用测试日或未来真实标签

该设置服务“新目标站少量本地历史适配”的公平比较，不与本文风险控制机制混合。

## 8. 公平性边界

公平比较强调以下内容保持一致：
- 正式539维输入
- 五折站点划分
- 训练/验证/测试时间边界
- 日前可用信息边界
- 网络主体容量级别
- 固定seed
- 主要评价指标和日级统计口径
- 超参数只能使用验证站选择

不要求不同算法内部的本地更新步骤机械相同。FedRep的多次head更新属于其算法定义的一部分，应保留。

## 9. 原始基线与本文后处理严格分开

Table 2中的FedProx和FedRep结果必须是其**原始算法预测结果**：
- 不加入本文站群总量校正
- 不加入本文近期表现约束
- 不加入本文累计损失约束
- 不加入本文基准回退/修正量重分配

完成原始基线训练后，R1-E2公平性分析可进一步将“相同固定的站群校正 + 风险控制”迁移到FedProx/FedRep OOF上。

该后处理迁移：
- 不针对FedProx/FedRep重新选择 `gamma/delta/beta`
- 仅用于分离“预测器贡献”与“通用后处理贡献”
- 不替代Table 2原始基线结果

## 10. 正式输出

训练脚本：
- `experiments/r1_train_fedprox_fedrep.py`

正式结果目录：
- `results_revision_r1/neural_baselines/`

至少保存：

### 10.1 运行元数据
- `run_metadata.json`（最近一次运行）
- `run_metadata_<methods>.json`（按方法组合保留，不被后续正式运行覆盖）

包括：
- Python/PyTorch/CUDA环境
- GPU名称
- Git commit
- 539D客户端缓存SHA-256
- 锁定OOF文件SHA-256
- folds/seeds/rounds
- FedProx mu网格
- FedRep更新协议

### 10.2 FedAvg校准
- `fedavg_calibration.json`
- `fedavg_calibration_seed_metrics.csv`

### 10.3 FedProx
- `fedprox_validation_grid.csv`
- `fedprox_selected_mu.json`
- `fedprox_seed_metrics.csv`
- `fedprox_oof.csv.gz`
- `fedprox_station.csv`
- `fedprox_type.csv`
- `fedprox_summary.json`

### 10.4 FedRep
- `fedrep_validation.csv`
- `fedrep_seed_metrics.csv`
- `fedrep_oof.csv.gz`
- `fedrep_station.csv`
- `fedrep_type.csv`
- `fedrep_summary.json`

### 10.5 汇总
- `new_baseline_summary.csv`

主要报告：
- 系统聚合WAPE
- nRMSE
- pooled WAPE（证据文件保留，论文是否展示另定）
- 改善站点数
- 最大站点相对退化率
- 三类型聚合WAPE
- 本文方法相对新增基线的44日部署组配对统计

## 11. smoke test

smoke只用于检查：
- CUDA是否可用
- 三种方法是否能完成前向/反向传播
- FedProx参数网格与验证逻辑是否可运行
- FedRep“head更新→representation更新→新客户端head适配”是否可运行
- 输出文件是否能正常生成

smoke模式默认写入 `results_revision_r1/neural_baselines_smoke/`，与正式结果目录隔离。

smoke模式：
- 1折
- 1个seed
- 2通信轮
- 结果标记为 `DEVELOPMENT_ONLY`
- 不得写入论文或Response Letter

命令：

`python experiments/r1_train_fedprox_fedrep.py --smoke --methods fedavg fedprox fedrep --device auto`

## 12. 正式运行

正式命令：

`python experiments/r1_train_fedprox_fedrep.py --methods fedavg fedprox fedrep --device auto`

正式运行必须满足：
- 5折
- seeds = 20260612, 20260613, 20260614
- 30通信轮
- FedAvg校准PASS
- FedProx仅由验证站选择mu
- FedRep协议按本文件固定
- 不根据测试结果修改任何训练参数

如果FedAvg校准未PASS，训练脚本应中止后续正式FedProx/FedRep。

## 13. 禁止事项

正式结果产生前后均禁止：
- 根据测试集调整FedProx mu网格
- 根据测试集改变FedRep head/representation epoch
- 根据测试集改变FedRep目标站适配epoch
- 改变539维正式输入
- 改变五折部署组
- 改变V1.0主方法参数
- 改变风险参数以迁就新增基线
- 挑选三个seed中的最好结果作为正式结果
- 用smoke结果替代正式结果
- 因新增方法结果不理想而删除已预先锁定的方法

## 14. 证据解释边界

FedProx用于验证异质性鲁棒全局联邦优化基线。

FedRep用于验证“共享表示 + 新客户端个性化head”的PFL基线。

新增基线的作用是扩展主流算法覆盖面，不以“必须显著优于所有方法”为目标。

若本文方法相对某一新增基线的日级统计不显著，应如实报告，不得改变评价口径或后验筛选日期/站点。

本协议锁定后，下一步只允许：
1. GPU环境检查
2. smoke test
3. FedAvg正式校准
4. 校准PASS后正式FedProx/FedRep
5. 固定后处理公平性扩展
6. 结果审计与论文/Response Letter修改

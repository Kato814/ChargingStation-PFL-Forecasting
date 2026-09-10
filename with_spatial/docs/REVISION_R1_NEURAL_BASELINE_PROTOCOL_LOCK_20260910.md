# R1新增神经基线协议锁定记录（FedProx + FedRep）

日期：2026-09-10

状态：`PROTOCOL_LOCKED`（本文件与训练脚本提交到 `revision/r1-202609` 后生效）

## 1. 锁定目的

用于回应第一审稿人关于“对比算法较少，建议增加主流方法”的意见。

本锁定记录建立在：
- V1.0冻结基线
- E1—E5非训练返修证据
- `revision/r1-202609@8205e89` 之后

新增基线不得改变原论文主方法或原有冻结结果。

## 2. 最终新增方法

- FedProx
- FedRep

弃用开发阶段候选FedPer，不进入本轮新增正式基线。

## 3. 选择依据

### FedProx
用于覆盖面向异质客户端的全局联邦优化基线，与FedAvg形成直接异质性对照。

### FedRep
用于覆盖“共享表示 + 个性化head”的PFL路线，并适合评估训练阶段未参与共享建模的新目标客户端。

## 4. 公平比较共同条件

- 输入：539D
- 输出：96点跨日残差
- 网络主体：539-128-128-96残差MLP
- 五折：36 train + 2 validation + 2 test
- 时间边界与V1.0一致
- 测试站不参与共享训练
- 预测日不使用当前日/未来标签
- seeds：20260612、20260613、20260614
- 正式结果：三个seed逐点均值
- 共享训练：30轮
- batch size：64
- shared lr：8e-4
- shared weight decay：1e-4

## 5. FedAvg校准闸门

新增训练框架必须先复现锁定FedAvg到可接受范围。

- 锁定FedAvg WAPE约0.101615
- 正式新实现与锁定值绝对差 `<=0.003`：PASS
- 否则：REVIEW并停止正式新增方法训练

## 6. FedProx

- 每客户端每轮1本地epoch
- 近端系数：`mu in {0.001, 0.01, 0.1, 1.0}`
- 每折仅由2座验证站选择mu
- 测试站不参与选择

## 7. FedRep

共享表示：
- fc1
- ln1
- fc2

个性化head：
- out: 128 -> 96

每轮每客户端：
- 10 head epochs（共享表示参数冻结，并关闭冻结表示路径中的dropout随机性）
- 1 representation epoch
- 仅聚合representation
- head保留本地

目标站：
- 作为未参与共享训练的新客户端
- representation固定
- 预测日前滚动30 d历史
- 新目标head从该seed共同初始head状态开始
- head-only训练10 epochs；共享表示保持frozen/eval状态
- 当日真实值仅在预测完成后进入后续历史

## 8. 原始基线与本文后处理分离

FedProx/FedRep主表结果不得加入：
- 站群总量校正
- 近期表现约束
- 累计损失约束
- 基准回退
- 修正量重分配

原始OOF完成后，才允许在R1-E2中施加与现有方法完全相同的固定后处理，用于贡献分解。

## 9. 禁止后验修改

正式训练开始后不得因测试结果：
- 改FedRep 10+1更新比例
- 改FedRep目标head 10 epoch
- 改FedProx mu网格
- 改seed
- 改五折
- 改训练/验证/测试时间
- 改539D输入
- 改V1.0风险参数

## 10. 代码

正式脚本：

`experiments/r1_train_fedprox_fedrep.py`

旧开发脚本：

`experiments/r1_train_fedprox_fedper.py`

应在本次protocol-lock提交中删除，避免误运行。

## 11. 下一状态

本文件与训练脚本进入 `revision/r1-202609` 后，即视为协议锁定；后续只记录该提交SHA，不再修改已锁定算法/超参数。随后才执行GPU环境确认和smoke test。

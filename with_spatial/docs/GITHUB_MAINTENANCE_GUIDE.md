# GitHub维护指南（539维权威版本）

本文件规定 `ChargingStation-PFL-Forecasting` 仓库中 `with_spatial/` 的长期维护方式。该目录是论文原最终实验的权威代码记录。

## 1. 分支策略

- `main`：稳定记录，只接受已经核验的代码、文档和结果元数据；
- `import-v1.0`：V1.0一次性整理/导入分支。PR #1合并后视为历史导入分支，不再作为后续开发主线；
- 后续方法修改从 `main` 新建独立分支，例如：
  - `dev/alpha-adaptation-v2`
  - `dev/group-correction-weighted`
  - `dev/negative-transfer-v2`
  - `revision/r1-202609`

不得直接在V1.0稳定记录上覆盖科学结果。

## 2. 什么修改必须升版本

以下任一变化均视为科学版本变化：

- 输入特征数量、定义或顺序；
- 联邦残差模型形式或正则化；
- 全局/类型先验组合规则；
- 目标站标量适配公式；
- 站群总量校正公式或差额分配规则；
- 近期表现/累计损失判定逻辑；
- 基准回退或修正量重分配逻辑；
- 折划分、时间区间或验证参数；
- 论文主评价口径。

此类修改必须：新分支 → 新结果 → 回归测试 → Changelog → 新版本/标签。

## 3. 可不升科学版本的维护修改

- 注释和文档补充；
- 不改变输出的代码整理；
- 浮点比较从位级相等改为合理容差；
- 路径、CLI帮助、日志增强；
- 只改变绘图外观且不改变绘图数据；
- Release链接、归档状态和校验信息更新。

即使不升科学版本，也应通过现有测试并记录重要维护变更。

## 4. 每次提交前最低检查

在 `with_spatial/` 目录下执行：

```bash
python -m py_compile src/*.py
python tests/test_information_boundary.py
python tests/test_paired_daily_statistics.py
```

涉及主链或输入时，应恢复冻结大文件后运行：

```bash
python verify_package.py
```

普通GitHub CI只执行不依赖大型归档文件的语法和元数据检查；完整科学回归仍以本地/恢复后的冻结档案验证为准。

## 5. 结果管理

- `docs/RESULTS_BASELINE.md`：V1.0不可覆盖的论文结果基线；
- `results_reference/`：冻结证据的轻量摘要；
- `results_reproduced/`：当前清洁代码重新计算的轻量结果；
- 大型OOF、pkl和完整5 min数据由匹配SHA-256的冻结ZIP恢复。

不得把 `no_spatial/` 结果复制为 `with_spatial/` 的结果。

## 6. Pull Request检查项

涉及科学代码的PR至少回答：

1. 改了哪一个论文模块？
2. 是否改变输入或信息边界？
3. 是否改变折级参数？
4. 是否改变V1.0锁定指标？
5. 运行了哪些测试？
6. 是否需要新版本号和新Release？

文档/维护型PR也应说明是否触及冻结数值、SHA、数据路径或恢复规则。

## 7. 冻结包与Private Release

权威V1.0归档：

`ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`

SHA-256：

`8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`

Private Release：

`https://github.com/Kato814/ChargingStation-PFL-Forecasting/releases/tag/v1.0-archive`

截至2026-09-09，完整ZIP和 `.sha256` 文件已经上传，GitHub记录的ZIP asset digest与上述SHA-256一致。完整恢复步骤见 `RELEASE_RESTORE_CHECKLIST.md`。

后续返修或新实验需要完整数据/缓存时，应从该Release恢复V1.0档案，再在新的revision/dev分支中工作，不得修改V1.0冻结ZIP本身。

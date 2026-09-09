# GitHub导入与长期维护审计

## 当前定位

GitHub仓库保存可维护、可审阅的论文代码和轻量证据；Private Release保存完整冻结ZIP及校验文件。两者职责不同：

- Git仓库：后续阅读、修改、代码审查、差异追踪；
- Private Release `v1.0-archive`：V1.0完整二进制恢复与完整数据证据。

`with_spatial/` 539维精确复现版是本仓库的权威论文结果记录；`no_spatial/` 仅作为后续研究版本独立保留。

## 已导入并核验的539维维护内容

- 论文主方法全部核心模块；
- 源CSV→539维特征构建模块；
- 五折端到端主入口及折级参数记录；
- 表2、表3、表4复算脚本与表5站点结果；
- 五折独立两站部署组边界下的正式日级配对统计；
- 对比方法协议、结果来源层级与锁定结果适配；
- 因果信息边界、回归等价、充分统计量、消融、基线协议和图件输入等测试；
- 图5—图9绘图脚本与结果来源说明；
- 风险参数敏感性冻结轻量证据；
- 冻结结果、类型结果、站点结果、消融结果等轻量表；
- 40站划分、数据清洗来源和源数据文件级SHA-256 manifest；
- 539维特征分组、12维空间输入位置及完整schema导出工具；
- 版本、数据、特征、折划分、环境、恢复、归档和后续修改政策文档；
- GitHub Actions轻量元数据/语法检查，最终审查时状态为PASS。

## 最终仓库审查结论

最终审查重点检查目录结构、路径引用、重复记录、过时说明、归档边界和CI状态。结论如下：

- 未发现需要删除的重复核心实现；`with_spatial/` 与 `no_spatial/` 的相似文件属于有意的版本隔离，不合并；
- `SOURCE_ARCHIVE.txt`、`archive/` 和详细归档文档分别承担快速识别、机器校验和人工说明职责，属于有意分层，不视为冗余；
- 源数据外层manifest与冻结数据目录内manifest用途不同，已在大文件与恢复文档中区分；
- 论文图5—图9、表1—表5和对应代码/数据路径已由 `PAPER_ARTIFACT_INDEX.md` 统一索引；
- `main` 已成为稳定V1.0维护基线，`import-v1.0` 仅保留为历史导入分支。

## Git中有意不保存的内容

由于文件体积和Git历史可维护性，普通Git树不保存：

- 40座站点完整5 min源CSV；
- `clients_with_spatial.pkl`；
- `federated_statistics_with_spatial.pkl`；
- 完整锁定与重建OOF `.csv.gz`；
- 冻结ZIP。

这些文件的路径、大小、SHA-256和恢复流程记录在 `../../docs/LARGE_FILES.md`、`ARCHIVE_RECORD.md` 与 `RELEASE_RESTORE_CHECKLIST.md`。

## 冻结二进制权威性

权威完整归档：

`ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`

SHA-256：

`8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`

Private Release：

`https://github.com/Kato814/ChargingStation-PFL-Forecasting/releases/tag/v1.0-archive`

GitHub记录的ZIP asset大小为145,516,950 bytes，asset digest与上述SHA-256完全一致。Release同时保存对应 `.sha256` 文件。

Git可对验证脚本、注释、文档和维护结构进行非科学性增强，但不得静默改变V1.0算法和冻结结果。任何科学变化必须新建分支和版本。

## 浮点兼容性增强

原冻结档案在制作环境中可得到逐元素零差异。长期跨CPU/BLAS复现检查发现，充分统计量可能出现约 `4.55e-13` 的机器精度舍入差异。因此Git维护版将该测试固定为 `rtol=0, atol=1e-10`。这是验证鲁棒性修改，不改变模型计算、预测值或论文结果。

## 远程归档状态

截至2026-09-09，原Issue #3所跟踪的远程归档任务已经完成：

- Private Release `v1.0-archive` 已发布；
- 完整WITH_SPATIAL V1.0 ZIP已上传；
- `.sha256` 配套文件已上传；
- GitHub asset digest已核对并与冻结SHA-256一致；
- Release位置已回填至恢复和大文件文档。

因此，当前仓库的539维V1.0长期归档链已经闭环：普通Git用于代码与轻量证据维护，Private Release用于完整二进制恢复。

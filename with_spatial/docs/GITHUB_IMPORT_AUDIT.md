# GitHub导入与长期维护审计

## 当前定位

GitHub仓库保存可维护、可审阅的论文代码和轻量证据；冻结ZIP保存完整二进制数据、缓存与逐点结果。两者职责不同：

- Git仓库：后续阅读、修改、代码审查、差异追踪；
- V1.0冻结ZIP：历史精确恢复与完整数据证据。

当前仓库维护优先级是 `with_spatial/` 539维精确复现版。

## 已导入的539维维护内容

- 论文主方法全部核心模块；
- 源CSV→539维特征构建模块；
- 五折端到端主入口；
- 表2/表3/表4复算脚本；
- 部署组日级配对统计；
- 对比方法协议与锁定结果适配；
- 因果信息边界、回归等价、充分统计量、消融、图件输入等测试；
- 图6、图8、图9绘图脚本；
- 冻结结果、类型结果、站点结果、消融结果等轻量表；
- 源数据文件级SHA-256 manifest；
- 版本、数据、特征、折划分、环境、恢复与修改政策文档。

## Git中有意不保存的内容

由于文件体积和Git历史可维护性，普通Git树不保存：

- 40座站点完整5 min源CSV；
- `clients_with_spatial.pkl`；
- `federated_statistics_with_spatial.pkl`；
- 完整锁定与重建OOF `.csv.gz`；
- 冻结ZIP。

这些文件的路径、大小、SHA-256和恢复流程已记录在 `../docs/LARGE_FILES.md`、`ARCHIVE_RECORD.md` 与 `RELEASE_RESTORE_CHECKLIST.md`。

## 冻结二进制权威性

V1.0历史精确恢复时，以以下ZIP为二进制权威记录：

`ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`

SHA-256:

`8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`

Git分支可以对验证脚本、注释、文档和维护结构进行非科学性增强，但不得静默改变V1.0算法和冻结结果。任何科学变化必须新建版本。

## 浮点兼容性增强

原冻结档案在制作环境中可得到逐元素零差异。长期跨CPU/BLAS复现检查发现，充分统计量可能出现约 `4.55e-13` 的机器精度舍入差异。因此Git维护版将该测试改为 `rtol=0, atol=1e-10`。这是验证鲁棒性修改，不改变模型计算、预测值或论文结果。

## 待人工完成的唯一大文件归档步骤

GitHub连接当前不提供Release附件上传能力。仓库Issue #2跟踪：

- 创建/使用私有Release `v1.0-archive`；
- 上传WITH_SPATIAL V1.0 ZIP；
- 下载后再次核验SHA-256；
- 将Release位置回填至恢复文档。

在Issue #2完成前，Git仓库可以正常用于代码管理，但不能单靠普通Git clone恢复完整大文件。

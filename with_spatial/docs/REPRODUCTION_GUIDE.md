# 复现指南

## 环境

建议优先使用 `requirements-lock.txt` 中的冻结依赖；跨环境浮点比较规则见 `docs/ENVIRONMENT_AND_NUMERICS.md`。

```bash
pip install -r requirements-lock.txt
```

## 0. 恢复冻结大文件

普通Git历史不保存完整源数据、客户端pkl、联邦统计量pkl和锁定逐点OOF证据。请先从SHA-256匹配的 `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip` 恢复，详见 `docs/RELEASE_RESTORE_CHECKLIST.md`。

## 1. 输入与联邦统计量

快速验证冻结539维缓存并重建联邦统计量：

```bash
python prepare_inputs.py --output-statistics /tmp/federated_statistics_rebuilt.pkl
```

从冻结源CSV完整重建539维模型输入，并与冻结缓存核验：

```bash
python prepare_inputs.py --rebuild-cache-from-raw
```

完整重建明显慢于直接读取冻结缓存，日常复现无需重复执行。

如需导出完整539维特征顺序：

```bash
python tools/export_feature_schema.py
```

## 2. 复现论文主方法

```bash
python run_paper_pipeline.py
```

输出：

- `results_reproduced/oof_predictions.csv.gz`
- `results_reproduced/summary.json`
- `results_reproduced/station_results.csv`
- `results_reproduced/type_aggregate_results.csv`
- `results_reproduced/paired_daily_statistics.json`

## 3. 复现论文表2与配对统计

```bash
python experiments/reproduce_table2_baselines.py
python experiments/reproduce_comparison_daily_inference.py
```

神经网络对比方法从锁定逐点OOF预测复算；全参数微调、本地MLP和pFedMe微调对应30 d目标站历史窗口。本地滚动岭回归采用60 d窗口，当前仅有锁定汇总结果。详见 `docs/BASELINE_ALIGNMENT_AUDIT.md`。

## 4. 复现论文消融

```bash
python experiments/reproduce_table3_shared_prior_ablation.py
python experiments/reproduce_table4_ablation.py
```

## 5. 生成论文结果图

```bash
python figures/plot_fig6_scalar_coefficients.py
python figures/plot_fig7_risk_sensitivity.py
python figures/plot_fig8_type_improvement.py
python figures/plot_fig9_typical_curves.py --start-date 2023-02-15
```

其中图7使用 `results_reference/` 中冻结的风险参数敏感性证据切片和5×5网格，不在本入口中重新搜索25个参数组合。各图来源见 `docs/FIGURE_REPRODUCTION.md`。

## 6. 一键复现

```bash
python reproduce_all.py
```

如需在复现完成后继续完整核验：

```bash
python reproduce_all.py --verify
```

## 7. 完整核验

```bash
python verify_package.py
```

完整核验包括：

- Python语法编译；
- 539维输入契约；
- 充分统计量重建一致性（跨BLAS使用 `rtol=0, atol=1e-10`）；
- 五折端到端主结果；
- 与冻结逐点预测的数值等价；
- 两站部署组边界下的日级配对统计；
- 因果信息边界；
- 表2对比结果与神经网络对比方法的部署组日级配对统计；
- 表3/表4消融结果；
- 已废弃评价口径未重新混入论文代码；
- 源站点数据manifest；
- 源CSV到539维输入样本等价；
- 图6/图7/图8/图9输入契约与绘图脚本。

## 8. 更新发布清单

在确认所有测试通过后：

```bash
python make_release.py
```

重新生成 `MANIFEST.csv`、`SHA256SUMS.txt` 与 `PACKAGE_SUMMARY.json`，用于版本冻结和差异追踪。

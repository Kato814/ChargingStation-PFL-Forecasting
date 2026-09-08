# V1.0冻结档案恢复检查清单

本仓库普通Git历史不保存完整5 min数据、两类大型pkl缓存和冻结ZIP。需要完整复现时，按以下步骤恢复。

## 1. 获取冻结包

文件：`ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`

期望 SHA-256：

`8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`

如果从GitHub Release或其他备份位置下载，首先核验SHA-256，不匹配时不得用于复现。

## 2. 恢复大文件

至少恢复：

- `data/source_station_dataset/`（完整40站源CSV）
- `data/clients_with_spatial.pkl`
- `data/federated_statistics_with_spatial.pkl`
- `results_reference/oof_predictions_source.csv.gz`

若希望执行回归测试，还应恢复：

- `results_reproduced/oof_predictions.csv.gz`（可由主流程重新生成）

## 3. 验证源数据

```bash
python tests/test_source_dataset_manifest.py
```

源数据manifest记录56个文件，总大小217,219,808 bytes。

## 4. 验证源CSV到539维输入

```bash
python tests/test_raw_feature_builder_sample.py
```

完整重建可使用：

```bash
python prepare_inputs.py --rebuild-cache-from-raw
```

默认不覆盖冻结缓存。

## 5. 运行论文主链

```bash
python run_paper_pipeline.py
```

应生成42,240条五折OOF逐点结果，并达到 `docs/RESULTS_BASELINE.md` 中的冻结指标。

## 6. 完整核验

```bash
python verify_package.py
```

浮点充分统计量跨环境比较使用 `rtol=0, atol=1e-10`；离散字段、站点/日期键和资格状态必须完全一致。

# ChargingStation PFL Paper Code — No-Spatial Variant

This package is the paper-aligned **non-spatial input variant** of the day-ahead charging-station forecasting method. It is intentionally separate from the 539-dimensional exact-reproduction archive.

## Scope

The end-to-end method implements:

1. previous-day load baseline and inter-day residual construction;
2. closed-form federated multi-output ridge residual models (global and type);
3. shared-prior combination and target-station scalar adaptation;
4. station-group total correction;
5. negative-transfer control with recent-performance and cumulative-loss checks;
6. five-fold out-of-fold evaluation using the paper deployment-group boundary.

The model input contains **527 features** and excludes coordinates, POI descriptors and location-derived zone attributes.

## Relationship to the exact-reproduction variant

- `../with_spatial/` is the frozen 539-D archive for exact reproduction of the reported final experiment.
- This 527-D variant uses the same fold-specific hyperparameter values as the frozen experiment so that the effect of removing location-derived inputs can be isolated.
- The original validation-search provenance needed to re-select those fold parameters is not available; this variant does **not** claim post-removal retuning.
- Neural Table-2 baseline predictions from the 539-D archive are intentionally not reused as matched-input results.

## Quick start

After restoring the matching frozen archive data:

```bash
python prepare_inputs.py
python run_paper_pipeline.py
python experiments/reproduce_table3_shared_prior_ablation.py
python experiments/reproduce_table4_ablation.py
python reproduce_all.py
python verify_package.py
```

## Large files

The complete 527-D caches, sufficient statistics and full station source files are stored in `ChargingStation_PFL_Paper_Code_NO_SPATIAL_V1.0.zip`, not normal Git history. See `../docs/LARGE_FILES.md`.

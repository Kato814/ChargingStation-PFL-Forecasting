# ChargingStation-PFL-Forecasting

Code and reproducibility records for the paper **“基于个性化联邦学习的多类型充电站负荷预测及负迁移抑制方法”**.

This repository contains two intentionally separated variants of the day-ahead charging-station forecasting codebase:

- `with_spatial/` — frozen 539-D exact-reproduction variant. It preserves the spatial inputs used by the final audited experiment and is the authoritative archive for reproducing the reported frozen result set.
- `no_spatial/` — 527-D paper-input-aligned research variant. It removes coordinates, POI descriptors, and location-derived zone features. It is intended for subsequent research and does not overwrite the frozen 539-D experiment.

The method pipeline is organized around the paper structure: federated residual modeling, shared-prior combination, target-station scalar adaptation, station-group total correction, negative-transfer control, and five-fold out-of-fold evaluation.

## Repository policy

The two variants must not share caches, sufficient statistics, reproduced outputs, or result claims. Large binary caches and full source datasets are kept out of normal Git history and are tracked separately by archive/checksum records. The Git repository stores maintainable source code, configuration, tests, documentation, schemas, lightweight results, and figure-generation scripts.

## Paper-result archive

The exact-reproduction archive is `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0`.

The non-spatial research archive is `ChargingStation_PFL_Paper_Code_NO_SPATIAL_V1.0`.

See `docs/VERSION_DIFFERENCES.md` and each variant's own `README.md` before modifying the code.

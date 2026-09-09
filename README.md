# ChargingStation-PFL-Forecasting

Code and reproducibility records for the paper **“基于个性化联邦学习的多类型充电站负荷预测及负迁移抑制方法”**.

## Repository priority

The repository contains two intentionally isolated day-ahead variants, but the current maintenance priority is the **539-D `with_spatial/` exact-reproduction record** because it matches the final audited experiment used to establish the frozen paper-result archive.

- `with_spatial/` — **authoritative 539-D exact-reproduction variant**. It preserves the coordinates, POI descriptors and location-derived static features present in the final audited experiment. Use this variant when reproducing or auditing the historical final result set.
- `no_spatial/` — secondary 527-D research variant. It removes the 12 spatial inputs and is retained for later paper-input-aligned research. It must not overwrite or redefine the 539-D record.

The maintainable method path follows the paper structure: federated residual modeling, shared-prior combination, target-station scalar adaptation, station-group total correction, negative-transfer control, and five-fold out-of-fold evaluation.

## Authoritative frozen archive

`ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`

SHA-256:

`8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`

The earlier final optimized source archive has SHA-256:

`ec5618f9a979049c57659c51716929514a6897d829652ba626e8b10a1e1be8ed`

## Repository policy

- Do not mix caches, sufficient statistics, reproduced outputs or result claims between the two variants.
- Do not commit the full 5 min source station payload, large pickle caches or frozen ZIP archives into ordinary Git history.
- Keep maintainable source code, configuration, tests, documentation, manifests/checksums, lightweight results and figure-generation scripts in Git.
- Store complete frozen archives separately (recommended: a private GitHub Release) and verify them by SHA-256 before restoration.
- The official daily paired evaluation preserves the five independent two-station deployment-group boundaries; no retired alternative daily aggregation metric belongs in the paper-code record.
- Scientific changes must use a new branch/version. Do not silently overwrite the V1.0 exact-reproduction record.

## Start here

For paper/result reproduction, read in this order:

1. `with_spatial/README.md`
2. `with_spatial/docs/ARCHIVE_RECORD.md`
3. `with_spatial/docs/CODE_TO_PAPER_MAPPING.md`
4. `with_spatial/docs/DATA_CONTRACT.md`
5. `with_spatial/docs/RESULTS_BASELINE.md`
6. `with_spatial/docs/ENVIRONMENT_AND_NUMERICS.md`

See `docs/VERSION_DIFFERENCES.md` and `docs/ARCHIVE_POLICY.md` for repository-wide version policy.

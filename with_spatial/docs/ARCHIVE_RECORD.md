# 539-D exact-reproduction archive record

## Role

`with_spatial/` is the authoritative maintainable source-code record for the final audited paper experiment. It preserves the original 539-dimensional input contract, including the spatial descriptors actually used by that experiment. It is the variant to use when historical reproduction of the reported final result set is required.

The corresponding frozen binary archive is:

- `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`
- SHA-256: `8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`
- Private Release: `https://github.com/Kato814/ChargingStation-PFL-Forecasting/releases/tag/v1.0-archive`
- Release asset size: 145,516,950 bytes
- GitHub asset digest: `sha256:8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`

The Release also stores `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip.sha256`.

The earlier final optimized source package from which the clean paper-code archive was reconstructed is:

- `ThreeType_ChargingLoadForecasting_FINAL_OPTIMIZED_20260715(1).zip`
- SHA-256: `ec5618f9a979049c57659c51716929514a6897d829652ba626e8b10a1e1be8ed`

## Frozen scientific scope

- Day-ahead charging-station load forecasting only; no 4 h intraday rolling module.
- 40 charging stations.
- 15 min resolution, 96 output points per day.
- 539-dimensional model input.
- Five folds; each fold contains two held-out target/test stations forming one deployment group.
- Training period: 2022-09-02 to 2022-12-15.
- Validation period used by the original study: 2022-12-16 to 2023-01-15.
- Test period: 2023-01-16 to 2023-02-28.
- 42,240 out-of-fold point forecasts in the locked evaluation table.

## Frozen method chain

1. previous-day same-time baseline and inter-day residual target;
2. federated sufficient-statistics aggregation;
3. closed-form global/type multi-output ridge residual models;
4. global/type shared-prior combination;
5. target-station rolling scalar adaptation;
6. station-group total correction;
7. recent-performance and cumulative-loss negative-transfer control;
8. baseline fallback and correction redistribution;
9. five-fold out-of-fold evaluation.

## Frozen result checks

The maintainable code must reproduce, within documented numerical tolerance:

- system WAPE: `0.09357050219189945`;
- nRMSE: `0.06608295125576832`;
- pooled WAPE: `0.21098808555609647`;
- improved target stations: `7/10`;
- maximum station relative degradation: `0.015618088958722623`;
- final correction enable rate: approximately `64.3182%`;
- deployment-group daily paired mean WAPE difference: `-0.011563263932537487`.

The paper daily paired statistic preserves the five independent two-station deployment-group boundaries. No alternative cross-fold daily aggregation metric is part of this repository's paper evaluation contract.

## Large-file policy

The normal Git tree intentionally excludes the full 5 min station CSV payload, `clients_with_spatial.pkl`, `federated_statistics_with_spatial.pkl`, and the frozen ZIP itself. Restore those files from the SHA-matched Private Release before running full end-to-end reproduction. Lightweight manifests, schemas, checksums, code, tests, result summaries and figure scripts belong in Git.

The combination of `main` plus the `v1.0-archive` Private Release is the closed long-term V1.0 record.

## Change policy

Do not alter the V1.0 scientific result record in place. Scientific changes must be made on a new branch and recorded in `CHANGELOG.md`. If a change alters features, fold parameters, baselines, eligibility rules, allocation rules or results, create a new version rather than silently replacing the V1.0 record.

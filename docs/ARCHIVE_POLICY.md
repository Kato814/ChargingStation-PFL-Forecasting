# Long-term archive policy

This repository is the maintainable source-code record. The complete frozen V1.0 ZIP archives remain the authoritative binary records for exact restoration.

## Integrity

Keep each frozen ZIP together with its SHA-256 file. Verify checksums before using a restored archive.

For the authoritative 539-D paper record, the remote binary archive is the private GitHub Release:

`https://github.com/Kato814/ChargingStation-PFL-Forecasting/releases/tag/v1.0-archive`

The Release contains `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip` and its `.sha256` sidecar. The ZIP asset digest recorded by GitHub matches the frozen SHA-256 maintained in the repository.

## Numerical environment

The archived requirements lock NumPy, pandas, SciPy and Matplotlib. Small floating-point differences at approximately machine precision may appear across Python/BLAS/CPU environments. Numerical regression tests should use an explicit tolerance rather than require bitwise identity unless the test is specifically checking byte-level archive integrity.

## Variant separation

- `with_spatial`: 539-D exact-reproduction record.
- `no_spatial`: 527-D paper-input-aligned research variant.

Never copy model-input caches, federated sufficient statistics or reproduced OOF results between variants.

## Large-file restoration

The normal Git tree excludes `.pkl` caches and full 5-min station CSVs. Restore the authoritative 539-D files from the SHA-matched `v1.0-archive` private Release before full reproduction. See `LARGE_FILES.md`.

## Future development

Create a branch for method changes. Keep the frozen `v1.0` result records and Release assets unchanged and record scientific changes in the repository changelog before creating a new tagged release.

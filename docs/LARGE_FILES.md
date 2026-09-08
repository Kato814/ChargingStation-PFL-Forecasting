# Large files and frozen archives

Normal Git history intentionally excludes the large binary caches, full 5-min station files and frozen ZIP archives.

## Frozen V1.0 archives

- `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`
  - SHA-256: `8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`
- `ChargingStation_PFL_Paper_Code_NO_SPATIAL_V1.0.zip`
  - SHA-256: `7d01668b045ca0c34dc592788d42c79da520d1e721334a85d36cc3a578115014`

The frozen ZIPs contain the complete source-station dataset, model-input caches, federated sufficient statistics, reference outputs and verification records. They should be attached to a GitHub Release (recommended tag: `v1.0-archive`) rather than committed directly to Git history.

## Files intentionally excluded from normal Git history

- `data/*.pkl`
- `data/source_station_dataset/**/station_processed_5min/*.csv`
- frozen `.zip` archives

The repository keeps schemas, manifests, checksums, lightweight source tables, code, tests, result summaries and figure scripts so that code modifications remain reviewable.

When restoring a complete variant, unpack the matching frozen archive or restore the excluded files to their documented paths before running full reproduction.

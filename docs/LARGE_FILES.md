# Large files and frozen archives

Normal Git history intentionally excludes large binary caches, full 5-min station files and frozen ZIP archives.

## Frozen V1.0 archives

### Authoritative paper-result archive

- `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip`
  - SHA-256: `8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`
  - Private Release: `https://github.com/Kato814/ChargingStation-PFL-Forecasting/releases/tag/v1.0-archive`
  - Release asset size: 145,516,950 bytes
  - GitHub asset digest: `sha256:8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`

The same Release also contains `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip.sha256`.

### Secondary research archive

- `ChargingStation_PFL_Paper_Code_NO_SPATIAL_V1.0.zip`
  - SHA-256: `7d01668b045ca0c34dc592788d42c79da520d1e721334a85d36cc3a578115014`

The authoritative 539-D frozen ZIP contains the complete source-station dataset, model-input cache, federated sufficient statistics, reference outputs and verification records. It is intentionally stored as a private Release asset rather than in ordinary Git history. The secondary no-spatial archive remains separately managed.

## WITH_SPATIAL large-file checksums

The following files are restored from the authoritative 539-D archive:

| Path | Size (bytes) | SHA-256 |
|---|---:|---|
| `with_spatial/data/clients_with_spatial.pkl` | 47,746,837 | `daf125ac4ced7f0e20e95ec07b133cd75623678d5b8365fad587185866b02b1c` |
| `with_spatial/data/federated_statistics_with_spatial.pkl` | 109,736,128 | `ebdd1713007a31c661e425663d40a15a64bfb2a5460ac38c6d7865b3f6cadba9` |
| `with_spatial/results_reference/oof_predictions_source.csv.gz` | 4,002,934 | `abb8d28757bc40686990ec8cb3a7e35de449984b22c9c64c21ae2476779fbc67` |
| `with_spatial/results_reproduced/oof_predictions.csv.gz` | 2,174,519 | `70525383ac6359a7c16d071a675141438e771432d4ce063338afb820a7a7bfd1` |

The complete `source_station_dataset/` directory contains **56 files totaling 217,219,808 bytes**, including `dataset_manifest.csv` itself. Because a manifest cannot hash itself, `dataset_manifest.csv` contains **55 payload entries totaling 217,213,748 bytes**; the manifest file is 6,060 bytes. File-level payload checksums are stored in `with_spatial/data/source_station_dataset/dataset_manifest.csv`, with the summary in `with_spatial/data/source_station_dataset_summary.json`.

## Files intentionally excluded from normal Git history

- `data/*.pkl`
- `data/source_station_dataset/**/station_processed_5min/*.csv`
- large daily-source CSV payloads when they are already covered by the frozen archive manifest
- locked/reproduced full OOF `.csv.gz` payloads
- frozen `.zip` archives

The repository keeps schemas/feature documentation, manifests, checksums, code, tests, lightweight result summaries and figure-generation scripts so modifications remain reviewable.

## Restoration

For full reproduction, download the authoritative ZIP from the private `v1.0-archive` Release, verify its SHA-256, and restore the excluded files to their documented paths. Follow `with_spatial/docs/RELEASE_RESTORE_CHECKLIST.md`.

The authoritative 539-D remote archive task previously tracked by Issue #3 is complete once this documentation update is merged.

# Changelog

## 2026-09-09 — authoritative V1.0 archive hosting completed

- Published private GitHub Release `v1.0-archive` for the authoritative `with_spatial` 539-D frozen record.
- Uploaded `ChargingStation_PFL_Paper_Code_WITH_SPATIAL_V1.0.zip` and its `.sha256` sidecar.
- Verified the GitHub ZIP asset digest matches the frozen SHA-256 `8e053ffea7ed6b1274517a5a515d6baef0e14fe7c45230c8ae608cec75b3cc9e`.
- Updated repository restoration and maintenance documentation to use the Release as the remote binary archive.
- No scientific code, input definition, parameter, result, or evaluation contract changed.

## 2026-09-09 — V1.0 repository audit and import completion

- Completed the paper-oriented import and final repository audit for the authoritative `with_spatial` 539-D record.
- Confirmed the cleaned main path, paper/result mapping, source-data manifests, frozen checksums, baseline/ablation records, figure provenance, and lightweight CI metadata checks.
- Kept `no_spatial` as a separate secondary research variant; its caches, sufficient statistics and result claims remain isolated from `with_spatial`.
- Merged the authoritative V1.0 source/documentation record to `main`.

## 2026-09-08 — repository initialization

- Created a dedicated repository for the paper code.
- Established two isolated variants: `with_spatial` and `no_spatial`.
- Defined the large-file archive policy and frozen V1.0 checksums.
- Imported the cleaned paper-oriented codebase, documentation and lightweight reproducibility records.

The historical Stage/patch development tree is intentionally not part of this repository's maintainable primary code path.

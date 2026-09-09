# Changelog

## 2026-09-09 — V1.0 repository audit and import completion

- Completed the paper-oriented import and final repository audit for the authoritative `with_spatial` 539-D record.
- Confirmed the cleaned main path, paper/result mapping, source-data manifests, frozen checksums, baseline/ablation records, figure provenance, and lightweight CI metadata checks.
- Kept `no_spatial` as a separate secondary research variant; its caches, sufficient statistics and result claims remain isolated from `with_spatial`.
- Deferred only the private GitHub Release attachment of the full frozen 539-D ZIP to Issue #3; this hosting step does not alter V1.0 scientific contents.

## 2026-09-08 — repository initialization

- Created a dedicated repository for the paper code.
- Established two isolated variants: `with_spatial` and `no_spatial`.
- Defined the large-file archive policy and frozen V1.0 checksums.
- Imported the cleaned paper-oriented codebase, documentation and lightweight reproducibility records.

The historical Stage/patch development tree is intentionally not part of this repository's maintainable primary code path.

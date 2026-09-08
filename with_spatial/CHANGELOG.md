# WITH_SPATIAL Changelog

## 1.0.0 — frozen paper record

Scientific scope remains identical to the final audited 539-D experiment.

- reorganized the final method into paper-oriented modules;
- removed Stage/patch/history workflow dependencies from the maintainable main path;
- retained the original 539-D input contract, including 12 spatial inputs;
- added source-CSV → 539-D feature reconstruction and archive checks;
- locked the five-fold two-station deployment-group daily paired statistic;
- documented baseline history windows: neural local updates 30 d, local rolling ridge 60 d;
- added Table 2, Table 3 and Table 4 reproduction paths;
- added causal information-boundary, OOF-regression and baseline-protocol tests;
- added paper figure-generation scripts;
- documented exact archive SHA-256, restoration policy and numerical environment/tolerance policy;
- hardened sufficient-statistics equivalence checking to `rtol=0, atol=1e-10` for cross-BLAS portability. This test-only change does not alter model calculations or reported results.

### GitHub archive hardening — 2026-09-08

The following maintenance records were added without changing the scientific implementation or frozen V1.0 results:

- promoted `with_spatial/` to the explicit authoritative repository priority;
- added complete paper metric definitions and result-provenance/evidence-level documentation;
- added the frozen 40-station train/validation/test split, source-data cleaning provenance and full source-file SHA-256 manifest;
- added 539-D feature-group index ranges and a tool to export the exact ordered feature schema from the restored frozen cache;
- added frozen archive/package/verification checksum records under `archive/`;
- documented figure reproduction for the scalar-coefficient, type-improvement and representative-curve figures;
- added a long-term GitHub branching/versioning/maintenance guide;
- tracked private Release upload of the authoritative frozen ZIP in repository Issue #3;
- corrected archive documentation so the full 5-min payload and large caches remain outside ordinary Git history while retaining exact restore paths and checksums.

## Historical engineering milestones

### 1.0.0-rc3

- added the complete 40-station source dataset to the frozen archive;
- added source-data SHA-256 manifest and raw-to-cache equivalence evidence;
- added `reproduce_all.py` and figure-input contracts;
- aligned representative-curve plotting with the paper display window beginning 2023-02-15.

### 1.0.0-rc2

- added Table 2 locked comparison result adapters;
- recorded 30 d history for full fine-tuning/local MLP/pFedMe and 60 d for local ridge;
- added comparison paired-daily inference and retired-metric guard.

### 1.0.0-rc1

- decomposed the final optimized method into paper sections;
- removed Stage/patch dependencies from the main path;
- established 539-D input/statistics, five-fold OOF and causal-boundary regression tests.

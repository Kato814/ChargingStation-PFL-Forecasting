# Environment and numerical reproducibility notes

## Package lock

The V1.0 archive records the following Python-package versions:

- NumPy `2.3.5`
- pandas `2.2.3`
- SciPy `1.17.0`
- Matplotlib `3.10.8`

See `requirements-lock.txt`.

## Archive-maintenance validation environment

The repository hardening/validation session used:

- Python `3.13.5`
- Linux x86_64
- NumPy `2.3.5`
- pandas `2.2.3`
- SciPy `1.17.0`
- Matplotlib `3.10.8`
- SciPy OpenBLAS `0.3.30`, 64-bit integer build

This records the environment used to re-check the cleaned archive. It is not claimed to be the only environment in which the original research experiment was executed.

## Floating-point policy

Byte-level archive integrity and numerical-model equivalence are different checks.

- ZIP/member/checksum tests should remain exact.
- Feature names, dates, station IDs, shapes and categorical metadata should remain exact.
- Floating-point arrays reconstructed on different BLAS/CPU/Python builds may differ at machine precision because reduction order can change.

For sufficient-statistics reconstruction, use an explicit absolute numerical tolerance rather than `diff == 0.0`. The repository maintenance threshold is `atol=1e-10`, `rtol=0`, which is substantially tighter than any precision relevant to the reported forecasts and safely covers the observed cross-environment difference of approximately `4.55e-13` in one `X^T X` element.

Forecast regression against the frozen OOF result should likewise use an explicit tight tolerance and must still preserve all discrete enable/fallback states exactly.

## Reproducibility interpretation

A test PASS means the clean paper-oriented implementation is numerically equivalent to the frozen experiment under the stated tolerance. It does not mean that every floating-point intermediate is guaranteed to be bitwise identical on every CPU/BLAS implementation.

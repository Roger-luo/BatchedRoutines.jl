# BatchedRoutines

[![CI](https://github.com/Roger-luo/BatchedRoutines.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Roger-luo/BatchedRoutines.jl/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/Roger-luo/BatchedRoutines.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Roger-luo/BatchedRoutines.jl)
[![Coveralls](https://coveralls.io/repos/github/Roger-luo/BatchedRoutines.jl/badge.svg?branch=master)](https://coveralls.io/github/Roger-luo/BatchedRoutines.jl?branch=master)

Batched routines (BLAS, LAPACK, etc.) for multi-dimensional arrays. This package provide both CPU support, for GPU support see [CuBatchedRoutines](https://github.com/Roger-luo/CuBatchedRoutines.jl).

## Installation

```julia
pkg> add BatchedRoutines
```

## Supported Routines

### BLAS Level 1

- [x] `batched_scal`
- [ ] `batched_dot`
- [ ] `batched_nrm2`
- [ ] `batched_asum`
- [ ] `batched_axpy!`
- [ ] `batched_axpby`
- [ ] `batched_iamax`

### BLAS Level 2

- [ ] `batched_gemv`
- [ ] `batched_gbmv`
- [ ] `batched_symv`
- [ ] `batched_hemv`
- [ ] `batched_sbmv`
- [ ] `batched_hbmv`
- [ ] `batched_trmv`
- [ ] `batched_trsv`
- [ ] `batched_ger`
- [ ] `batched_syr`
- [ ] `batched_her`

### BLAS Level 3

- [x] `batched_gemm` (TODO: use `gemm_batch` when mkl is available)
- [ ] `batched_symm`
- [ ] `batched_hemm`
- [ ] `batched_syrk`
- [ ] `batched_herk`
- [ ] `batched_syr2k`
- [ ] `batched_trmm`
- [ ] `batched_trsm`
- [ ] `batched_trmm`

### Linear Algebra

- [x] `batched_tr`

### Conventions

For routines (e.g gemm), we use a prefix batched_ for their corresponding routines in BLAS or LAPACK and they should only define with AbstractArray{T, 3} (batched matrix) or AbstractArray{T, 2} (batched vector).

## License

MIT

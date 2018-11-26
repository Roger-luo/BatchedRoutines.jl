# BatchedRoutines

[![Build Status](https://travis-ci.com/Roger-luo/BatchedRoutines.jl.svg?branch=master)](https://travis-ci.com/Roger-luo/BatchedRoutines.jl)
[![Codecov](https://codecov.io/gh/Roger-luo/BatchedRoutines.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Roger-luo/BatchedRoutines.jl)
[![Coveralls](https://coveralls.io/repos/github/Roger-luo/BatchedRoutines.jl/badge.svg?branch=master)](https://coveralls.io/github/Roger-luo/BatchedRoutines.jl?branch=master)

## Installation

```julia
pkg> install BatchedRoutines
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

- [x] `batched_gemm`
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

## License

MIT

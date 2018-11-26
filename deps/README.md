# Build BatchedRoutines

The build script will try to find CUDA Toolkit with `CUDAapi`, and if CUDA is usable
it will add `CuArrays` and `GPUArrays` as dependency in `Project.toml` and let
`USE_CUDA = true`, then the package will precompile with CUDA methods.

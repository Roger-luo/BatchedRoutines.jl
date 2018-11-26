module BatchedRoutines

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("TestCUDA.jl has not been built, please run Pkg.build(\"TestCUDA\").")
include(ext)

include("blas.jl")
include("lapack.jl")
include("linalg.jl")

@static if USE_CUDA
    include("cuda/cuda.jl")
end

end # module

module BatchedRoutines

macro __PKGNAME__()
    return "BatchedRoutines"
end

const ext = joinpath(dirname(@__DIR__), "deps", "ext.jl")
isfile(ext) || error("$(@__PKGNAME__).jl has not been built, please run Pkg.build(\"$(@__PKGNAME__)\").")
include(ext)

include("blas.jl")
include("lapack.jl")
include("linalg.jl")

@static if USE_CUDA
    include("cuda/cuda.jl")
end

end # module

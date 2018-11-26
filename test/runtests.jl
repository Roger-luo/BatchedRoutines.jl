using BatchedRoutines
using Test

@testset "Testing BLAS" begin
    include("blas.jl")
end

@static if BatchedRoutines.USE_CUDA
    @testset "Testing CUDA" begin
        include("cuda/cuda.jl")
    end
end

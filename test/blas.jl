using BatchedRoutines
using Test

@testset "Testing batched_scal! with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, 100)
    B = rand(elty, 10, 10, 100)
    expect_B = copy(B)
    batched_scal!(A, B)

    for k in 1:100
        expect_B[:, :, k] .= A[k] * expect_B[:, :, k]
    end

    @test expect_B ≈ B
end

@testset "Testing batched_scal with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, 100)
    B = rand(elty, 10, 10, 100)
    expect_B = copy(B)
    test_B = batched_scal(A, B)

    for k in 1:100
        expect_B[:, :, k] .= A[k] * expect_B[:, :, k]
    end

    @test expect_B ≈ test_B
end

@testset "Testing batched_gemm! with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]

    alpha, beta = rand(elty), rand(elty)
    A = rand(elty, 10, 10, 100)
    B = rand(elty, 10, 10, 100)
    C = rand(elty, 10, 10, 100)
    test_C = copy(C)

    batched_gemm!('N', 'N', alpha, A, B, beta, C)

    for k in 1:100
        test_C[:, :, k] = alpha * (A[:, :, k]) * B[:, :, k] + beta * test_C[:, :, k]
    end

    @test C ≈ test_C
end

@testset "Testing batched_gemm with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    alpha = rand(elty)
    A = rand(elty, 10, 10, 100)
    B = rand(elty, 10, 10, 100)

    C = batched_gemm('N', 'N', alpha, A, B)

    test_C = zeros(elty, 10, 10, 100)
    for k in 1:100
        test_C[:, :, k] = alpha * (A[:, :, k]) * B[:, :, k]
    end

    @test test_C ≈ C
end

using BatchedRoutines, Test, CuArrays

@testset "Testing batched_scal! with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, 100)
    B = rand(elty, 10, 10, 100)

    dA, dB = CuArray(A), CuArray(B)
    batched_scal!(dA, dB)

    for k in 1:100
        B[:, :, k] .= A[k] * B[:, :, k]
    end

    test_B = Array(dB)

    @test B ≈ test_B
end

@testset "Testing batched_scal with $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, 100)
    B = rand(elty, 10, 10, 100)

    dA, dB = CuArray(A), CuArray(B)
    test_B = Array(batched_scal(A, B))

    for k in 1:100
        B[:, :, k] .= A[k] * B[:, :, k]
    end

    @test B ≈ test_B
end

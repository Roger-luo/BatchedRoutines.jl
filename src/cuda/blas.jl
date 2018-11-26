using GPUArrays
import CuArrays

function BatchedRoutines.batched_scal!(A::CuArray{T, 1}, B::CuArray{T, 3}) where T
    gpu_call(B, (A, B)) do state, A, B
        batch = @linearidx(A)
        s = zero(eltype(B))
        for i in axes(B, 1), j in axes(B, 2)
            B[i, j, batch] *= A[batch]
        end
        return
    end
    return B
end


for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))

    @eval begin
        function BatchedRoutines.batched_gemm!(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::CuArray{$elty, 3}, B::CuArray{$elty, 3}, beta::($elty), C::CuArray{$elty, 3})
            CuArrays.CUBLAS.gemm_strided_batched!(transA, transB, alpha, A, B, beta, C)
        end

        function BatchedRoutines.batched_gemm(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::CuArray{$elty, 3}, B::CuArray{$elty, 3})
            CuArray.CUBLAS.gemm_strided_batched(transA, transB, alpha, A, B)
        end

        function BatchedRoutines.batched_gemm(transA::AbstractChar, transB::AbstractChar, A::CuArray{$elty, 3}, B::CuArray{$elty, 3})
            CuArray.CUBLAS.gemm_strided_batched(transA, transB, A, B)
        end
    end
end

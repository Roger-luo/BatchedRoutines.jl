struct CUDANotImplementedError <: Exception end

Base.show(io::IO, ::CUDANotImplementedError) = print(io, "CUDA version is not implemented")

batched_scal!(A::CuArray{T, 1}, B::CuArray{T, 3}) where T = throw(CUDANotImplementedError())

for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))

    @eval begin
        function batched_gemm!(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::CuArray{$elty, 3}, B::CuArray{$elty, 3}, beta::($elty), C::CuArray{$elty, 3})
            throw(CUDANotImplementedError())
        end

        function batched_gemm(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::CuArray{$elty, 3}, B::CuArray{$elty, 3})
            throw(CUDANotImplementedError())
        end

        function batched_gemm(transA::AbstractChar, transB::AbstractChar, A::CuArray{$elty, 3}, B::CuArray{$elty, 3})
            throw(CUDANotImplementedError())
        end
    end
end

batched_tr(A::CuArray{T, 3}) where T = throw(CUDANotImplementedError())
batched_tr!(B::CuArray{T, 1}, A::CuArray{T, 3}) where T = throw(CUDANotImplementedError())

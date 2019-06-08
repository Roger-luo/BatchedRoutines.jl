export batched_scal, batched_scal!, batched_gemm!, batched_gemm

import LinearAlgebra: BLAS

# level 1
"""
    batched_scal!(s, X)

Overwrite `X[:, :, i]` with `a[i] * X[:, :, i]`, where `i` is the batch dimension.
"""
function batched_scal! end

"""
    batched_scal(s, X)

Return `X` scaled by `a` for all the batch.
"""
function batched_scal end

batched_scal(s::AbstractVector{T}, X::AbstractArray{T, 3}) where T = batched_scal!(s, copy(X))

function batched_scal!(A::AbstractVector{T}, B::AbstractArray{T, 3}) where T
    @iterate_batch T A, B (1, 2) begin
        BLAS.scal!(stride(B, 3), A[batch_k], ptrB, 1)
    end
    return B
end

# TODO: use gemm_batch when mkl is available
"""
    batched_gemm!(transA, transB, alpha, A, B, beta, C)

Batched gemm!.
"""
function batched_gemm! end

"""
    batched_gemm(transA, transB, alpha, A, B)

Batched gemm.
"""
function batched_gemm end

"""
    batched_gemm(alpha, A, B)

"""
batched_gemm(alpha::T, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T = batched_gemm('N', 'N', alpha, A, B)

for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))
    @eval begin

        function batched_gemm!(
            transA::AbstractChar, transB::AbstractChar,
            alpha::($elty), A::AbstractArray{($elty), 3}, B::AbstractArray{($elty), 3},
            beta::($elty), C::AbstractArray{($elty), 3})
        
            @assert !BLAS.has_offset_axes(A, B, C)
            @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
            end

            @iterate_batch $(elty) A, B, C (2, 2, 2) begin
            ccall((BLAS.@blasfunc($gemm), BLAS.libblas), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt},
                     Ref{BLAS.BlasInt}, Ref{$(elty)}, Ptr{$(elty)}, Ref{BLAS.BlasInt},
                     Ptr{$(elty)}, Ref{BLAS.BlasInt}, Ref{$(elty)}, Ptr{$(elty)},
                     Ref{BLAS.BlasInt}),
                     transA, transB, m, n,
                     ka, alpha, ptrA, max(1,stride(A,2)),
                     ptrB, max(1,stride(B,2)), beta, ptrC,
                     max(1,stride(C,2)))    
            end
            return C
        end

        function batched_gemm(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            batched_gemm!(transA, transB, alpha, A, B, zero($elty), similar(B, $elty, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1), size(B, 3))))
        end
        function batched_gemm(transA::AbstractChar, transB::AbstractChar, A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            batched_gemm(transA, transB, one($elty), A, B)
        end
    end
end

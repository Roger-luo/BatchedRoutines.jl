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

function batched_scal!(s::AbstractVector{T}, X::AbstractArray{T, 3}) where T
    ptrX = Base.unsafe_convert(Ptr{T}, X)
    chunk_size = size(X, 1) * size(X, 2)
    @inbounds for k in 1:size(X, 3)
        BLAS.scal!(chunk_size, s[k], ptrX, 1)
        ptrX += chunk_size * sizeof(T)
    end
    X
end

function batched_dot end

function batched_gemm! end
function batched_gemm end

for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))
    @eval begin
        function batched_gemm!(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3}, beta::($elty), C::AbstractArray{$elty, 3})
            @assert !BLAS.has_offset_axes(A, B, C)
            @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
            end
            BLAS.chkstride1(A)
            BLAS.chkstride1(B)
            BLAS.chkstride1(C)

            ptrA = Base.unsafe_convert(Ptr{$elty}, A)
            ptrB = Base.unsafe_convert(Ptr{$elty}, B)
            ptrC = Base.unsafe_convert(Ptr{$elty}, C)

            for k in 1:size(A, 3)
                ccall((BLAS.@blasfunc($gemm), BLAS.libblas), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt},
                     Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BLAS.BlasInt},
                     Ptr{$elty}, Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty},
                     Ref{BLAS.BlasInt}),
                     transA, transB, m, n,
                     ka, alpha, ptrA, max(1,stride(A,2)),
                     ptrB, max(1,stride(B,2)), beta, ptrC,
                     max(1,stride(C,2)))

                ptrA += size(A, 1) * size(A, 2) * sizeof($elty)
                ptrB += size(B, 1) * size(B, 2) * sizeof($elty)
                ptrC += size(C, 1) * size(C, 2) * sizeof($elty)
            end
            C
        end
        function batched_gemm(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            batched_gemm!(transA, transB, alpha, A, B, zero($elty), similar(B, $elty, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1), size(B, 3))))
        end
        function batched_gemm(transA::AbstractChar, transB::AbstractChar, A::AbstractArray{$elty, 3}, B::AbstractArray{$elty, 3})
            batched_gemm(transA, transB, one($elty), A, B)
        end
    end
end

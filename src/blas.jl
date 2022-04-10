export batched_scal, batched_scal!, batched_gemm!, batched_gemm

import LinearAlgebra: BLAS
import LinearAlgebra.BLAS: @blasfunc, BlasInt
@static if VERSION < v"1.7"
    import LinearAlgebra.BLAS: libblas, liblapack
else
    const libblas = BLAS.libblastrampoline
    const liblapack = BLAS.libblastrampoline
end

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


for (fname, elty, lib) in ((:dsyr_,:Float64,libblas),
    (:ssyr_,:Float32,libblas),
    (:zsyr_,:ComplexF64,liblapack),
    (:csyr_,:ComplexF32,liblapack))
    @eval begin
        function batched_syr!(uplo::AbstractChar, α::$elty, x::AbstractArray{$elty, 2}, A::AbstractArray{$elty, 3})
            @assert !Base.has_offset_axes(A, x)
            n = checksquare(A)
            if length(x) != n
            throw(DimensionMismatch("A has size ($n,$n), x has length $(length(x))"))
            end

            @iterate_batch $(elty) (x, A) (1, 2) begin
                ccall((@blasfunc($fname), $lib), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
                Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                uplo, n, α, ptrx,
                stride(x, 1), ptrA, max(1,stride(A, 2)))
            end
            A
        end
    end
end


for (fname, elty, relty) in ((:zher_,:ComplexF64, :Float64),
                             (:cher_,:ComplexF32, :Float32))
    @eval begin
        function batched_her!(uplo::AbstractChar, α::$relty, x::AbstractMatrix{$elty}, A::AbstractArray{$elty, 3})
            @assert !Base.has_offset_axes(A, x)
            n = checksquare(A)
            if length(x) != n
                throw(DimensionMismatch("A has size ($n,$n), x has length $(length(x))"))
            end
            @iterate_batch $(elty) (x, A) (1, 2) begin
                ccall((@blasfunc($fname), libblas), Cvoid,
                    (Ref{UInt8}, Ref{BlasInt}, Ref{$relty}, Ptr{$elty},
                    Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                    uplo, n, α, x,
                    stride(x, 1), A, max(1,stride(A,2)))
            end
            A
        end
    end
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

            @assert !Base.has_offset_axes(A, B, C)
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

function batched_syrk! end
function batched_syrk end

for (fname, elty) in ((:dsyrk_,:Float64),
                      (:ssyrk_,:Float32),
                      (:zsyrk_,:ComplexF64),
                      (:csyrk_,:ComplexF32))
   @eval begin
       # SUBROUTINE DSYRK(UPLO,TRANS,N,K,ALPHA,A,LDA,BETA,C,LDC)
       # *     .. Scalar Arguments ..
       #       REAL ALPHA,BETA
       #       INTEGER K,LDA,LDC,N
       #       CHARACTER TRANS,UPLO
       # *     .. Array Arguments ..
       #       REAL A(LDA,*),C(LDC,*)
       function syrk!(uplo::AbstractChar, trans::AbstractChar,
                      alpha::($elty), A::AbstractVecOrMat{$elty},
                      beta::($elty), C::AbstractMatrix{$elty})
           @assert !has_offset_axes(A, C)
           n = checksquare(C)
           nn = size(A, trans == 'N' ? 1 : 2)
           if nn != n throw(DimensionMismatch("C has size ($n,$n), corresponding dimension of A is $nn")) end
           k  = size(A, trans == 'N' ? 2 : 1)

           ccall((@blasfunc($fname), libblas), Cvoid,
                 (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                  Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty},
                  Ptr{$elty}, Ref{BlasInt}),
                 uplo, trans, n, k,
                 alpha, A, max(1,stride(A,2)), beta,
                 C, max(1,stride(C,2)))
            C
        end
    end
end
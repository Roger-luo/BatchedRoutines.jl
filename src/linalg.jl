export batched_tr, batched_tr!

"""
    batched_tr(A::AbstractArray{T, 3}) where T

Batched version of trace.
"""
batched_tr(A::AbstractArray{T, 3}) where T = batched_tr!(A, fill!(similar(A, (size(A, 3), )), 0))

function batched_tr!(A::AbstractArray{T, 3}, B::AbstractVector{T}) where T
    @boundscheck size(A, 1) == size(A, 2) || error("Expect a square matrix")
    @boundscheck size(A, 3) == length(B) || error("Batch size mismatch")
    @inbounds for k in 1:size(A, 3), i in 1:size(A, 1)
        B[k] += A[i, i, k]
    end
    B
end

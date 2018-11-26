using GPUArrays

# CuArrays#207
batched_tr(A::CuArray{T, 3}) where T = BatchedRoutines.batched_tr!(similar(A, size(A, 3)), A)

function batched_tr!(B::CuArray{T, 1}, A::CuArray{T, 3}) where T
    gpu_call(B, (A, B)) do state, A, B
        batch = @linearidx(B)
        s = zero(eltype(B))
        for i in axes(A, 1)
            s += A[i, i, batch]
        end
        B[batch] = s
        return
    end
    return B
end

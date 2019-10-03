module BatchedRoutines

function reshaped_batch_size(n::Int, A::AbstractArray)
    return prod(k->size(A, k+n), ndims(A) - n)
end

function reshaped_batch_size(n::Int, A::AbstractArray{T, 3}) where T
    n == 2 && return size(A, 3)
    return prod(k->size(A, k+n), ndims(A) - n)
end

macro iterate_batch(elty, Xs, dims, f)
    ex = Expr(:block)
    for X in Xs.args
        push!(ex.args, :(BLAS.chkstride1($(esc(X)))))
    end
    # check batch size
    batch_check = Expr(:comparison)
    for k in 1:length(Xs.args)-1
        X = Xs.args[k]; n = dims.args[k]
        push!(batch_check.args, :(reshaped_batch_size($n, $(esc(X)))))
        push!(batch_check.args, :(==))
    end
    push!(ex.args, :($(esc(:batch_n)) = reshaped_batch_size($(dims.args[end]), $(esc(Xs.args[end])))))
    push!(batch_check.args, esc(:batch_n))
    push!(ex.args, Expr(:(||), batch_check, :(error("batch size mismatch"))))

    # assign batch size

    # generated ptrs
    for_block = Expr(:block, esc(f))

    for (X, n) in zip(Xs.args, dims.args)
        ptrX = Symbol(:ptr, X)
        push!(ex.args, :($(esc(ptrX)) = Base.unsafe_convert(Ptr{$(esc(elty))}, $(esc(X)))))
        push!(for_block.args, :($(esc(ptrX)) += stride($(esc(X)), $(n + 1)) * sizeof($(esc(elty)))))
    end

    forloop = Expr(:for, :($(esc(:batch_k)) = 1:$(esc(:batch_n))), for_block)
    push!(ex.args, forloop)
    return ex
end

include("blas.jl")
include("lapack.jl")
include("linalg.jl")

end # module

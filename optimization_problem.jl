include("direct_MPO_computation.jl")
include("middle_out_contraction.jl")

# Middle-out contraction calculation of M and L for the optimization problem

function gram_matrix_middle_out(n, J, t, ks, sites, psi; cutoff=0.0, maxdim=50, order::Int=2)
    r = length(ks)
    M = zeros(Float64, r, r)

    Fs = build_F(n, J, t, ks, sites, cutoff, maxdim; order=order)

    idx = 1
    for i in 1:r
        for j in 1:r
            amp = inner(psi', Fs[idx], psi)
            M[i, j] = abs2(amp)
            idx += 1
        end
    end

    return M
end

function L_vector_middle_out(n, J, t, ks, k_ref, sites, psi; cutoff=0.0, maxdim=50, order::Int=2, order_ref::Int=2)
    Fs = build_F_between_lists(
        n, J, t, ks, [k_ref], sites, cutoff, maxdim;
        order_left=order, order_right=order_ref
    )

    L = zeros(Float64, length(ks))
    for j in eachindex(ks)
        amp = inner(psi', Fs[j], psi)
        L[j] = abs2(amp)
    end

    return L
end

function dynamic_mpf_coefficients(M::AbstractMatrix, L::AbstractVector)
    r = size(M, 1)
    @assert size(M, 2) == r
    @assert length(L) == r

    A = zeros(Float64, r + 1, r + 1)
    b = zeros(Float64, r + 1)

    A[1:r, 1:r] .= M
    A[1:r, r+1] .= -1.0
    A[r+1, 1:r] .= 1.0

    b[1:r] .= L
    b[r+1] = 1.0

    x = A \ b
    c = x[1:r]
    λ = x[r+1]

    return c, λ
end

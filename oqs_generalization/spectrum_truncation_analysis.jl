using ITensors, ITensorMPS
using LinearAlgebra
include("open_optimization_problem.jl")   # pulls in the full include chain
                                           # (liouville_space -> open_product_formula_generation
                                           #  -> open_middle_out_contraction -> bond_dimension_tracking
                                           #  -> F_diagnostics -> open_optimization_problem)

# =============================================================================
# PART 1: Full singular-value spectrum across the middle cut
# =============================================================================
#
# `operator_schmidt_rank` (in bond_dimension_tracking.jl) already does the
# hard part -- orthogonalize to the middle bond, then SVD the bond tensor --
# but throws away everything except a count of singular values above
# `cutoff`. `operator_schmidt_spectrum` does the identical construction and
# instead returns the full vector of singular values (untruncated: cutoff=0
# in the SVD call), sorted descending. This is what you actually want to look
# at to answer "is the tail numerically negligible even though the nominal
# rank/bond-dim count is large" -- a question `bond_dim`/`schmidt_rank` alone
# cannot answer, since both are just counts against a threshold.
#
# NOTE: this SVD is over whatever bond dimension is *already stored* in M
# (i.e. it does not undo any truncation baked in during construction via
# `apply(...; cutoff, maxdim)`). If you want the spectrum of the "true"
# untruncated F, build F with a very loose maxdim (e.g. maxdim=4^n, the
# theoretical ceiling) and cutoff=0, then call this on the result. If you
# want the spectrum of F *as it was actually built* at production settings,
# just pass in that F directly -- this is exactly what you want for the
# truncation-error sweep in Part 2.

function operator_schmidt_spectrum(M::MPO)
    n = length(M)
    mid = n ÷ 2
    mid == 0 && return [1.0]

    M_orth = deepcopy(M)
    orthogonalize!(M_orth, mid)

    left_tensor = M_orth[mid]
    right_link = commonind(M_orth[mid], M_orth[mid+1])

    U, S, V = svd(left_tensor, right_link; cutoff=0.0)
    d = dim(commonind(U, S))
    return [S[i, i] for i in 1:d] |> s -> sort(s; rev=true)
end

# =============================================================================
# Layer-by-layer spectrum tracer for F_ii = [S(t/k)^k]^† S(t/k)^k
# =============================================================================
#
# Same incremental construction as `track_Fii_bond_dimension`, but records
# the full singular-value spectrum at every layer instead of (or alongside)
# just the bond dimension / rank count. This is the direct tool for your
# n=5, gamma=0.01 case: run this, then plot log10.(spectrum[layer] ./
# spectrum[layer][1]) vs index for a few layers (e.g. the last one, where
# bond_dim already reads 256) to see whether the tail is a fast-decaying
# numerical shoulder or genuinely flat/significant all the way out.
#
# Returns a NamedTuple with `layer` (Vector{Int}), `bond_dim` (Vector{Int},
# same as track_Fii_bond_dimension for cross-checking), and `spectrum`
# (Vector{Vector{Float64}}, one entry per layer).

function track_Fii_spectrum(
    n, J, gammas, t, k, lsites::LiouvilleSites, cutoff, maxdim;
    order::Int=1, dissipation::Bool=true
)
    dt = t / k
    step_mpo     = get_open_step_MPO(n, J, gammas, dt, lsites, cutoff, maxdim; order=order, dissipation=dissipation)
    step_dag_mpo = get_open_step_MPO_dag(n, J, gammas, dt, lsites, cutoff, maxdim; order=order, dissipation=dissipation)

    F = identity_liouville_mpo(lsites)

    layers = Int[]
    bond_dims = Int[]
    spectra = Vector{Float64}[]

    for layer in 1:k
        F = right_multiply(F, step_mpo; cutoff=cutoff, maxdim=maxdim)
        F = left_multiply(step_dag_mpo, F; cutoff=cutoff, maxdim=maxdim)

        push!(layers, layer)
        push!(bond_dims, middle_bond_dim(F))
        push!(spectra, operator_schmidt_spectrum(F))
    end

    return (layer=layers, bond_dim=bond_dims, spectrum=spectra, F_final=F)
end

# Convenience: how many singular values are needed at a given layer's
# spectrum to capture a target fraction of the total Frobenius weight
# (sum of squares of singular values), e.g. 1 - 1e-6. This turns a raw
# spectrum into a single "effective rank" number directly comparable to
# `bond_dim`/`schmidt_rank`, but based on weight retained rather than a
# fixed absolute cutoff -- usually the more meaningful quantity for deciding
# a practical maxdim.

function effective_rank(spectrum::Vector{Float64}; weight_fraction::Float64=1 - 1e-6)
    weights = spectrum .^ 2
    total = sum(weights)
    total == 0 && return 0
    cum = cumsum(weights) ./ total
    return findfirst(>=(weight_fraction), cum)
end

# =============================================================================
# PART 2: maxdim truncation-error sweep
# =============================================================================
#
# The spectrum tells you how much weight *should* be discardable in
# principle. This sweep tells you what actually happens to the quantities you
# care about (M_ij, the DMPF coefficients c, and the DMPF error E_mpf) when
# you build F at a sequence of increasingly generous maxdim values and
# compare each against one high-maxdim reference -- exactly mirroring how
# main.pdf itself validates maxdim=50 against maxdim=200, rather than
# assuming any particular maxdim is "enough."
#
# For each maxdim in `maxdims`, this:
#   1. builds M_opt/L_opt/c at that maxdim (the "candidate" accuracy),
#   2. evaluates E_mpf and E_trot using c but with the FIXED high-accuracy
#      reference (maxdim_ref) -- i.e. exactly `test_dynamic_mpf_open`'s
#      opt/ref split, just scanning maxdim_opt over a list instead of a
#      single value,
#   3. additionally reports ||M_opt - M_ref||_F (Frobenius distance of the
#      Gram matrix itself from the reference), which is a more direct,
#      model-free signal of "did this maxdim already capture the important
#      structure" than the downstream DMPF error alone.
#
# Reuses `test_dynamic_mpf_open` unchanged for each maxdim, so all the
# opt/ref bookkeeping (purity, error formulas, etc.) stays exactly consistent
# with the rest of the codebase -- this file adds no new physics, only a
# scan + comparison layer on top.

function maxdim_truncation_sweep(
    n, J, gammas, t, ks, k_ref_opt, k_ref_eval, lsites::LiouvilleSites, rho0::MPS,
    maxdims::Vector{Int};
    maxdim_ref::Int=256, cutoff::Float64=1e-12,
    order::Int=2, order_ref_opt::Int=2, order_ref_eval::Int=2,
    dissipation::Bool=true
)
    @assert maxdim_ref >= maximum(maxdims) "maxdim_ref should be at least as large as every maxdim being swept, so it serves as a genuine upper reference."

    # High-accuracy reference Gram matrix, built once and reused for every
    # comparison below (this is the "truth" each candidate maxdim is judged
    # against -- not the exact 4^n-dimensional object, which is infeasible
    # for n=5, but the best affordable stand-in, consistent with how
    # main.pdf treats its own maxdim=200 reference as "quasi-exact" rather
    # than exact).
    M_ref_full, _ = open_gram_matrix(
        n, J, gammas, t, ks, lsites, rho0;
        cutoff=cutoff, maxdim=maxdim_ref, order=order, dissipation=dissipation
    )

    results = NamedTuple[]

    for md in maxdims
        r = test_dynamic_mpf_open(
            n, J, gammas, t, ks, k_ref_opt, k_ref_eval, lsites, rho0;
            cutoff_opt=cutoff,  maxdim_opt=md,
            cutoff_eval=cutoff, maxdim_eval=maxdim_ref,
            order=order, order_ref_opt=order_ref_opt, order_ref_eval=order_ref_eval,
            dissipation=dissipation
        )

        M_dist = norm(r.M_opt .- M_ref_full)   # Frobenius distance, Gram matrix built at md vs maxdim_ref

        push!(results, (
            maxdim   = md,
            M_dist   = M_dist,
            coeffs   = r.coeffs,
            E_mpf    = r.E_mpf,
            E_trot   = r.E_trot,
            M_opt    = r.M_opt,
        ))
    end

    return (maxdims=maxdims, results=results, M_ref=M_ref_full)
end

# =============================================================================
# Convenience printer
# =============================================================================

function print_sweep_summary(sweep)
    println("maxdim | ||M-M_ref||_F | E_mpf        | coeffs")
    for res in sweep.results
        println(
            rpad(res.maxdim, 6), " | ",
            rpad(round(res.M_dist; sigdigits=4), 13), " | ",
            rpad(round(res.E_mpf; sigdigits=4), 12), " | ",
            round.(res.coeffs; digits=4)
        )
    end
end

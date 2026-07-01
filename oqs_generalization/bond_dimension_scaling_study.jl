# =============================================================================
# bond_dimension_scaling_study.jl
#
# Diagnostic study of the middle-bond dimension of F_ii = (S^†)^k S^k built
# with the tensor-network machinery in bond_dimension_tracking.jl, as a
# check on whether the dense/direct-exponentiation route (liouville_space.jl
# + open_product_formula_generation.jl) is a computational bottleneck.
#
# Two experiments:
#   (1) bond dimension vs. Trotter layer, several gamma curves, fixed n
#   (2) bond dimension vs. system size n, several gamma curves
# both compared against the theoretical maximum bond dimension.
#
# -----------------------------------------------------------------------
# THEORETICAL MAXIMUM BOND DIMENSION
# -----------------------------------------------------------------------
# Each site of the Liouville-space MPO carries 4 indices -- ket, bra, and
# their primed counterparts, each of dimension 2 (see liouville_space.jl:
# identity_liouville_mpo builds one ITensor per site from
# op("Id",ket_s)*op("Id",bra_s), i.e. a *single* vectorized "super-site" of
# local dimension d=4 per leg (in AND out), NOT two separate dimension-2
# MPO sites). Treating the chain as n sites of local operator dimension
# d=4 (in=4, out=4), an *arbitrary* (untruncated) operator at the cut that
# splits the chain into l sites on one side and n-l on the other has
# Schmidt rank bounded by d^(2*min(l,n-l)) = 16^min(l,n-l) -- the "d^2"
# because both the in- and out-legs of the l sites on the small side
# contribute to the effective vector space dimension seen across that cut.
#
# middle_bond_dim(M) in bond_dimension_tracking.jl reads linkdims(M)[n÷2],
# i.e. the cut with l = n÷2 sites on the left (Julia floor division), so:
#
#   chi_max(n) = 16 ^ min(n÷2, n - n÷2)
#
# Sanity check against the dense (non-TN) computations already reported in
# the project notes: n=4 -> 16^2 = 256 (matches "saturating the 4^4=256
# ceiling"); n=6 -> 16^3 = 4096 (matches "ceiling of 4096").
# -----------------------------------------------------------------------
#
# A NOTE ON cutoff / maxdim FOR THIS STUDY (as opposed to production use):
#   - `cutoff` is the relative-weight SVD truncation threshold passed to
#     `apply`. We default to 1e-10: comfortably above double-precision
#     noise (~1e-14 relative, with margin), but tight enough to actually
#     discard the long tail of negligible singular values rather than
#     keeping numerical noise as "bond dimension". Feel free to sweep this
#     too (see `cutoffs_to_check` below) to confirm the qualitative growth
#     picture doesn't depend sensitively on the exact choice.
#   - `maxdim` should be set GENEROUSLY for this diagnostic. The whole
#     point is to observe the *true* growth of the bond dimension; if
#     maxdim is set too small it will simply cap the measured value and
#     the plot will just show a flat line, telling you nothing about the
#     underlying growth rate. Because chi_max(n) = 16^min(n÷2,n-n÷2) grows
#     very fast, this is unavoidable for large n -- points that hit the
#     cap are explicitly flagged in the n-scaling plot (marked with a red
#     cross) so you can tell "genuinely saturated" apart from "artificially
#     capped".
# =============================================================================

using ITensors, ITensorMPS
using LinearAlgebra
using Distributions
using Random
using Plots
using DelimitedFiles

# Pulls in bond_dimension_tracking.jl, open_middle_out_contraction.jl,
# open_product_formula_generation.jl, liouville_space.jl via its own
# include chain -- adjust the path below if this script lives somewhere
# other than next to the project files.
include("F_diagnostics.jl")

# =============================================================================
# Theoretical maximum
# =============================================================================

theoretical_max_bond_dim(n::Int) = 16 ^ min(n ÷ 2, n - n ÷ 2)

# =============================================================================
# Experiment 1: bond dimension vs. layer, several gamma curves, fixed n
# =============================================================================

"""
    layer_growth_study(n, gammas_list, t, k; cutoff, maxdim, order, seed)

Build F_ii = (S^†)^k S^k layer by layer for each dissipation rate in
`gammas_list` (uniform gamma on every site), recording the TN-stored
middle-bond dimension after every layer. Returns a Dict gamma => bond_dims
(a length-k Vector{Int}), plus the theoretical max for this n.
"""
function layer_growth_study(n::Int, gammas_list::Vector{Float64}, t::Float64, k::Int;
                             cutoff::Float64=1e-10, maxdim::Int=4000,
                             order::Int=2, seed::Int=1234)
    Random.seed!(seed)
    J = rand(Uniform(1/4, 3/4), n - 1)
    lsites = liouville_siteinds(n)

    results = Dict{Float64,Vector{Int}}()
    for gamma in gammas_list
        gammas = fill(gamma, n)
        res = track_Fii_bond_dimension(
            n, J, gammas, t, k, lsites, cutoff, maxdim;
            order=order, track_exact_rank=false, dissipation=(gamma > 0.0)
        )
        results[gamma] = res.bond_dim
        capped = any(res.bond_dim .>= maxdim)
        println("n=$n, gamma=$gamma: bond_dim by layer = ", res.bond_dim,
                capped ? "   [hit maxdim cap somewhere -- increase maxdim to see true growth]" : "")
    end
    return results, theoretical_max_bond_dim(n)
end

function plot_layer_growth(n::Int, gammas_list::Vector{Float64}, results::Dict, chi_max::Int;
                            maxdim::Union{Int,Nothing}=nothing)
    k = length(first(values(results)))
    plt = plot(xlabel="Trotter layer", ylabel="Middle-bond dimension χ",
               yscale=:log10, legend=:outertopright, size=(750, 500),
               title="F_ii bond-dimension growth vs. layer (n=$n qubits)")
    for gamma in sort(gammas_list)
        bd = results[gamma]
        plot!(plt, 1:k, bd, marker=:circle, label="γ = $gamma")
    end
    hline!(plt, [chi_max], linestyle=:dash, color=:black, lw=2,
           label="theoretical max = $chi_max")
    if maxdim !== nothing
        hline!(plt, [maxdim], linestyle=:dot, color=:red,
               label="maxdim cap = $maxdim")
    end
    return plt
end

# =============================================================================
# Experiment 2: bond dimension vs. system size n, several gamma curves
# =============================================================================

"""
    n_scaling_study(ns, gammas_list, t, k; cutoff, maxdim, order, seed)

For each system size n in `ns` and each gamma in `gammas_list`, build
F_ii over k layers and record the FINAL (layer-k) middle-bond dimension.
Also records whether that final value merely hit the maxdim cap (in which
case it's a lower bound on the true bond dimension, not an exact
measurement). A fresh random J of length n-1 is drawn per n (reproducibly,
via a seed offset by n).

WARNING: cost grows very fast with n, since chi_max(n) = 16^min(n÷2,n-n÷2)
(e.g. n=4 -> 256, n=6 -> 4096, n=8 -> 65536). Start with small `ns` (e.g.
3:6) and a moderate maxdim before pushing further.
"""
function n_scaling_study(ns::Vector{Int}, gammas_list::Vector{Float64}, t::Float64, k::Int;
                          cutoff::Float64=1e-10, maxdim::Int=4000,
                          order::Int=2, seed::Int=1234)
    final_bd = Dict{Float64,Vector{Int}}(gamma => Int[] for gamma in gammas_list)
    capped   = Dict{Float64,Vector{Bool}}(gamma => Bool[] for gamma in gammas_list)
    chi_max  = Int[]

    for n in ns
        Random.seed!(seed + n)   # reproducible but distinct J per n
        J = rand(Uniform(1/4, 3/4), n - 1)
        lsites = liouville_siteinds(n)
        push!(chi_max, theoretical_max_bond_dim(n))

        for gamma in gammas_list
            gammas = fill(gamma, n)
            res = track_Fii_bond_dimension(
                n, J, gammas, t, k, lsites, cutoff, maxdim;
                order=order, track_exact_rank=false, dissipation=(gamma > 0.0)
            )
            bd_final = res.bond_dim[end]
            push!(final_bd[gamma], bd_final)
            push!(capped[gamma], bd_final >= maxdim)
            println("n=$n, gamma=$gamma: final bond dim (layer $k) = $bd_final",
                    bd_final >= maxdim ? "  [CAPPED by maxdim]" : "",
                    "   (theoretical max = $(theoretical_max_bond_dim(n)))")
        end
    end
    return ns, final_bd, capped, chi_max
end

function plot_n_scaling(ns::Vector{Int}, gammas_list::Vector{Float64},
                         final_bd::Dict, capped::Dict, chi_max::Vector{Int};
                         maxdim::Union{Int,Nothing}=nothing)
    plt = plot(xlabel="System size n", ylabel="Middle-bond dimension χ (final layer)",
               yscale=:log10, legend=:outertopright, size=(750, 500),
               title="Bond-dimension scaling with system size")
    for gamma in sort(gammas_list)
        bd  = final_bd[gamma]
        cap = capped[gamma]
        plot!(plt, ns, bd, marker=:circle, label="γ = $gamma")
        capped_ns = ns[cap]
        capped_bd = bd[cap]
        if !isempty(capped_ns)
            scatter!(plt, capped_ns, capped_bd, marker=:xcross, markersize=9,
                     color=:red, label=nothing)
        end
    end
    plot!(plt, ns, chi_max, linestyle=:dash, color=:black, marker=:square, lw=2,
          label="theoretical max 16^min(n÷2,n-n÷2)")
    if maxdim !== nothing
        hline!(plt, [maxdim], linestyle=:dot, color=:gray, label="maxdim cap")
    end
    annotate!(plt, ns[end], maximum(chi_max) * 0.05,
              text("× = point hit maxdim cap\n(lower bound only)", 8, :red, :right))
    return plt
end

# =============================================================================
# Driver
# =============================================================================

function main()
    # ---- experiment 1: layer growth at fixed n ----
    n1 = 5
    gammas_list = [0.01, 0.05, 0.1, 0.2]
    t = 3.0
    k = 8
    cutoff = 1e-10
    maxdim1 = 4000          # generous relative to chi_max(5)=16^2=256

    results, chi_max1 = layer_growth_study(n1, gammas_list, t, k; cutoff=cutoff, maxdim=maxdim1)
    plt1 = plot_layer_growth(n1, gammas_list, results, chi_max1; maxdim=maxdim1)
    savefig(plt1, "layer_growth_n$(n1).png")

    open("layer_growth_n$(n1).csv", "w") do io
        println(io, "layer,", join(["gamma_$g" for g in sort(gammas_list)], ","))
        for layer in 1:k
            row = [string(layer); [string(results[g][layer]) for g in sort(gammas_list)]]
            println(io, join(row, ","))
        end
    end

    # ---- experiment 2: scaling with n ----
    ns = [3, 4, 5, 6]        # extend cautiously -- cost grows like 16^(n/2)
    maxdim2 = 4000            # chi_max(6) = 4096; already brushing the cap at n=6

    ns_out, final_bd, capped, chi_max2 = n_scaling_study(ns, gammas_list, t, k;
                                                          cutoff=cutoff, maxdim=maxdim2)
    plt2 = plot_n_scaling(ns_out, gammas_list, final_bd, capped, chi_max2; maxdim=maxdim2)
    savefig(plt2, "n_scaling.png")

    open("n_scaling.csv", "w") do io
        println(io, "n,", join(["gamma_$g" for g in sort(gammas_list)], ","), ",theoretical_max")
        for (idx, n) in enumerate(ns_out)
            row = [string(n); [string(final_bd[g][idx]) for g in sort(gammas_list)]; string(chi_max2[idx])]
            println(io, join(row, ","))
        end
    end

    println("\nSaved: layer_growth_n$(n1).png, layer_growth_n$(n1).csv, n_scaling.png, n_scaling.csv")
    return plt1, plt2
end

# Uncomment to run when this file is executed directly:
# main()

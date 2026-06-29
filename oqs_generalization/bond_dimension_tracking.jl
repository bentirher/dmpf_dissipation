using ITensors, ITensorMPS
using LinearAlgebra
include("open_middle_out_contraction.jl")

# =============================================================================
# Generic bond-dimension / operator-Schmidt-rank utilities for any MPO
# =============================================================================
#
# `linkdims(M)` already gives the *stored* bond dimension at every internal
# bond of an MPO M (i.e. however large the MPO currently is in memory, after
# whatever truncation has already been applied by `apply(...; cutoff,
# maxdim)`). This is usually what you want to monitor as you build F
# incrementally layer-by-layer.
#
# `middle_bond_dim(M)` returns just the bond dimension across the middle
# cut, i.e. the one separating the left half of the chain from the right
# half -- this is the standard place to look at entanglement/operator
# complexity for a 1D chain.
#
# `operator_schmidt_rank(M; cutoff)` goes one step further: it actually
# performs an SVD across the middle cut (treating M's left dangling
# site-indices as one collective index and the right ones as another) and
# returns the number of singular values above `cutoff`. This is the
# "exact-rank" analogue of the bond dimension reported by `linkdims`: if M
# was built with a loose `maxdim`/`cutoff` along the way, the *stored* bond
# dimension at the middle link may already be smaller than the true
# operator-Schmidt rank computed by truncation-free re-SVD; conversely if
# you used a tight maxdim while building M, the stored linkdim may be an
# underestimate of the true rank because growth was artificially capped.
# `operator_schmidt_rank` always gives you the true number for the *MPO as
# currently stored* (it does not retroactively "undo" prior truncations,
# it just doesn't introduce any new ones beyond the SVD cutoff you pass).

function middle_bond_dim(M::MPO)
    n = length(M)
    mid = n ÷ 2
    mid == 0 && return 1
    return linkdims(M)[mid]
end

function operator_schmidt_rank(M::MPO; cutoff::Float64=1e-12)
    n = length(M)
    mid = n ÷ 2
    mid == 0 && return 1

    # Orthogonalize/contract M into one big tensor split at the middle bond
    # via successive SVDs from both ends, then read off the rank of the
    # remaining single SVD across the cut. The simplest robust way to do
    # this with ITensorMPS, without hand-rolling environment contractions,
    # is to first bring M into a left-canonical-like form up to site `mid`
    # via orthogonalize!, then SVD the bond tensor directly.
    M_orth = deepcopy(M)
    orthogonalize!(M_orth, mid)

    # After orthogonalize!(M_orth, mid), the link between site mid and
    # site mid+1 carries (an upper bound on) the full operator-Schmidt rank
    # of the bipartition. We isolate that one bond and SVD across it
    # explicitly to get an exact, cutoff-controlled rank rather than relying
    # on whatever bond dimension orthogonalize! happens to have left in
    # place (which it does not truncate further on its own).
    left_tensor = M_orth[mid]
    right_link = commonind(M_orth[mid], M_orth[mid+1])

    # All other indices of left_tensor (site indices + any link to site
    # mid-1) are grouped on one side of the SVD; right_link is the other.
    U, S, V = svd(left_tensor, right_link; cutoff=cutoff)
    return dim(commonind(U, S))
end

# =============================================================================
# Layer-by-layer rank tracer for F_ii = [S(t/k)^k]^† S(t/k)^k
# =============================================================================
#
# This directly reproduces, with real MPOs and real SVD truncation (rather
# than dense matrices), the numerical experiment described in the project
# notes (Section 3.3): build F_ii one Trotter layer at a time, alternating
# one forward step (right_multiply by S) with one backward/adjoint step
# (left_multiply by S^†), and record both the middle-bond dimension and
# (optionally, more expensively) the true operator-Schmidt rank after each
# pair of layers.
#
# Returns a NamedTuple with vectors `layer`, `bond_dim`, and (if
# track_exact_rank=true) `schmidt_rank`, suitable for direct plotting.

function track_Fii_bond_dimension(
    n, J, gammas, t, k, lsites::LiouvilleSites, cutoff, maxdim;
    order::Int=1, track_exact_rank::Bool=false, rank_cutoff::Float64=1e-10, dissipation::Bool=true
)
    dt = t / k
    step_mpo     = get_open_step_MPO(n, J, gammas, dt, lsites, cutoff, maxdim; order=order, dissipation=dissipation)
    step_dag_mpo = get_open_step_MPO_dag(n, J, gammas, dt, lsites, cutoff, maxdim; order=order, dissipation=dissipation)

    F = identity_liouville_mpo(lsites)

    layers = Int[]
    bond_dims = Int[]
    schmidt_ranks = Int[]

    for layer in 1:k
        # One full Trotter layer of F_ii is: F <- S^† * (F * S)
        # i.e. one right_multiply by the forward step, then one
        # left_multiply by the adjoint step -- consistent with the clock
        # logic in build_open_F for the i==j (ki==kj) case, where forward
        # and backward clocks always advance in lockstep.
        F = right_multiply(F, step_mpo; cutoff=cutoff, maxdim=maxdim)
        F = left_multiply(step_dag_mpo, F; cutoff=cutoff, maxdim=maxdim)

        push!(layers, layer)
        push!(bond_dims, middle_bond_dim(F))
        if track_exact_rank
            push!(schmidt_ranks, operator_schmidt_rank(F; cutoff=rank_cutoff))
        end
    end

    if track_exact_rank
        return (layer=layers, bond_dim=bond_dims, schmidt_rank=schmidt_ranks, F_final=F)
    else
        return (layer=layers, bond_dim=bond_dims, F_final=F)
    end
end

# =============================================================================
# Full bond-dimension profile (all internal bonds, not just the middle one)
# =============================================================================
#
# Useful for sanity-checking that the middle bond is in fact the bottleneck
# (it usually is for a translationally-near-invariant chain, but it is worth
# confirming rather than assuming, especially with non-uniform gammas).

function full_bond_dimension_profile(M::MPO)
    return linkdims(M)
end

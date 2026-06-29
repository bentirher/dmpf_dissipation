using ITensors, ITensorMPS
using LinearAlgebra
include("open_product_formula_generation.jl")

# =============================================================================
# Left / right multiplication for Liouville-space MPOs
# =============================================================================
#
# Identical in spirit to left_multiply/right_multiply in
# middle_out_contraction.jl. The only difference is that "operator dagger"
# now means: complex-conjugate every tensor AND swap primed/unprimed on BOTH
# the ket leg and the bra leg (since each site of our MPO carries two
# physical-style index pairs, ket and bra, each with its own prime
# structure). `dag(::MPO)` from ITensorMPS already conjugates all tensors and
# swaps primes on every unprimed/primed index pair it finds, so the existing
# `swapprime(dag(A), 0 => 1)` recipe is index-structure-agnostic and works
# unchanged here -- we keep the name `op_dag` for parity with the
# closed-system file.

function op_dag(A::MPO)
    return swapprime(dag(A), 0 => 1)
end

# Left multiplication: A * F
function left_multiply(A::MPO, F::MPO; cutoff, maxdim)
    return apply(A, F; cutoff=cutoff, maxdim=maxdim)
end

# Right multiplication: F * A, implemented as (A† F†)† to avoid index-
# convention issues, exactly as in the closed-system code.
function right_multiply(F::MPO, A::MPO; cutoff, maxdim)
    X = apply(op_dag(A), op_dag(F); cutoff=cutoff, maxdim=maxdim)
    return op_dag(X)
end

# =============================================================================
# Building F_ij (Eq. 50) via middle-out contraction
# =============================================================================
#
# Fij = [S(t/ki)^ki]^†  S(t/kj)^kj
#
# IMPORTANT (see project notes, Section 2): this is the *adjoint*, never the
# inverse, of the i-side channel. We build it incrementally exactly as in
# Algorithm 1 / build_F in middle_out_contraction.jl: maintain two "clocks"
# time_i, time_j, and at each step either right-multiply by one more S_j step
# (if the j-clock is behind or equal) or left-multiply by one more S_i^†
# step (if the i-clock is behind). The per-step MPOs used here are the
# *open-system* step_MPO / step_MPO_dag built in
# open_product_formula_generation.jl (Heisenberg unitary layers +
# amplitude-damping dissipator layer per step), not their closed-system
# counterparts.
#
# Caveat carried over from the project notes: because S(dt)^† is not itself
# a physical channel once gamma>0, there is no guarantee Fij stays close to
# the identity, and the bond dimension of the resulting MPO may grow quickly
# with the number of layers (this is exactly what build_bond_dimension_trace
# below is meant to let you measure directly, rather than assume).

function build_open_F(n, J, gammas, t, k, lsites::LiouvilleSites, cutoff, maxdim; order::Int=1)
    Id = identity_liouville_mpo(lsites)
    F_components = MPO[]

    for i in eachindex(k)
        dt_i = t / k[i]
        step_i_dag_mpo = get_open_step_MPO_dag(n, J, gammas, dt_i, lsites, cutoff, maxdim; order=order)

        for j in eachindex(k)
            dt_j = t / k[j]
            step_j_mpo = get_open_step_MPO(n, J, gammas, dt_j, lsites, cutoff, maxdim; order=order)

            F = deepcopy(Id)
            time_i = 0.0
            time_j = 0.0

            while (time_i < t - 1e-12) || (time_j < t - 1e-12)
                if (time_j <= time_i) && (time_j < t - 1e-12)
                    # F <- F * S_j
                    F = right_multiply(F, step_j_mpo; cutoff=cutoff, maxdim=maxdim)
                    time_j += dt_j
                elseif time_i < t - 1e-12
                    # F <- S_i† * F
                    F = left_multiply(step_i_dag_mpo, F; cutoff=cutoff, maxdim=maxdim)
                    time_i += dt_i
                end
            end

            push!(F_components, F)
        end
    end

    return F_components
end

function build_open_F_between_lists(n, J, gammas, t, ks_left, ks_right, lsites::LiouvilleSites, cutoff, maxdim;
                                     order_left::Int=1, order_right::Int=1)
    Id = identity_liouville_mpo(lsites)
    F_components = MPO[]

    for i in eachindex(ks_left)
        dt_i = t / ks_left[i]
        step_i_dag_mpo = get_open_step_MPO_dag(n, J, gammas, dt_i, lsites, cutoff, maxdim; order=order_left)

        for j in eachindex(ks_right)
            dt_j = t / ks_right[j]
            step_j_mpo = get_open_step_MPO(n, J, gammas, dt_j, lsites, cutoff, maxdim; order=order_right)

            F = deepcopy(Id)
            time_i = 0.0
            time_j = 0.0

            while (time_i < t - 1e-12) || (time_j < t - 1e-12)
                if (time_j <= time_i) && (time_j < t - 1e-12)
                    F = right_multiply(F, step_j_mpo; cutoff=cutoff, maxdim=maxdim)
                    time_j += dt_j
                elseif time_i < t - 1e-12
                    F = left_multiply(step_i_dag_mpo, F; cutoff=cutoff, maxdim=maxdim)
                    time_i += dt_i
                end
            end

            push!(F_components, F)
        end
    end

    return F_components
end

# =============================================================================
# Building F_ex,j (Eq. 51) via middle-out contraction
# =============================================================================
#
# Fex,j = [e^{tL}]^†  S(t/kj)^kj
#
# As noted in main.pdf (and reproduced in build_F_between_lists for the
# closed-system code via k0 >> kj), in practice we approximate e^{tL} by a
# very fine-grained product formula S(t/k0)^{k0} with k0 >> kj, so this is
# just build_open_F_between_lists specialized to a single very large
# reference k0 on the left. We expose it as its own function purely for
# notational parity with Eq. 51 and to make the "this is the reference/exact
# evolution" role of k0 explicit at the call site.

function build_open_F_ex(n, J, gammas, t, ks, k0, lsites::LiouvilleSites, cutoff, maxdim;
                          order::Int=1, order_ref::Int=1)
    return build_open_F_between_lists(
        n, J, gammas, t, [k0], ks, lsites, cutoff, maxdim;
        order_left=order_ref, order_right=order
    )
end

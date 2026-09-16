using ITensors, ITensorMPS
using LinearAlgebra

# =============================================================================
# symmetric_splitting.jl
#
# A genuinely symmetric (palindromic) product formula for the vectorized
# Lindbladian, and a dispatcher that lets any caller choose between it and the
# existing project composition.
#
# -----------------------------------------------------------------------------
# THE BUG THIS FIXES
# -----------------------------------------------------------------------------
# get_open_step_gates_order2 in open_product_formula_generation.jl composes, in
# application order,
#
#     odd(dt/2) , even(dt) , diss(dt) , odd(dt/2)
#
# i.e. the operator e^{A dt/2} e^{C dt} e^{B dt} e^{A dt/2}  with A = odd
# Hamiltonian layer, B = even layer, C = dissipator layer. The outer A is
# symmetrized, but B and C are composed as a bare first-order product:
#
#     e^{C dt} e^{B dt} = exp( (B+C) dt + [C,B] dt^2/2 + ... )
#
# so the per-step error keeps an uncancelled [C,B] dt^2 term and the scheme is
# GLOBALLY FIRST ORDER whenever the dissipator and the even layer fail to
# commute -- which they do not, in general.
#
# At gamma = 0 the dissipator layer is EMPTY (dissipator_layer_channel_gates
# skips every site with gamma == 0), the composition collapses to plain Strang,
# and it is genuinely second order. That is why the entire closed-system half of
# this project never saw this.
#
# Consequence for order 4: get_open_step_gates_order4 applies the Yoshida
# triple-jump to five order-2 sub-steps. That construction reaches fourth order
# ONLY if the base is second order AND symmetric. Applied to the composition
# above it is not fourth order, and in practice returns roughly the base's own
# order.
#
# -----------------------------------------------------------------------------
# EVIDENCE
# -----------------------------------------------------------------------------
# (a) step0 log, n=6, gamma=0.05, UNTRUNCATED at the chi=64 ceiling. Self-
#     convergence ||rho(k0) - rho(768)|| / ||rho||:
#
#        k0        24       48       96      192      384
#        order 2   8.2e-3   3.1e-3   1.3e-3   5.4e-4   1.8e-4
#        order 4   6.2e-3   3.0e-3   1.4e-3   6.0e-4   2.0e-4
#
#     Ratio per doubling ~2.3 for BOTH. Order 2 should give 4, order 4 should
#     give 16. The two curves coinciding is the signature of the Yoshida
#     composition failing on a non-symmetric base.
#
# (b) Independent measurement in vectorized_evolution.jl (n=8, gamma=0.05,
#     maxdim=256 = exact ceiling), effective order from successive differences:
#
#        t           2      4      6     10
#        :project   1.6    1.6    2.3   sign flip
#        :strang    2.0    2.0    2.0   2.1
#
#     The sign flip at t=10 is the giveaway: a palindromic scheme expands in
#     even powers of dt only, so its error shrinks monotonically. A scheme with
#     both parities does not have to.
#
# -----------------------------------------------------------------------------
# THE FIX
# -----------------------------------------------------------------------------
#     odd(dt/2) , even(dt/2) , diss(dt) , even(dt/2) , odd(dt/2)
#
# Palindromic for any number of terms, hence S(-dt) = S(dt)^{-1}, hence the
# effective generator has no odd powers of dt and the leading error is dt^2.
# Costs one extra two-site layer per step.
#
# This is the same construction as strang_open_step_gates in
# vectorized_evolution.jl. Loading both files is harmless -- Julia redefines
# functions silently and the definitions are identical -- but there is no need
# to: this file is self-contained given the existing include chain.
#
# -----------------------------------------------------------------------------
# WHAT TO CHANGE IN THE PROJECT, AND WHAT NOT TO
# -----------------------------------------------------------------------------
# Minimal, low-disruption path: change the REFERENCE only.
#
#   * candidates rho_kj  -- leave on :project. DMPF corrects whatever formula
#                           the candidates use; the base order is not part of
#                           the claim, and keeping them preserves continuity
#                           with every number already computed.
#   * reference rho      -- switch to :strang (order 2 or 4). The reference has
#                           to be CONVERGED, and at 1/k0 that is unreachable:
#                           the step0 ladder showed E_mpf still ~1% off at
#                           k0 = 768, needing k0 ~ 2000 to hit 1e-3.
#
# The order/order_ref split already exists throughout the codebase, so this
# needs no structural change. If the whole project is later moved to :strang,
# note that get_open_step_gates_order2_dag must be changed to match -- it is
# currently the correct adjoint OF THE WRONG FORWARD, so the two are consistent
# with each other and the MOC results are internally sound, just built on an
# order-1 formula.
# =============================================================================


"""
    strang_open_step_gates(n, J, gammas, dt, lsites; order=2, dissipation=true)

Palindromic Strang splitting of the odd / even / dissipator layers:

    odd(dt/2), even(dt/2), diss(dt), even(dt/2), odd(dt/2)

`order=4` applies the Yoshida triple-jump to this base, which IS legitimate
here because the base is symmetric. Note that p2 < 0, so the dissipative
sub-step runs backwards and is not a CP map: harmless for accuracy at small dt
(the composition is still a valid approximation of the exact propagator), but
it can push Tr(rho) slightly above 1. The step0 log shows Tr = 1.00000316 at
k0 = 96 for the existing order-4 path, which is this effect.
"""
function strang_open_step_gates(n, J, gammas, dt, lsites::LiouvilleSites;
                                order::Int=2, dissipation::Bool=true)
    # Mirror get_open_step_gates: honour `dissipation` by zeroing the rates,
    # which dissipator_layer_channel_gates then skips site by site.
    eff = dissipation ? gammas : zeros(length(gammas))

    if order == 2
        return vcat(
            odd_layer_channel_gates(n, J, dt / 2, lsites),
            even_layer_channel_gates(n, J, dt / 2, lsites),
            dissipator_layer_channel_gates(n, eff, dt, lsites),
            even_layer_channel_gates(n, J, dt / 2, lsites),
            odd_layer_channel_gates(n, J, dt / 2, lsites),
        )
    elseif order == 4
        p1 = 1 / (4 - 4^(1/3))
        p2 = 1 - 4p1
        return vcat(
            strang_open_step_gates(n, J, gammas, p1 * dt, lsites; order=2, dissipation=dissipation),
            strang_open_step_gates(n, J, gammas, p1 * dt, lsites; order=2, dissipation=dissipation),
            strang_open_step_gates(n, J, gammas, p2 * dt, lsites; order=2, dissipation=dissipation),
            strang_open_step_gates(n, J, gammas, p1 * dt, lsites; order=2, dissipation=dissipation),
            strang_open_step_gates(n, J, gammas, p1 * dt, lsites; order=2, dissipation=dissipation),
        )
    end
    error("strang_open_step_gates supports order 2 or 4, got $order.")
end


"""
    split_step_gates(n, J, gammas, dt, lsites; order, dissipation, splitting)

Dispatch between `:project` (the existing `get_open_step_gates`, kept as the
default so nothing changes by accident) and `:strang` (the symmetric fix above).
"""
function split_step_gates(n, J, gammas, dt, lsites::LiouvilleSites;
                          order::Int=2, dissipation::Bool=true,
                          splitting::Symbol=:project)
    if splitting === :project
        return get_open_step_gates(n, J, gammas, dt, lsites;
                                   order=order, dissipation=dissipation)
    elseif splitting === :strang
        return strang_open_step_gates(n, J, gammas, dt, lsites;
                                      order=order, dissipation=dissipation)
    end
    error("splitting must be :project or :strang, got $splitting.")
end


"""
    split_step_MPO(n, J, gammas, dt, lsites, cutoff, maxdim; order, dissipation, splitting)

Splitting-aware replacement for `get_open_step_MPO`. Identical in every other
respect: contract the gate list into the identity MPO, truncating at each
`apply` exactly as the existing code does.
"""
function split_step_MPO(n, J, gammas, dt, lsites::LiouvilleSites, cutoff, maxdim;
                        order::Int=2, dissipation::Bool=true, splitting::Symbol=:project)
    gates = split_step_gates(n, J, gammas, dt, lsites;
                             order=order, dissipation=dissipation, splitting=splitting)
    S = identity_liouville_mpo(lsites)
    return apply(gates, S; cutoff=cutoff, maxdim=maxdim)
end

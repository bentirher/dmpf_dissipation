using ITensors, ITensorMPS
using LinearAlgebra
include("F_diagnostics.jl")   # pulls in the full include chain

# =============================================================================
# Open-system analogue of optimization_problem.jl
# =============================================================================
#
# Implements the Gram matrix M_ij, overlap vector L_j, and the MPF coefficient
# solve for the OPEN-SYSTEM case (Lindblad / amplitude-damping dynamics),
# following Eqs. 40-53 of main.pdf.
#
# KEY SEMANTIC DIFFERENCES FROM THE CLOSED-SYSTEM CODE:
#
# 1. M[i,j] = real(<<rho(0)|F_ij|rho(0)>>) -- real(), NOT abs2().
#    Eq. 46: M_ij = Tr(rho_ki rho_kj) = <<rho_ki|rho_kj>>, which is always
#    real because it is the trace of a product of two Hermitian operators.
#    The amplitude <<rho(0)|F_ij|rho(0)>> is mathematically real for the same
#    reason; any imaginary residual is floating-point noise and is discarded.
#    The closed system uses abs2() because M_ij there reduces to
#    |<psi(0)|S^{-ki}S^{kj}|psi(0)>|^2 (a squared modulus of a complex
#    amplitude), which coincidentally equals the real part for normalized
#    pure states -- but the open-system formula does NOT go through that
#    amplitude-squared step, so using abs2() here would be wrong.
#
# 2. L[j] = real(<<rho(0)|F_ex,j|rho(0)>>) -- same reasoning as M_ij.
#    Eq. 47: L_j = Tr(rho(t) rho_kj(t)) = <<rho(t)|rho_kj(t)>>, real.
#
# 3. The optimization problem (Eq. 40) and its linear-algebraic solution
#    (Eqs. 20-24) are STRUCTURALLY IDENTICAL to the closed system. Only the
#    physical content of M and L differs. So dynamic_mpf_coefficients is
#    reused unchanged.
#
# 4. Error expressions differ because Tr(rho^2) <= 1 for open systems (it
#    equals 1 only for pure states). See open_dynamic_mpf_error below.

# =============================================================================
# Gram matrix M_ij = Tr(rho_ki rho_kj) = <<rho(0)|F_ij|rho(0)>>  (Eq. 46+52)
# =============================================================================

function open_gram_matrix(n, J, gammas, t, ks, lsites::LiouvilleSites, rho0::MPS;
                           cutoff=1e-10, maxdim=200, order::Int=2, dissipation::Bool=true)
    r = length(ks)
    M = zeros(Float64, r, r)

    Fs = build_open_F(n, J, gammas, t, ks, lsites, cutoff, maxdim;
                      order=order, dissipation=dissipation)

    idx = 1
    for i in 1:r
        for j in 1:r
            # real() discards numerical noise in the imaginary part.
            # M_ij = Tr(rho_ki rho_kj) is guaranteed real by construction.
            M[i, j] = real(inner(rho0', Fs[idx], rho0))
            idx += 1
        end
    end

    return M, Fs   # return Fs too so the caller can reuse them for error estimation
end

# =============================================================================
# Overlap vector L_j = Tr(rho(t) rho_kj(t)) = <<rho(0)|F_ex,j|rho(0)>>  (Eq. 47+53)
# =============================================================================

function open_L_vector(n, J, gammas, t, ks, k_ref, lsites::LiouvilleSites, rho0::MPS;
                        cutoff=1e-10, maxdim=200,
                        order::Int=2, order_ref::Int=2, dissipation::Bool=true)
    # F_ex,j = (e^{tL})^dag S(t/kj)^{kj}
    # built via build_open_F_between_lists with ks_left=[k_ref] (the reference
    # approximate exact evolution) and ks_right=ks (the product formulas to compare).
    Fex_list = build_open_F_between_lists(
        n, J, gammas, t, [k_ref], ks, lsites, cutoff, maxdim;
        order_left=order_ref, order_right=order, dissipation=dissipation
    )

    L = zeros(Float64, length(ks))
    for j in eachindex(ks)
        L[j] = real(inner(rho0', Fex_list[j], rho0))
    end

    return L, Fex_list
end

# =============================================================================
# Purity Tr(rho^2(t)) -- needed for open-system error estimation (Eq. 39)
# =============================================================================
#
# Tr(rho^2(t)) = <<rho(t)|rho(t)>>
#              = ||S(t/k_ref)^{k_ref} |rho(0)>>||^2
#
# Computed cheaply by applying the k_ref-step forward channel MPO once to the
# initial state rho0, giving the approximate evolved vectorized state, then
# taking its squared norm. This avoids building an extra MPO contraction
# (which would cost O(chi^4) vs O(chi^2) for the MPS route).

function reference_purity(n, J, gammas, t, k_ref, lsites::LiouvilleSites, rho0::MPS;
                           cutoff=1e-10, maxdim=200, order::Int=2, dissipation::Bool=true)
    dt_ref = t / k_ref
    S_ref = get_open_step_MPO(n, J, gammas, dt_ref, lsites, cutoff, maxdim;
                               order=order, dissipation=dissipation)

    # Apply k_ref Trotter steps to rho0 to get the approximate |rho(t)>>
    rho_t = deepcopy(rho0)
    for _ in 1:k_ref
        rho_t = apply(S_ref, rho_t; cutoff=cutoff, maxdim=maxdim)
    end

    # Tr(rho^2(t)) = <<rho(t)|rho(t)>> = norm^2 of the vectorized state
    return real(inner(rho_t, rho_t))
end

# =============================================================================
# MPF coefficients -- UNCHANGED from closed system (Eqs. 20-24 / Eq. 40)
# =============================================================================
#
# The optimization problem has identical structure to the closed-system case:
# min c^T M c - 2 L^T c  s.t. sum(c) = 1, solved via the (r+1)x(r+1) block
# system with a Lagrange multiplier for the normalization constraint.
# Only M and L have different physical content (see above).

function dynamic_mpf_coefficients(M::AbstractMatrix, L::AbstractVector)
    r = size(M, 1)
    @assert size(M, 2) == r "M must be square"
    @assert length(L) == r "L must have the same length as the side of M"

    A = zeros(Float64, r + 1, r + 1)
    b = zeros(Float64, r + 1)

    A[1:r, 1:r] .= M
    A[1:r, r+1] .= -1.0
    A[r+1, 1:r] .= 1.0

    b[1:r] .= L
    b[r+1] = 1.0

    x = A \ b
    c = x[1:r]
    lambda = x[r+1]

    return c, lambda
end

# =============================================================================
# Error estimation  (Eq. 39 of main.pdf, open-system version)
# =============================================================================
#
# DMPF error:
#   E_mpf = ||rho(t) - mu(t)||^2_F
#         = Tr(rho^2) + c^T M c - 2 L^T c      (Eq. 39)
#
# Note: in the CLOSED system, Tr(rho^2) = 1 for all t, so the formula reduces
# to the familiar 1 + c^T M c - 2 L^T c (Eq. 7). In the open system,
# Tr(rho^2) <= 1 and is time-dependent -- it must be supplied from
# reference_purity(), called with the same high-accuracy reference parameters
# used for M_ref / L_ref below.

function open_dynamic_mpf_error(M::AbstractMatrix, L::AbstractVector,
                                 c::AbstractVector, purity_ref::Float64)
    return purity_ref + dot(c, M * c) - 2.0 * dot(L, c)
end

# Single-Trotter errors:
#   E_kj = ||rho(t) - rho_kj(t)||^2_F
#        = Tr(rho^2) + Tr(rho_kj^2) - 2 Tr(rho rho_kj)    (from Eq. 38)
#        = purity_ref + M[j,j]       - 2 * L[j]
#
# Note: Tr(rho_kj^2) = <<rho_kj|rho_kj>> = <<rho(0)|F_jj|rho(0)>> = M[j,j],
# which is the diagonal of the Gram matrix -- already computed, no extra cost.

function open_single_trotter_errors(M::AbstractMatrix, L::AbstractVector,
                                     purity_ref::Float64)
    r = length(L)
    return [purity_ref + M[j, j] - 2.0 * L[j] for j in 1:r]
end

# =============================================================================
# Main wrapper: compute M, L, c, and all errors in one call
# =============================================================================
#
# Mirrors test_dynamic_mpf_closed in error_estimation.jl, adapted for the
# open system. Two sets of M/L are computed:
#   - "opt": cheap (small maxdim, low-order or coarse k_ref) -- used to
#     find the coefficients c
#   - "ref": accurate (large maxdim, fine k_ref, high order) -- used to
#     evaluate the errors with the fixed c from the opt step
#
# The purity Tr(rho^2(t)) is always computed at "ref" accuracy since it enters
# the error expressions, not the optimization.

function test_dynamic_mpf_open(
    n, J, gammas, t, ks, k_ref_opt, k_ref_eval, lsites::LiouvilleSites, rho0::MPS;
    cutoff_opt=1e-10,  maxdim_opt=50,
    cutoff_eval=1e-10, maxdim_eval=200,
    order=2,       order_ref_opt=2,
    order_ref_eval=2,
    dissipation::Bool=true
)
    # --- Optimization step (coarse) ---
    M_opt, _ = open_gram_matrix(
        n, J, gammas, t, ks, lsites, rho0;
        cutoff=cutoff_opt, maxdim=maxdim_opt,
        order=order, dissipation=dissipation
    )
    L_opt, _ = open_L_vector(
        n, J, gammas, t, ks, k_ref_opt, lsites, rho0;
        cutoff=cutoff_opt, maxdim=maxdim_opt,
        order=order, order_ref=order_ref_opt, dissipation=dissipation
    )

    c, lambda = dynamic_mpf_coefficients(M_opt, L_opt)

    # --- Reference evaluation step (accurate) ---
    M_ref, _ = open_gram_matrix(
        n, J, gammas, t, ks, lsites, rho0;
        cutoff=cutoff_eval, maxdim=maxdim_eval,
        order=order, dissipation=dissipation
    )
    L_ref, _ = open_L_vector(
        n, J, gammas, t, ks, k_ref_eval, lsites, rho0;
        cutoff=cutoff_eval, maxdim=maxdim_eval,
        order=order, order_ref=order_ref_eval, dissipation=dissipation
    )

    purity = reference_purity(
        n, J, gammas, t, k_ref_eval, lsites, rho0;
        cutoff=cutoff_eval, maxdim=maxdim_eval,
        order=order_ref_eval, dissipation=dissipation
    )

    E_mpf  = open_dynamic_mpf_error(M_ref, L_ref, c, purity)
    E_trot = open_single_trotter_errors(M_ref, L_ref, purity)

    return (
        M_opt    = M_opt,
        L_opt    = L_opt,
        coeffs   = c,
        lambda   = lambda,
        M_ref    = M_ref,
        L_ref    = L_ref,
        purity   = purity,
        E_mpf    = E_mpf,
        E_trot   = E_trot,
    )
end

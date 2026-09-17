using ITensors, ITensorMPS
using LinearAlgebra

include("F_diagnostics.jl")   # top of the include chain -- pulls in, in order:
                              #   F_diagnostics.jl            vectorized_initial_state_mps, expect_F
                              #   bond_dimension_tracking.jl  middle_bond_dim
                              #   open_middle_out_contraction op_dag, left/right_multiply, build_open_F
                              #   open_product_formula_gen    get_open_step_MPO(_dag)
                              #   liouville_space.jl          liouville_siteinds, identity_liouville_mpo, ID2, SIGMA_Z
                              #
                              # Same include discipline as trotter_error_gram.jl: do NOT include
                              # open_middle_out_contraction.jl directly.
include("symmetric_splitting.jl")   # split_step_MPO, :project vs :strang.
                                    # MUST come after the chain above: it uses
                                    # odd/even/dissipator_layer_channel_gates and
                                    # identity_liouville_mpo.

# =============================================================================
# liouville_state_tools.jl
#
# Everything needed to run the DMPF pipeline on the STATE route: evolve
# |rho>> as a Liouville MPS, take overlaps, and evaluate Tr(O rho).
#
# WHY THIS FILE EXISTS
# --------------------
# The MOC / MPO machinery (trotter_error_gram.jl) computes N without ever
# forming a state. That was the right thing to build when the goal was an
# ASYMPTOTIC efficiency claim, because a state-based route was disqualified in
# principle ("if you can evolve rho classically you don't need the QPU").
#
# The claim we are now testing is different -- accuracy at MATCHED classical
# cost, in the sense of Robertson et al., arXiv:2609.05024 Fig. 4 -- and under
# that claim the state route is not disqualified. It is the natural production
# route, because:
#
#   chi(MPO) scales as chi(MPS)^2, and the Liouville MPS ceiling 4^(n/2) is the
#   square root of the Liouville MPO ceiling 16^(n/2). At n = 8 that is 256 vs
#   65536. The state route reaches n = 8-12; the MPO route dies at n = 6.
#
# So Steps 0 and 1 use ONLY this file plus the existing gate/step-MPO
# construction. trotter_error_gram.jl is not needed and is not included: it
# becomes the small-n validation oracle, which is what it is genuinely good at.
#
# NUMERICAL RULES INHERITED FROM THE PROJECT README (each was a real bug)
# ----------------------------------------------------------------------
#  1. Differences of near-equal MPS/MPO MUST use alg="directsum". The default
#     density-matrix `+` forms rho = M M^dag and diagonalises, so it carries
#     only sqrt(eps_mach) ~ 1.5e-8 accuracy and silently discards the very
#     difference being measured. Delta_j = rho_kj - rho_ref is exactly such a
#     difference (||Delta||/||rho|| ~ 1e-2), so mps_difference below uses
#     directsum unconditionally.
#
#     NOTE: this is also a (mild) bug in the existing
#     validate_N_against_direct in trotter_error_gram.jl, which builds the
#     error vectors with `+(rho_j, -1*rho_ref; cutoff=cutoff, maxdim=maxdim)`.
#     Low risk there because the cutoff is relative to ||Delta|| itself, but
#     there is no reason to accept it: directsum is free when the only thing
#     done with Delta afterwards is an inner product.
#
#  2. maxdim must never be set above the ceiling for the object in question.
#     For a Liouville MPS that is 4^min(l, n-l), NOT the 16^min(l, n-l) MPO
#     ceiling used by theoretical_max_bond_dim. Use state_max_bond_dim.
#
#  3. BLAS threads must be pinned in the submit script in BOTH places.
# =============================================================================


# -----------------------------------------------------------------------------
# Bond-dimension ceilings
# -----------------------------------------------------------------------------
#
# theoretical_max_bond_dim (trotter_error_gram.jl) is the MPO ceiling,
# 16^min(l, n-l). A Liouville MPS carries one factor of 4 per site, not 16, so
# its ceiling is the square root of that. At this bond dimension the MPS is
# EXACT: no truncation is possible, which is what makes small n usable as
# ground truth.

state_max_bond_dim(n::Int) = 4^min(n ÷ 2, n - n ÷ 2)

# The MPO ceiling, 16^min(l, n-l). Identical to theoretical_max_bond_dim in
# trotter_error_gram.jl, defined here under a different name so these scripts
# do NOT have to include that file (it pulls in the whole four-object recursion,
# which Steps 0 and 1 do not use). Only needed to cap the step-channel MPO.
mpo_max_bond_dim(n::Int) = 16^min(n ÷ 2, n - n ÷ 2)


# -----------------------------------------------------------------------------
# Observables as Liouville MPS
# -----------------------------------------------------------------------------
#
# Convention check (liouville_space.jl): |rho>> has ket = row index, bra =
# column index, so the component at (ket=a, bra=b) is rho_ab.
#
# Build |O>> with the SAME convention, component (ket=a, bra=b) = O_ab. Then
#
#     inner(O_mps, rho_mps) = sum_ab conj(O_ab) rho_ab = Tr(O^dag rho)
#                           = Tr(O rho)   for Hermitian O.
#
# ITensors' `inner(x, y)` conjugates its FIRST argument, which is why the
# observable goes first. Every observable used in this project is Hermitian, so
# the dagger is harmless -- but do not reuse this helper for non-Hermitian O
# without putting the conjugation back by hand.

function observable_mps(lsites::LiouvilleSites, ops::Dict{Int,<:AbstractMatrix})
    n = lsites.n
    links = [Index(1, "Link,l=$j") for j in 1:(n-1)]
    tensors = ITensor[]
    for j in 1:n
        M = Matrix{ComplexF64}(get(ops, j, ID2))
        # itensor(A, i, j) fills T[a,b] = A[a,b] with the index order given.
        t = itensor(M, lsites.ket[j], lsites.bra[j])
        j > 1 && (t *= onehot(links[j-1] => 1))
        j < n && (t *= onehot(links[j] => 1))
        push!(tensors, t)
    end
    return MPS(tensors)
end

z_observable(lsites::LiouvilleSites, m::Int) =
    observable_mps(lsites, Dict{Int,Matrix{ComplexF64}}(m => SIGMA_Z))

zz_observable(lsites::LiouvilleSites, m1::Int, m2::Int) =
    observable_mps(lsites, Dict{Int,Matrix{ComplexF64}}(m1 => SIGMA_Z, m2 => SIGMA_Z))

# |1>> -- the vectorized identity. <<1|rho>> = Tr(rho), which must equal 1 for
# any trace-preserving channel. This is the cheapest end-to-end check that the
# vectorization, the gate construction and the dissipator sign are all right,
# and it costs one inner product. Both step scripts print it.
identity_observable(lsites::LiouvilleSites) =
    observable_mps(lsites, Dict{Int,Matrix{ComplexF64}}())

expval(O::MPS, rho::MPS) = inner(O, rho)


# -----------------------------------------------------------------------------
# Differences of MPS
# -----------------------------------------------------------------------------
#
# Rule 1 above. directsum gives chi(a) + chi(b) with nothing discarded, which is
# exactly right when the only downstream operation is an inner product.

mps_difference(a::MPS, b::MPS) = +(a, -1 * b; alg="directsum")


# -----------------------------------------------------------------------------
# Truncation with MEASURED discarded weight
# -----------------------------------------------------------------------------
#
# We need the truncation error eps_chi as defined in arXiv:2609.05024: the sum
# over steps of the squared singular values discarded. Rather than reach into
# ITensors' Spectrum object (whose field names are not stable across versions),
# we measure it by norms, which is exact and API-stable:
#
#   truncate! puts the MPS in canonical form and removes a component ORTHOGONAL
#   to what it keeps. Orthogonal components add in quadrature, so
#
#       discarded weight = ||psi||^2_before - ||psi||^2_after
#
#   exactly, with no approximation.
#
# The norms are taken in left-canonical form so each costs O(chi^2), not a full
# O(n chi^3) contraction: after orthogonalize!(psi, 1) every tensor except the
# first is right-orthogonal, so ||psi|| = ||psi[1]||.

function _norm2_canonical!(psi::MPS)
    orthogonalize!(psi, 1)
    return norm(psi[1])^2
end

# ==============================================================================
# THE CUTOFF TRAP -- read this before changing any cutoff in this file
# ==============================================================================
# ITensors' `cutoff` bounds the sum of the SQUARES of the discarded singular
# values, relative to the total. So a cutoff of eps permits a discarded
# AMPLITUDE of sqrt(eps):
#
#     cutoff = 1e-16   ->   state error ~1e-8 PER TRUNCATION
#
# That is not machine precision, it is the square root of it. ITensors' own
# default is 0.0 (exact), so passing 1e-16 is strictly worse than passing
# nothing.
#
# MEASURED, from the step0 gate-mode log (n=6, gamma=0.05, untruncated at the
# chi=64 ceiling, strang:4). Raw trace deviation |1 - Tr(rho)|:
#
#     k0        24        48        96       192       384       768
#     |1-Tr|  2.65e-7   5.43e-7   1.11e-6   2.41e-6   5.15e-6   1.06e-5
#
# Ratios 2.05, 2.04, 2.17, 2.14, 2.06 -- LINEAR in k0, i.e. a fixed per-step
# error of ~1.4e-8 = sqrt(1e-16) accumulating coherently over the steps. Every
# gate is exactly trace-preserving, so this is entirely the cutoff.
#
# It set the ~1e-5 floor that made strang:4's p_eff unreadable on the main
# ladder, and it is why strang:4's E_mpf drifted monotonically downward
# (2.6870e-4 -> 2.6859e-4) across a ladder on which it was already converged.
#
# Hence the default below. 1e-32 is used rather than 0.0 only to avoid any
# edge-case handling of an exactly-zero cutoff; it corresponds to a discarded
# amplitude of 1e-16, i.e. genuinely at round-off. Pass CUTOFF explicitly if you
# want to study the effect.
#
# CONSEQUENCE FOR THE GOLD REFERENCE: for a converged high-order scheme the
# Trotter error falls as k0^-p while this numerical error GROWS as k0^1, so the
# best reference is the SMALLEST k0 that is Trotter-converged, not the largest.
# Scoring against rho(k0_max) scores against the worst-conditioned state on the
# ladder.
# ==============================================================================

const EXACT_CUTOFF = 1e-32

function truncate_tracked!(psi::MPS; maxdim::Int, cutoff::Float64=EXACT_CUTOFF)
    nrm2_before = _norm2_canonical!(psi)
    truncate!(psi; maxdim=maxdim, cutoff=cutoff)
    nrm2_after = _norm2_canonical!(psi)
    d = nrm2_before - nrm2_after
    # Small negative values are round-off on two O(1) numbers; anything large
    # and negative means truncate! renormalised, which would invalidate the
    # measurement, so fail loudly rather than silently reporting eps = 0.
    @assert d > -1e-8 * max(nrm2_before, 1.0) "truncate! increased the norm by $(-d); the norm-difference estimator is invalid for this ITensors version."
    return max(d, 0.0)
end


# -----------------------------------------------------------------------------
# Tracked evolution of a Liouville MPS
# -----------------------------------------------------------------------------
#
# Applies the step MPO `nsteps` times, truncating to `maxdim` after each step
# and accumulating the discarded weight.
#
# The `apply` itself is done at the exact cap chi_psi * chi_S (bounded by the
# state ceiling) with a machine-level cutoff, so that ALL of the loss is
# concentrated in the subsequent truncate_tracked! call and is therefore
# measured. If `apply` were allowed to truncate as well, eps_chi would be an
# undercount and the x-axis of the headline figure would be wrong.

function evolve_tracked(rho0::MPS, S::MPO, nsteps::Int;
                        n::Int, maxdim::Int, cutoff::Float64=EXACT_CUTOFF)
    ceil_ = state_max_bond_dim(n)
    @assert maxdim <= ceil_ "maxdim=$maxdim exceeds the n=$n Liouville MPS ceiling $ceil_"
    cap = min(ceil_, maxdim * maxlinkdim(S))

    psi = deepcopy(rho0)
    eps_acc = 0.0
    for _ in 1:nsteps
        psi = apply(S, psi; cutoff=cutoff, maxdim=cap)
        eps_acc += truncate_tracked!(psi; maxdim=maxdim, cutoff=cutoff)
    end
    return psi, eps_acc
end


# -----------------------------------------------------------------------------
# Tracked evolution by DIRECT GATE APPLICATION (the default)
# -----------------------------------------------------------------------------
#
# Same contract as evolve_tracked, but the per-step gate list is applied
# straight to the MPS instead of being compressed into a step MPO first.
#
# WHY THIS IS THE DEFAULT, NOT THE MPO ROUTE
# ------------------------------------------
# The step-MPO route pays a compression cost that grows with the DEPTH of the
# product formula, and for a deep formula that cost is not just time -- it is
# accuracy. Concretely, at n=6 the middle cut is bond 3, an ODD bond, so only
# the odd layers cross it:
#
#     strang:2   2 odd half-layers   -> rank <= 16^2  = 256   (matches the
#                                                              measured chi_S)
#     strang:4   5 blocks x 2        -> rank <= 16^10, capped by the n=6 MPO
#                                       ceiling 4096
#
# With mpo_maxdim defaulting to 512, the strang:4 step MPO is truncated, and
# that truncation is the most likely source of the ~5e-6 relative floor seen in
# the step0 v2 log (strang:4 self-convergence bouncing around 4e-6 to 1e-5 with
# no trend, and its raw trace wandering over 4.4e-6 with no trend -- the same
# number).
#
# Applying 25 two-site layers to an MPS is both cheaper and more accurate than
# compressing them into a rank-4096 operator and then applying that. There is no
# reuse argument either: the step MPO is reused across k0 steps, but a two-site
# gate list is just as reusable and costs nothing to hold.
#
# The gates are applied at the state CEILING, so the application itself is
# exact and ALL of the loss stays in the subsequent tracked truncation. That
# keeps eps_chi an honest measurement, which is the whole point of the Step 1
# x-axis. It also means this does not scale past the n where the ceiling is
# affordable -- which is fine, because Steps 0 and 1 need an exact reference
# anyway and so are capped at n <= 10 regardless.

function evolve_tracked_gates(rho0::MPS, gates::Vector{ITensor}, nsteps::Int;
                              n::Int, maxdim::Int, cutoff::Float64=EXACT_CUTOFF)
    ceil_ = state_max_bond_dim(n)
    @assert maxdim <= ceil_ "maxdim=$maxdim exceeds the n=$n Liouville MPS ceiling $ceil_"

    psi = deepcopy(rho0)
    eps_acc = 0.0
    for _ in 1:nsteps
        psi = apply(gates, psi; cutoff=cutoff, maxdim=ceil_)
        eps_acc += truncate_tracked!(psi; maxdim=maxdim, cutoff=cutoff)
    end
    return psi, eps_acc
end


# -----------------------------------------------------------------------------
# One call: rho_k(t) for a given number of Trotter steps
# -----------------------------------------------------------------------------
#
# `splitting` selects the product formula (see symmetric_splitting.jl):
#   :project  the existing get_open_step_gates. Order 2 is really order 1 once
#             gamma > 0, and order 4 is not order 4. Kept as the default so the
#             candidates can stay exactly as they have always been.
#   :strang   palindromic, genuinely order 2 (and genuinely order 4). Use this
#             for the REFERENCE, which has to be converged.
#
# `renormalize_trace` divides out Tr(rho) after the evolution. Every gate is
# exactly trace-preserving, so any deviation from 1 is pure accumulated `apply`
# error -- the step0 log shows it growing monotonically to 1.1e-6 at k0 = 768,
# about 1.4e-9 per step, which is a ~1% contamination of quantities of order
# 1e-4. Dividing it out is strictly an improvement, and the raw trace is
# returned either way so it can be watched rather than hidden.

# `mode` selects how the per-step propagator is applied:
#   :gates  (DEFAULT) apply the two-site gate list straight to the MPS. Cheaper
#           and more accurate for deep formulas; see evolve_tracked_gates.
#   :mpo    compress the gate list into a step MPO first, as the rest of the
#           codebase does. Kept so the two can be compared directly, and because
#           the MOC route needs the MPO anyway.
# `chi_S` in the returned tuple is the step-MPO bond dimension in :mpo mode and
# the NUMBER OF GATES in :gates mode -- reported either way so a truncated step
# operator cannot hide.

function evolve_trotter(n, J, gammas, t::Float64, k::Int, lsites::LiouvilleSites,
                        rho0::MPS; maxdim::Int, order::Int=2, dissipation::Bool=true,
                        cutoff::Float64=EXACT_CUTOFF, mpo_maxdim::Int=512,
                        splitting::Symbol=:project, mode::Symbol=:gates,
                        renormalize_trace::Bool=true, id_mps::Union{MPS,Nothing}=nothing)

    local psi, eps, chiS, truncated
    if mode === :gates
        gates = split_step_gates(n, J, gammas, t / k, lsites;
                                 order=order, dissipation=dissipation, splitting=splitting)
        psi, eps = evolve_tracked_gates(rho0, gates, k; n=n, maxdim=maxdim, cutoff=cutoff)
        chiS, truncated = length(gates), false
    elseif mode === :mpo
        cap = min(mpo_maxdim, mpo_max_bond_dim(n))
        S = split_step_MPO(n, J, gammas, t / k, lsites, cutoff, cap;
                           order=order, dissipation=dissipation, splitting=splitting)
        psi, eps = evolve_tracked(rho0, S, k; n=n, maxdim=maxdim, cutoff=cutoff)
        chiS = maxlinkdim(S)
        # A step MPO sitting exactly at the cap has been truncated, and that
        # truncation is a silent systematic on every step of the evolution.
        truncated = chiS >= cap
    else
        error("mode must be :gates or :mpo, got $mode")
    end

    idm = id_mps === nothing ? identity_observable(lsites) : id_mps
    tr = inner(idm, psi)
    if renormalize_trace && abs(tr) > 1e-12
        psi[1] = psi[1] / tr
    end
    return (rho=psi, eps=eps, chi_S=chiS, chi=maxlinkdim(psi), trace=tr,
            step_truncated=truncated, mode=mode)
end


# -----------------------------------------------------------------------------
# Trotter-error Gram matrix from state overlaps
# -----------------------------------------------------------------------------
#
# N_ij = Tr[(rho_ki - rho)(rho_kj - rho)] = <<Delta_i|Delta_j>>.
#
# Identical object to trotter_error_gram() in trotter_error_gram.jl, obtained
# from states instead of from the four-object MPO recursion. Real and symmetric
# PSD by construction; symmetrised explicitly to kill asymmetric round-off.

function trotter_error_gram_from_states(rhos::Vector{MPS}, rho_ref::MPS)
    r = length(rhos)
    dvecs = [mps_difference(rhos[j], rho_ref) for j in 1:r]
    N = zeros(Float64, r, r)
    for i in 1:r, j in i:r
        v = real(inner(dvecs[i], dvecs[j]))
        N[i, j] = v
        N[j, i] = v
    end
    N .= 0.5 .* (N .+ N')
    return N, dvecs
end


# -----------------------------------------------------------------------------
# Coefficients from N  (re-implemented here so these scripts do not need to
# include trotter_error_gram.jl)
# -----------------------------------------------------------------------------
#
#   minimize c' N c   s.t.  sum(c) = 1
#   =>  c = N^{-1} 1 / (1' N^{-1} 1),   E_mpf = 1 / (1' N^{-1} 1)
#
# Identical to trotter_error_coefficients() in trotter_error_gram.jl, including
# the eigenvalue floor (N is PSD in exact arithmetic, so anything at or below
# the floor is truncation noise).

function coefficients_from_N(N::AbstractMatrix; rel_floor::Float64=1e-12)
    r = size(N, 1)
    Nsym = 0.5 * (N + N')
    F = eigen(Symmetric(Nsym))
    lam = F.values
    lam_max = maximum(abs, lam)
    floorval = rel_floor * lam_max
    lam_reg = max.(lam, floorval)

    ones_v = ones(Float64, r)
    y = F.vectors' * ones_v
    Ninv_1 = F.vectors * (y ./ lam_reg)
    denom = dot(ones_v, Ninv_1)

    # A cond of exactly 1/rel_floor means lam_min was CLIPPED: N is numerically
    # singular and both E_mpf and the coefficients are fictions below the
    # resolution of the Gram matrix that produced them. n_clipped and lam_min
    # are returned so callers can say so out loud instead of quoting 1e-13.
    return (coeffs=Ninv_1 ./ denom,
            E_mpf=1.0 / denom,
            lam_min=minimum(lam),
            singular=(minimum(lam) <= floorval),
            E_trot=[Nsym[j, j] for j in 1:r],
            eigvals=lam,
            eigvecs=F.vectors,
            cond=lam_max / max(minimum(lam), floorval),
            n_clipped=count(<(floorval), lam))
end


# -----------------------------------------------------------------------------
# The exact propagation identity that Step 1 is built around
# -----------------------------------------------------------------------------
#
#   E(c) = || sum_j c_j rho_kj - rho ||_F^2 = c' N c    (using sum(c) = 1)
#
# At the constrained minimum, N c* = E_mpf * 1. Hence for ANY perturbation
# with 1'dc = 0 (which every coefficient error satisfies, since both c and
# c+dc are normalised):
#
#   E(c* + dc) = E_mpf + 2 dc' N c* + dc' N dc
#              = E_mpf + 2 E_mpf (1'dc) + dc' N dc
#              = E_mpf + dc' N dc.
#
# The cross term vanishes IDENTICALLY -- no expansion, no small-dc assumption.
# So the damage done by a coefficient error is dc' N dc, not |dc|, and the
# observable error obeys
#
#   |sum_j dc_j <O>_kj| = |Tr(O sum_j dc_j Delta_j)| <= ||O||_F sqrt(dc' N dc).
#
# This is the rigorous form of the argument in arXiv:2609.05024 Appendix D. It
# matters because dc is amplified along the small-eigenvalue directions of N
# (alpha_i ~ p_i / lambda_i), so dc' N dc = sum_i p_i^2 / lambda_i: truncation
# error reaches the ANSWER amplified by 1/lambda_min, but reaches |dc|
# amplified by 1/lambda_min^2. One full power of the condition number is
# returned for free. With cond(N) ~ 800, that is the factor the project has
# been charging itself by scoring on max|dc|.

induced_error(N::AbstractMatrix, dc::AbstractVector) = dot(dc, Symmetric(0.5 * (N + N')) * dc)


# -----------------------------------------------------------------------------
# Truncation-error correlation -- the MECHANISM diagnostic
# -----------------------------------------------------------------------------
#
# The whole hypothesis is that N survives truncation better than <O> does
# because the errors made on rho_ki and on rho_ref are nearly the SAME error,
# so they cancel in the difference. This measures that directly:
#
#   corr = Re <drho_a, drho_b> / (||drho_a|| ||drho_b||),  drho = rho(chi) - rho(exact)
#
# corr near 1 confirms the cancellation and turns an observed effect into an
# explained one. corr near 0 means the effect, if present, has another cause --
# and that the method will not survive to longer t or larger gamma.

function truncation_error_correlation(rho_chi::MPS, rho_exact::MPS,
                                      sigma_chi::MPS, sigma_exact::MPS)
    da = mps_difference(rho_chi, rho_exact)
    db = mps_difference(sigma_chi, sigma_exact)
    na = sqrt(max(real(inner(da, da)), 0.0))
    nb = sqrt(max(real(inner(db, db)), 0.0))
    (na < 1e-300 || nb < 1e-300) && return (corr=NaN, norm_a=na, norm_b=nb)
    return (corr=real(inner(da, db)) / (na * nb), norm_a=na, norm_b=nb)
end

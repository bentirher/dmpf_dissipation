using ITensors, ITensorMPS
using LinearAlgebra
include("F_diagnostics.jl")   # top of the include chain -- pulls in, in order:
                              #   F_diagnostics.jl            vectorized_initial_state_mps, expect_F
                              #   bond_dimension_tracking.jl  middle_bond_dim
                              #   open_middle_out_contraction op_dag, left/right_multiply, build_open_F
                              #   open_product_formula_gen    get_open_step_MPO(_dag)
                              #   liouville_space.jl          liouville_siteinds, identity_liouville_mpo
                              #
                              # Do NOT include open_middle_out_contraction.jl directly: it sits BELOW
                              # F_diagnostics and bond_dimension_tracking in the chain, so
                              # vectorized_initial_state_mps and middle_bond_dim would be undefined.

# =============================================================================
# trotter_error_gram.jl
#
# Route-1 reformulation: build the TROTTER-ERROR Gram matrix
#
#     N_ij = Tr[ (rho_ki - rho) (rho_kj - rho) ] = <<rho0| Phi_ij |rho0>>
#     Phi_ij = Delta_i^dag Delta_j ,   Delta_j = S(t/k_j)^{k_j} - E(t)
#
# instead of the raw Gram matrix M_ij = <<rho0| F_ij |rho0>>, F_ij = S_i^dag S_j.
#
# WHY (see project notes sections 6-8): F_ij = G + Xi_i + Psi_j + Phi_ij, where
# G = E^dag E is O(1) and carries >99% of ||F||_F (measured: s_1 = 16.3 vs
# s_2 = 0.78 at n=5, layer 8), while the quantity that determines the DMPF
# coefficients is O(Trotter error^2) ~ 2e-4. Truncating F at relative level eta
# therefore gives relative error ~ eta / 2e-4 ~ 5000*eta on the answer. That
# amplification is the root cause of the negative E_mpf (section 7) and the
# E_k3 < E_k8 inversion (section 8) -- both are structurally impossible here,
# since N is a Gram matrix of the error vectors and hence PSD by construction.
#
# The four objects below close under the SAME two-clock schedule already used by
# build_open_F, so this is a drop-in replacement for open_gram_matrix +
# open_L_vector + reference_purity, not a new algorithm class.
#
# Objects tracked at independent left/right clocks (t_a, t_b):
#
#     G      = E(t_a)^dag E(t_b)          O(1)          <- needs LEAST accuracy
#     Xi_i   = Delta_i(t_a)^dag E(t_b)    O(eps)
#     Psi_j  = E(t_a)^dag Delta_j(t_b)    O(eps)
#     Phi_ij = Delta_i(t_a)^dag Delta_j(t_b)  O(eps^2)  <- the answer
#
# Update rules (derived from Delta_j(t+dt) = Delta_j(t) A_j + E(t) delta_j):
#
#   advance right clock by one A_j:        advance left clock by one A_i:
#     Phi <- Phi A_j + Xi delta_j            Phi <- A_i^dag Phi + delta_i^dag Psi
#     Psi <- Psi A_j + G  delta_j            Psi <- B_i^dag Psi
#     Xi  <- Xi  B_j                         Xi  <- A_i^dag Xi + delta_i^dag G
#     G   <- G   B_j                         G   <- B_i^dag G
#
# with A_j = S(t/k_j), B_j = S(t/k0)^{k0/k_j} (reference over the SAME duration),
# delta_j = A_j - B_j (a single-step defect: short time interval, low rank).
#
# REQUIREMENT: k0 must be an integer multiple of every k_j. With ks=[3,8],
# k0 = 48 or 96 works (the existing code already uses k_ref = 40 / 100).
# =============================================================================


# -----------------------------------------------------------------------------
# Zero-MPO handling
# -----------------------------------------------------------------------------
# Phi, Psi, Xi all start at exactly zero. Rather than carry an explicit
# all-zeros MPO (which wastes an apply per step and pollutes the relative
# cutoff), we use `nothing` as the zero element and define arithmetic on it.

# -----------------------------------------------------------------------------
# Bond-dimension ceiling (findings_thematic.md section 4)
# -----------------------------------------------------------------------------
# Each Liouville site has local operator dimension d = 4, so a cut with l sites
# on the small side admits Schmidt rank at most d^(2 min(l, n-l)) = 16^min(...).
#
# Passing `apply` a maxdim ABOVE this is actively harmful: it forces enormous
# intermediate tensors to be built and SVD-factorized before truncation, even
# though the final rank can never exceed the ceiling. This is the bug that
# turned an n=5 case from ">8 h, timed out" into "~4 minutes".

theoretical_max_bond_dim(n::Int) = 16^min(n ÷ 2, n - n ÷ 2)

const MaybeMPO = Union{MPO,Nothing}

_lmul(A::MPO, X::Nothing; kwargs...) = nothing
_lmul(A::MPO, X::MPO; cutoff, maxdim) = left_multiply(A, X; cutoff=cutoff, maxdim=maxdim)

_rmul(X::Nothing, A::MPO; kwargs...) = nothing
_rmul(X::MPO, A::MPO; cutoff, maxdim) = right_multiply(X, A; cutoff=cutoff, maxdim=maxdim)

_add(X::Nothing, Y::Nothing; kwargs...) = nothing
_add(X::MPO, Y::Nothing; kwargs...) = X
_add(X::Nothing, Y::MPO; kwargs...) = Y
_add(X::MPO, Y::MPO; cutoff, maxdim) = +(X, Y; cutoff=cutoff, maxdim=maxdim)

_fnorm(X::Nothing) = 0.0
_fnorm(X::MPO) = sqrt(abs(real(inner(X, X))))

_expect(X::Nothing, rho0::MPS) = 0.0 + 0.0im
_expect(X::MPO, rho0::MPS) = expect_F(X, rho0)   # = inner(rho0', X, rho0), F_diagnostics.jl


# -----------------------------------------------------------------------------
# Reference propagator over a given duration, and the single-step defect
# -----------------------------------------------------------------------------
#
# B_j = S(t/k0)^{q_j} with q_j = k0/k_j: the fine-grained reference evolution
# over exactly the duration of one coarse step A_j = S(t/k_j).
#
# This is built by repeated `apply` of the fine step MPO starting from the
# identity. It is a propagator over a SHORT time (t/k_j), so its bond dimension
# is modest -- unlike the full-time propagator, which is what the closed-system
# paper's Appendix A shows grows exponentially in t and which we never form.

function reference_block_MPO(n, J, gammas, duration::Float64, q::Int,
                             lsites::LiouvilleSites, cutoff, maxdim;
                             order::Int=2, dissipation::Bool=true)
    @assert q >= 1 "q = k0/k_j must be a positive integer"
    tau = duration / q
    fine = get_open_step_MPO(n, J, gammas, tau, lsites, cutoff, maxdim;
                             order=order, dissipation=dissipation)
    B = deepcopy(fine)
    for _ in 2:q
        B = left_multiply(fine, B; cutoff=cutoff, maxdim=maxdim)
    end
    return B
end

# delta_j = A_j - B_j, formed ONCE per k_j at high precision.
#
# Forming delta explicitly (rather than computing X*delta = X*A - X*B on the fly)
# matters: it confines the A-minus-B cancellation to a single short-time
# subtraction where A and B differ at relative order dt^p, instead of repeating
# that cancellation against every running object at every step.

function step_defect_MPO(n, J, gammas, dt_coarse::Float64, q::Int,
                         lsites::LiouvilleSites, cutoff, maxdim;
                         order::Int=2, order_ref::Int=2, dissipation::Bool=true,
                         exact_delta::Bool=true, verify::Bool=false)
    A = get_open_step_MPO(n, J, gammas, dt_coarse, lsites, cutoff, maxdim;
                          order=order, dissipation=dissipation)
    B = reference_block_MPO(n, J, gammas, dt_coarse, q, lsites, cutoff, maxdim;
                            order=order_ref, dissipation=dissipation)

    # The recursion relies on B + delta == A EXACTLY: every update rule was
    # derived by substituting A_j = B_j + delta_j, e.g.
    #     G' + Xi' + Psi' + Phi' = (G+Xi)(B_j + delta_j) + (Psi+Phi)A_j
    # collapses to F A_j only if that identity holds numerically. Compressing
    # delta breaks it, and the resulting defect propagates into every step.
    #
    # exact_delta=true keeps the direct-sum bond dimension, so B + delta = A
    # holds by construction with nothing discarded. Cost: chi(delta) roughly
    # doubles, and delta is used twice per step.
    delta = if exact_delta
        +(A, -1 * B; cutoff=0.0, maxdim=maxlinkdim(A) + maxlinkdim(B))
    else
        +(A, -1 * B; cutoff=cutoff, maxdim=maxdim)
    end

    if verify
        recon = +(B, delta; cutoff=0.0, maxdim=maxlinkdim(B) + maxlinkdim(delta))
        D     = +(recon, -1 * A; cutoff=0.0, maxdim=maxlinkdim(recon) + maxlinkdim(A))
        rel   = _fnorm(D) / max(_fnorm(A), eps())
        @info "step_defect_MPO verify" dt=dt_coarse q=q exact_delta=exact_delta chi_A=maxlinkdim(A) chi_B=maxlinkdim(B) chi_delta=maxlinkdim(delta) norm_ratio=(_fnorm(delta)/_fnorm(A)) reconstruction_relerr=rel
    end

    return (A=A, B=B, delta=delta)
end


# -----------------------------------------------------------------------------
# The four-object recursion for a single (i, j) pair
# -----------------------------------------------------------------------------
#
# `maxdim_G` is deliberately separate from `maxdim`: G only ever reaches Phi
# pre-multiplied by two factors of delta, so an error eta_G on G contributes
# eta_G * ||delta||^2 to Phi -- i.e. RELATIVE error eta_G on the answer, with no
# amplification. G is the object that saturates the bond-dimension ceiling and
# it is also the one you need least accurately. Set maxdim_G well below maxdim.

function build_Phi_pair(n, J, gammas, t, k_i::Int, k_j::Int, k0::Int,
                        lsites::LiouvilleSites;
                        cutoff=1e-12, maxdim=128, maxdim_G=48,
                        order::Int=2, order_ref::Int=2, dissipation::Bool=true,
                        trace::Bool=false, identity_check::Bool=false,
                        exact_delta::Bool=true, verify::Bool=false)

    @assert k0 % k_i == 0 && k0 % k_j == 0 "k0 must be an integer multiple of every k_j"

    chi_ceiling = theoretical_max_bond_dim(n)
    @assert maxdim   <= chi_ceiling "maxdim=$maxdim exceeds the n=$n ceiling $chi_ceiling (see section 4 note above)"
    @assert maxdim_G <= chi_ceiling "maxdim_G=$maxdim_G exceeds the n=$n ceiling $chi_ceiling"

    dt_i, dt_j = t / k_i, t / k_j

    Li = step_defect_MPO(n, J, gammas, dt_i, k0 ÷ k_i, lsites, cutoff, maxdim;
                         order=order, order_ref=order_ref, dissipation=dissipation,
                         exact_delta=exact_delta, verify=verify)
    Rj = step_defect_MPO(n, J, gammas, dt_j, k0 ÷ k_j, lsites, cutoff, maxdim;
                         order=order, order_ref=order_ref, dissipation=dissipation,
                         exact_delta=exact_delta, verify=verify)

    Ai_dag     = op_dag(Li.A)
    Bi_dag     = op_dag(Li.B)
    deltai_dag = op_dag(Li.delta)

    G::MaybeMPO   = identity_liouville_mpo(lsites)
    Xi::MaybeMPO  = nothing
    Psi::MaybeMPO = nothing
    Phi::MaybeMPO = nothing

    time_i = 0.0
    time_j = 0.0
    hist = NamedTuple[]

    # Optional companion object for the section-4.5 consistency check. Since
    # F = G + Xi + Psi + Phi identically, running F through the SAME schedule
    # and comparing at every step localises any sign or ordering error in the
    # recursion to a single step and a single term. F here is built with the
    # coarse steps A_i, A_j exactly as build_open_F does.
    F_chk::MaybeMPO = identity_check ? identity_liouville_mpo(lsites) : nothing
    id_resid = Float64[]

    while (time_i < t - 1e-12) || (time_j < t - 1e-12)

        if (time_j <= time_i) && (time_j < t - 1e-12)
            # ---- advance RIGHT clock by one coarse j-step ----
            # All four updates read the OLD values, so compute before assigning.
            Phi_new = _add(_rmul(Phi, Rj.A; cutoff=cutoff, maxdim=maxdim),
                           _rmul(Xi,  Rj.delta; cutoff=cutoff, maxdim=maxdim);
                           cutoff=cutoff, maxdim=maxdim)
            Psi_new = _add(_rmul(Psi, Rj.A; cutoff=cutoff, maxdim=maxdim),
                           _rmul(G,   Rj.delta; cutoff=cutoff, maxdim=maxdim);
                           cutoff=cutoff, maxdim=maxdim)
            Xi_new  = _rmul(Xi, Rj.B; cutoff=cutoff, maxdim=maxdim)
            G_new   = _rmul(G,  Rj.B; cutoff=cutoff, maxdim=maxdim_G)

            Phi, Psi, Xi, G = Phi_new, Psi_new, Xi_new, G_new
            identity_check && (F_chk = _rmul(F_chk, Rj.A; cutoff=cutoff, maxdim=maxdim))
            time_j += dt_j
            side = :right

        elseif time_i < t - 1e-12
            # ---- advance LEFT clock by one coarse i-step ----
            Phi_new = _add(_lmul(Ai_dag,     Phi; cutoff=cutoff, maxdim=maxdim),
                           _lmul(deltai_dag, Psi; cutoff=cutoff, maxdim=maxdim);
                           cutoff=cutoff, maxdim=maxdim)
            Xi_new  = _add(_lmul(Ai_dag,     Xi;  cutoff=cutoff, maxdim=maxdim),
                           _lmul(deltai_dag, G;   cutoff=cutoff, maxdim=maxdim);
                           cutoff=cutoff, maxdim=maxdim)
            Psi_new = _lmul(Bi_dag, Psi; cutoff=cutoff, maxdim=maxdim)
            G_new   = _lmul(Bi_dag, G;   cutoff=cutoff, maxdim=maxdim_G)

            Phi, Psi, Xi, G = Phi_new, Psi_new, Xi_new, G_new
            identity_check && (F_chk = _lmul(Ai_dag, F_chk; cutoff=cutoff, maxdim=maxdim))
            time_i += dt_i
            side = :left
        end

        if identity_check
            # ||(G + Xi + Psi + Phi) - F||_F / ||F||_F
            Ssum = _add(_add(G, Xi; cutoff=cutoff, maxdim=maxdim),
                        _add(Psi, Phi; cutoff=cutoff, maxdim=maxdim);
                        cutoff=cutoff, maxdim=maxdim)
            D = _add(Ssum, -1 * F_chk; cutoff=cutoff, maxdim=maxdim)
            push!(id_resid, _fnorm(D) / max(_fnorm(F_chk), eps()))
        end

        if trace
            push!(hist, (
                side    = side,
                t_i     = time_i,
                t_j     = time_j,
                chi_G   = G   === nothing ? 0 : middle_bond_dim(G),
                chi_Xi  = Xi  === nothing ? 0 : middle_bond_dim(Xi),
                chi_Psi = Psi === nothing ? 0 : middle_bond_dim(Psi),
                chi_Phi = Phi === nothing ? 0 : middle_bond_dim(Phi),
                nrm_G   = _fnorm(G),
                nrm_Xi  = _fnorm(Xi),
                nrm_Psi = _fnorm(Psi),
                nrm_Phi = _fnorm(Phi),
            ))
        end
    end

    return (Phi=Phi, G=G, Xi=Xi, Psi=Psi, history=hist,
            F_check=F_chk, identity_residual=id_resid)
end


# -----------------------------------------------------------------------------
# The Trotter-error Gram matrix N
# -----------------------------------------------------------------------------
#
# N_ij = <<rho0| Phi_ij |rho0>>. Real and symmetric PSD by construction; we
# symmetrize explicitly to kill asymmetric truncation noise (a free improvement
# that the M/L formulation cannot make, since M and L are truncated separately).

function trotter_error_gram(n, J, gammas, t, ks, k0::Int,
                            lsites::LiouvilleSites, rho0::MPS;
                            cutoff=1e-12, maxdim=128, maxdim_G=48,
                            order::Int=2, order_ref::Int=2, dissipation::Bool=true,
                            trace::Bool=false, exact_delta::Bool=true)
    r = length(ks)
    N = zeros(Float64, r, r)
    traces = Dict{Tuple{Int,Int},Vector{NamedTuple}}()

    for i in 1:r, j in i:r          # symmetric: build the upper triangle only
        res = build_Phi_pair(n, J, gammas, t, ks[i], ks[j], k0, lsites;
                             cutoff=cutoff, maxdim=maxdim, maxdim_G=maxdim_G,
                             order=order, order_ref=order_ref,
                             dissipation=dissipation, trace=trace,
                             exact_delta=exact_delta)
        val = real(_expect(res.Phi, rho0))
        N[i, j] = val
        N[j, i] = val
        trace && (traces[(i, j)] = res.history)
    end

    N .= 0.5 .* (N .+ N')
    return N, traces
end


# -----------------------------------------------------------------------------
# Coefficients and errors straight from N
# -----------------------------------------------------------------------------
#
#   minimize c^T N c   s.t.  sum(c) = 1
#   =>  c = N^{-1} 1 / (1^T N^{-1} 1),   E_mpf = 1 / (1^T N^{-1} 1)
#
# No Lagrange block system, no purity, no L vector. N is near rank-1 (all the
# Delta_j are Trotter errors of the same dynamics and hence nearly parallel), so
# the solve is genuinely ill-conditioned -- that ill-conditioning is INTRINSIC
# and was merely hidden inside the M formulation. We therefore solve via a
# symmetric eigendecomposition with an explicit floor rather than a raw `\`,
# and return the condition number so the caller can see it.

function trotter_error_coefficients(N::AbstractMatrix; rel_floor::Float64=1e-12)
    r = size(N, 1)
    Nsym = 0.5 * (N + N')
    F = eigen(Symmetric(Nsym))
    lam = F.values
    lam_max = maximum(abs, lam)
    floorval = rel_floor * lam_max

    # Clip non-positive / negligible modes: N is PSD in exact arithmetic, so any
    # eigenvalue at or below the floor is truncation noise, not physics.
    lam_reg = max.(lam, floorval)
    ones_v = ones(Float64, r)
    y = F.vectors' * ones_v
    Ninv_1 = F.vectors * (y ./ lam_reg)

    denom = dot(ones_v, Ninv_1)
    c = Ninv_1 ./ denom
    E_mpf = 1.0 / denom

    return (coeffs=c,
            E_mpf=E_mpf,
            E_trot=[N[j, j] for j in 1:r],
            eigvals=lam,
            cond=lam_max / max(minimum(lam), floorval),
            n_clipped=count(<(floorval), lam))
end


# -----------------------------------------------------------------------------
# One-call driver, mirroring test_dynamic_mpf_open
# -----------------------------------------------------------------------------

function test_dmpf_open_N(n, J, gammas, t, ks, k0, lsites::LiouvilleSites, rho0::MPS;
                          cutoff=1e-12, maxdim=128, maxdim_G=48,
                          order::Int=2, order_ref::Int=2, dissipation::Bool=true,
                          trace::Bool=false, exact_delta::Bool=true)
    N, traces = trotter_error_gram(n, J, gammas, t, ks, k0, lsites, rho0;
                                   cutoff=cutoff, maxdim=maxdim, maxdim_G=maxdim_G,
                                   order=order, order_ref=order_ref,
                                   dissipation=dissipation, trace=trace,
                                   exact_delta=exact_delta)
    sol = trotter_error_coefficients(N)
    return (N=N, coeffs=sol.coeffs, E_mpf=sol.E_mpf, E_trot=sol.E_trot,
            eigvals=sol.eigvals, cond=sol.cond, n_clipped=sol.n_clipped,
            traces=traces)
end


# =============================================================================
# VALIDATION against the existing M / L / purity route
# =============================================================================
#
# The identity N_ij = M_ij - L_i - L_j + P holds in EXACT arithmetic. The whole
# point of this file is that the right-hand side is numerically useless (that is
# section 7's catastrophic cancellation) while the left-hand side is not. So
# this comparison is a CONSISTENCY CHECK AT LOW ACCURACY, not a precision test:
# expect agreement only to the level of the M-route's own truncation floor
# (~2e-3 absolute at n=5, maxdim=256), which may well be larger than N itself.
#
# The check that actually matters is against the DIRECT state-overlap oracle
# (section 9a): evolve rho_kj and rho_ref explicitly as MPS, form the error
# vectors, take their Gram matrix. That is disqualified as a production method
# but is exactly right as ground truth at small n.

function validate_N_against_direct(n, J, gammas, t, ks, k_ref,
                                   lsites::LiouvilleSites, rho0::MPS;
                                   cutoff=1e-12, maxdim=256,
                                   order::Int=2, order_ref::Int=2,
                                   dissipation::Bool=true)
    r = length(ks)

    # Reference state |rho(t)>>
    S_ref = get_open_step_MPO(n, J, gammas, t / k_ref, lsites, cutoff, maxdim;
                              order=order_ref, dissipation=dissipation)
    rho_ref = deepcopy(rho0)
    for _ in 1:k_ref
        rho_ref = apply(S_ref, rho_ref; cutoff=cutoff, maxdim=maxdim)
    end

    # Error vectors |delta_j>> = |rho_kj(t)>> - |rho(t)>>
    dvecs = MPS[]
    for kj in ks
        S_j = get_open_step_MPO(n, J, gammas, t / kj, lsites, cutoff, maxdim;
                                order=order, dissipation=dissipation)
        rho_j = deepcopy(rho0)
        for _ in 1:kj
            rho_j = apply(S_j, rho_j; cutoff=cutoff, maxdim=maxdim)
        end
        push!(dvecs, +(rho_j, -1 * rho_ref; cutoff=cutoff, maxdim=maxdim))
    end

    N_direct = zeros(Float64, r, r)
    for i in 1:r, j in 1:r
        N_direct[i, j] = real(inner(dvecs[i], dvecs[j]))
    end
    return N_direct
end


function print_N_report(res; N_direct=nothing)
    println("Trotter-error Gram matrix N:")
    display(res.N)
    println()
    println("eigenvalues of N : ", res.eigvals)
    println("condition number : ", round(res.cond; sigdigits=4),
            "   (clipped modes: ", res.n_clipped, ")")
    println("coefficients c   : ", round.(res.coeffs; digits=5))
    println("E_mpf            : ", res.E_mpf)
    println("E_trot (= diag N): ", res.E_trot)
    if N_direct !== nothing
        println()
        println("|| N - N_direct ||_F        : ", norm(res.N .- N_direct))
        println("relative to ||N_direct||_F  : ", norm(res.N .- N_direct) / norm(N_direct))
        sd = trotter_error_coefficients(N_direct)
        println("c_direct                    : ", round.(sd.coeffs; digits=5))
        println("max |dc|                    : ", maximum(abs.(res.coeffs .- sd.coeffs)))
    end
end


# -----------------------------------------------------------------------------
# The measurement that decides whether Route 1 is enough on its own
# -----------------------------------------------------------------------------
#
# Runs the pair recursion with `trace=true` and prints, per step, the bond
# dimension AND Frobenius norm of each of the four objects. Two things to read
# off:
#
#  (1) the norm hierarchy ||G|| >> ||Xi||,||Psi|| >> ||Phi|| should hold, with
#      ||Phi||^2 landing near E_kj ~ 1e-4 at the final step. If it does not, the
#      k0 reference is not converged and Delta_j is measuring reference error
#      rather than Trotter error.
#
#  (2) chi_Phi vs chi_G. Route 1 succeeds outright if chi_Phi stays materially
#      below chi_G. If chi_Phi ALSO saturates, Route 1 has still bought the
#      precision fix (relative rather than amplified error) but not a complexity
#      win -- and the answer then has to come from Route 2 or 3.

function trace_report(hist::Vector{NamedTuple})
    println("side  t_i    t_j    | chi:   G   Xi  Psi  Phi | norms:      G        Xi       Psi       Phi")
    for h in hist
        println(rpad(h.side, 5), " ",
                rpad(round(h.t_i; digits=3), 6), " ",
                rpad(round(h.t_j; digits=3), 6), " |     ",
                rpad(h.chi_G, 4), rpad(h.chi_Xi, 4), rpad(h.chi_Psi, 5), rpad(h.chi_Phi, 5), "|  ",
                rpad(round(h.nrm_G;   sigdigits=3), 9),
                rpad(round(h.nrm_Xi;  sigdigits=3), 9),
                rpad(round(h.nrm_Psi; sigdigits=3), 9),
                round(h.nrm_Phi; sigdigits=3))
    end
end

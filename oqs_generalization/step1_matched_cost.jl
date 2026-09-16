# =============================================================================
# step1_matched_cost.jl   --   THE DECISIVE MEASUREMENT
#
# THE QUESTION
# ------------
# Give a classical tensor-network budget chi to two workflows:
#
#   (A) PURELY CLASSICAL. Spend chi simulating rho(t) and read off <O>.
#       Error: |<O>_chi - <O>_exact|.
#
#   (B) HYBRID (DMPF). Spend chi computing the COEFFICIENTS c_j only, and take
#       the expectation values <O>_kj from the quantum computer.
#       Error: |sum_j c_j(chi) <O>_kj - <O>_exact|.
#
# Does (B) beat (A) at the same chi? This is exactly the comparison in
# Robertson et al., arXiv:2609.05024, Fig. 4, transplanted from noise mitigation
# to Trotter error. It is NOT the claim the project has been chasing
# (asymptotically cheaper classical preprocessing); it is the claim L-MPF
# actually makes and demonstrates, and it is the one that does not collapse
# under the "if you can simulate the state you don't need the QPU" objection --
# because the answer is allowed to depend on the QPU's <O>_kj, which is where
# the accuracy at large n is going to come from.
#
# THE MECHANISM BEING TESTED
# --------------------------
# Truncation hurts (A) at FIRST order: |O>> is a fixed vector with generic
# overlap with the discarded part of rho(t), so the loss shows up directly.
#
# Truncation should hurt (B) at SECOND order, twice over:
#
#   (i) N_ij = <<Delta_i|Delta_j>> is an overlap of two NEARLY PARALLEL states
#       produced by near-identical circuits at the same cutoff. Their discarded
#       subspaces are nearly the same, so the first-order terms
#       <dDelta_i|Delta_j> + <Delta_i|dDelta_j> largely cancel. This is
#       equations (21)-(23) of L-MPF ("the errors on the coefficients are
#       determined by the DIFFERENCE in the truncation errors, not their
#       magnitude"), and it is the same effect the project already met twice:
#       maxdim_G = maxdim/2 costing 10x in dc, and an exact P mixed with a
#       truncated L destroying the gauge cancellation.
#
#  (ii) A coefficient error dc reaches the answer only through dc' N dc (exact
#       identity, see induced_error in liouville_state_tools.jl), so the
#       amplification is 1/lambda_min rather than the 1/lambda_min^2 that
#       max|dc| reports. With cond(N) ~ 800 that is the factor the project has
#       been charging itself by scoring on max|dc|.
#
# The script measures the OUTCOME (does B beat A) and the MECHANISM (are the
# truncation errors actually correlated, and does dc actually live in the
# near-null direction of N) separately. If the outcome holds but the mechanism
# does not, the effect has another cause and will not survive to larger t or
# gamma -- which is exactly what we need to know before scaling up.
#
# KILL CONDITIONS -- read these off the output before anything else
# -----------------------------------------------------------------
#   * err_proj (the chi-independent projection floor, = the error of the DMPF
#     combination with PERFECT coefficients) is not below err_direct at the
#     small-chi end. Then there is nothing to win at any chi and the programme
#     stops here.
#   * ratio_hw <= 1 across the board: (B) never beats (A). Same conclusion.
#   * corr_ref_kj near 0: the cancellation in (i) is absent, so any advantage
#     seen is accidental.
#
# WHAT IS DELIBERATELY IDEALISED
# ------------------------------
# <O>_kj are taken EXACT (untruncated), as a stand-in for a perfect QPU. Step 1
# isolates the classical-error question; hardware noise and shot noise are a
# separate axis and would only be added once this passes. Saying so plainly in
# the paper is better than pretending otherwise: the honest claim from this
# script is "at matched classical bond dimension, the coefficients are a much
# easier classical target than the observable", which is a statement about
# tensor networks, not about hardware.
#
# Environment: N_QUBITS GAMMA TVAL KS K0 ORDER ORDER_REF CHI_LIST MPO_MAXDIM TAG
# Output: step1_observables<_tag>.csv, step1_coefficients<_tag>.csv
# =============================================================================

import Distributions, Random
using LinearAlgebra, Printf
include("liouville_state_tools.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n          = parse(Int,     getenv("N_QUBITS",   8))
gamma      = parse(Float64, getenv("GAMMA",      0.05))
t          = parse(Float64, getenv("TVAL",       3.0))
k0         = parse(Int,     getenv("K0",         96))    # <- from Step 0
order      = parse(Int,     getenv("ORDER",      2))
order_ref  = parse(Int,     getenv("ORDER_REF",  2))     # <- from Step 0
mpo_maxdim = parse(Int,     getenv("MPO_MAXDIM", 512))
tag        = getenv("TAG", "")
ks         = parse.(Int, split(getenv("KS", "3,8"), ","))

# Product-formula splitting; see symmetric_splitting.jl.
#   candidates -> :project by default. DMPF corrects whatever formula they use,
#                 so their base order is not part of the claim, and keeping them
#                 preserves continuity with everything already computed.
#   reference  -> whatever Step 0 says. It must be CONVERGED, and :project is
#                 first order once gamma > 0, which makes that unreachable.
splitting     = Symbol(getenv("SPLITTING",     "project"))
splitting_ref = Symbol(getenv("SPLITTING_REF", "strang"))
# :gates (default) applies the two-site gate list straight to the MPS; :mpo
# compresses it into a step MPO first. For a deep formula (strang:4 is 25
# layers) the MPO route is both slower and less accurate, because the step
# operator itself gets truncated at MPO_MAXDIM. See liouville_state_tools.jl.
evo_mode = Symbol(getenv("EVO_MODE", "gates"))

sfx       = isempty(tag) ? "" : "_" * tag
chi_exact = state_max_bond_dim(n)
r         = length(ks)
# See the CUTOFF TRAP block in liouville_state_tools.jl. ITensors' cutoff bounds
# the SQUARED discarded weight, so 1e-16 permits a state error of 1e-8 per
# truncation -- which accumulates linearly in k0 and is NOT machine precision.
# It must be at round-off here, because eps_chi is supposed to be the ONLY
# source of loss: any cutoff-induced error is unmeasured and would corrupt the
# x-axis of the headline figure.
ct        = parse(Float64, getenv("CUTOFF", EXACT_CUTOFF))

chi_grid = let raw = getenv("CHI_LIST", "")
    g = isempty(raw) ? [4, 8, 16, 32, 64, 128, 256, 512, 1024] : parse.(Int, split(raw, ","))
    filter(<=(chi_exact), g)
end

@printf("step1_matched_cost\n")
@printf("  n=%d gamma=%.3f t=%.1f ks=%s k0=%d\n", n, gamma, t, string(ks), k0)
@printf("  candidates: splitting=%s order=%d    reference: splitting=%s order=%d    mode=%s\n",
        splitting, order, splitting_ref, order_ref, evo_mode)
@printf("  Liouville MPS ceiling = %d (exact point); chi grid = %s\n",
        chi_exact, string(chi_grid))
@printf("  cutoff = %.1e  ->  permitted state error per truncation = %.1e\n", ct, sqrt(ct))
if splitting_ref === :project
    println("  WARNING: the reference is using :project, which is FIRST order once gamma > 0")
    println("           (see symmetric_splitting.jl). Unless Step 0 says otherwise, this")
    println("           reference is not converged and every 'error' below is a distance")
    println("           to a wrong state.")
end
if k0 % lcm(ks...) != 0
    @printf("  WARNING: k0=%d is not a multiple of lcm(ks)=%d. Fine for this script\n", k0, lcm(ks...))
    @printf("           (state route), but these coefficients cannot then be cross-checked\n")
    @printf("           against the MOC/N-route, whose B_j blocks require k0 %% k_j == 0.\n")
end
println()
flush(stdout)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

# -----------------------------------------------------------------------------
# Observables
# -----------------------------------------------------------------------------
# Single-site Z on every site, plus ZZ on the middle bond. Mean-Z is derived
# from the single-site values (a sum of observables is not a product MPS, but
# every workflow here is linear in O, so averaging afterwards is exact).

mid = n ÷ 2
obs_names = String[]
obs_mps   = MPS[]
for m in 1:n
    push!(obs_names, "Z$m"); push!(obs_mps, z_observable(lsites, m))
end
push!(obs_names, "Z$(mid)Z$(mid+1)"); push!(obs_mps, zz_observable(lsites, mid, mid + 1))
O_id = identity_observable(lsites)
n_obs = length(obs_names)

# -----------------------------------------------------------------------------
# EXACT PASS -- ground truth and the QPU surrogate
# -----------------------------------------------------------------------------
println("="^100)
println("EXACT PASS at chi = $chi_exact (untruncated)")
println("="^100)
flush(stdout)

t0 = time()
e_ref = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                       maxdim=chi_exact, order=order_ref, cutoff=ct, mpo_maxdim=mpo_maxdim,
                       splitting=splitting_ref, mode=evo_mode, id_mps=O_id)
rho_ref_exact = e_ref.rho
@printf("  reference k0=%-5d  chi_S=%-4d chi_rho=%-5d eps=%.2e  Tr=%+.12f  (%.0f s)\n",
        k0, e_ref.chi_S, e_ref.chi, e_ref.eps, real(e_ref.trace), time() - t0)
if e_ref.eps > 1e-10
    @warn "reference at the ceiling discarded weight $(e_ref.eps): it is NOT exact, so every 'error' below is a distance to a truncated state. Check state_max_bond_dim and MPO_MAXDIM." eps=e_ref.eps
end
# (a value around 1e-16 is expected: the discarded weight is measured as a
#  difference of two O(1) norms, so it inherits machine round-off. Anything
#  above 1e-10 means real truncation is happening at the ceiling.)

rhos_exact = MPS[]
for kj in ks
    ek = evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                        maxdim=chi_exact, order=order, cutoff=ct, mpo_maxdim=mpo_maxdim,
                        splitting=splitting, mode=evo_mode, id_mps=O_id)
    push!(rhos_exact, ek.rho)
    @printf("  candidate k=%-5d  chi_S=%-4d chi_rho=%-5d eps=%.2e  Tr=%+.12f\n",
            kj, ek.chi_S, ek.chi, ek.eps, real(ek.trace))
end
println()
flush(stdout)

N_exact, _ = trotter_error_gram_from_states(rhos_exact, rho_ref_exact)
sol_exact  = coefficients_from_N(N_exact)
c_exact    = sol_exact.coeffs

println("  N_exact       = ", N_exact)
println("  eigenvalues   = ", sol_exact.eigvals)
@printf("  cond(N)       = %.4g\n", sol_exact.cond)
@printf("  c_exact       = %s\n", string(round.(c_exact; digits=6)))
@printf("  E_mpf         = %.6e     E_kj = %s\n", sol_exact.E_mpf,
        string([@sprintf("%.4e", x) for x in sol_exact.E_trot]))
println()

# <O>_exact (target), and <O>_kj (the QPU surrogate).
O_exact = [real(expval(obs_mps[a], rho_ref_exact)) for a in 1:n_obs]
O_kj    = [real(expval(obs_mps[a], rhos_exact[j])) for a in 1:n_obs, j in 1:r]

# The chi-independent projection floor: the DMPF combination with PERFECT
# coefficients. Analogue of eps_proj in L-MPF eq. (20). If this is not below the
# direct simulation's error, no amount of chi helps.
err_proj = [abs(sum(c_exact[j] * O_kj[a, j] for j in 1:r) - O_exact[a]) for a in 1:n_obs]
# The bar every method has to clear: the best SINGLE Trotter circuit.
err_best_single = [minimum(abs(O_kj[a, j] - O_exact[a]) for j in 1:r) for a in 1:n_obs]

println("  observable            <O>_exact     err_proj      err_best_single   proj/single")
println("  " * "-"^84)
for a in 1:n_obs
    @printf("  %-18s  %+.8f   %.4e     %.4e        %.3f\n",
            obs_names[a], O_exact[a], err_proj[a], err_best_single[a],
            err_proj[a] / max(err_best_single[a], 1e-300))
end
println()
println("  (proj/single < 1 means DMPF with perfect coefficients beats the best single")
println("   circuit. If it is >= 1 the multiproduct fit is not buying anything here and")
println("   the candidate family ks needs redesigning before Step 1 means anything.)")
println()
flush(stdout)

# -----------------------------------------------------------------------------
# SWEEP
# -----------------------------------------------------------------------------

obs_rows = ["obs,chi,eps_chi,O_exact,O_direct,err_direct,O_dmpf_hw,err_dmpf_hw," *
            "O_dmpf_cl,err_dmpf_cl,err_proj,err_best_single,ratio_hw,ratio_cl"]

coef_hdr = "chi,eps_ref,eps_cand_total,relerr_N,dc_max,dcNdc,sqrt_dcNdc,bound_Zobs," *
           "E_mpf_chi,E_mpf_exact,lam_min_chi,cond_chi,dc_frac_null,time_ref_s,time_cand_s," *
           join(["corr_ref_k$kj" for kj in ks], ",") * "," *
           join(["proj_lam$i" for i in 1:r], ",")
coef_rows = [coef_hdr]

println("="^100)
println("SWEEP")
println("="^100)
println("  chi | eps_chi   | relerr_N | dc_max   | dc'Ndc   | corr     | " *
        "err_direct | err_dmpf | ratio")
println("  " * "-"^96)

for chi in chi_grid
    t0 = time()
    er = evolve_trotter(n, J, gammas, t, k0, lsites, rho0;
                        maxdim=chi, order=order_ref, cutoff=ct, mpo_maxdim=mpo_maxdim,
                        splitting=splitting_ref, mode=evo_mode, id_mps=O_id)
    time_ref = time() - t0
    rho_ref_chi = er.rho

    t0 = time()
    rhos_chi = MPS[]
    eps_cand = 0.0
    for kj in ks
        ek = evolve_trotter(n, J, gammas, t, kj, lsites, rho0;
                            maxdim=chi, order=order, cutoff=ct, mpo_maxdim=mpo_maxdim,
                            splitting=splitting, mode=evo_mode, id_mps=O_id)
        push!(rhos_chi, ek.rho)
        eps_cand += ek.eps
    end
    time_cand = time() - t0

    # ---- coefficients at this chi -------------------------------------------
    N_chi, _ = trotter_error_gram_from_states(rhos_chi, rho_ref_chi)
    sol_chi  = coefficients_from_N(N_chi)
    c_chi    = sol_chi.coeffs
    dc       = c_chi .- c_exact                       # 1'dc = 0 by construction
    dcNdc    = induced_error(N_exact, dc)             # EXACT damage: E(c*+dc) - E_mpf
    relerr_N = norm(N_chi .- N_exact) / max(norm(N_exact), 1e-300)

    # ---- mechanism: are the truncation errors correlated? -------------------
    corrs = [truncation_error_correlation(rho_ref_chi, rho_ref_exact,
                                          rhos_chi[j], rhos_exact[j]).corr for j in 1:r]

    # ---- mechanism: where does dc live in the eigenbasis of N_exact? --------
    # eigvecs are ordered by ASCENDING eigenvalue, so proj[1] is the near-null
    # (worst-conditioned) direction. dc_frac_null near 1 is the L-MPF Appendix D
    # picture: the error is large but points where it does not matter.
    projs = abs.(sol_exact.eigvecs' * dc)
    dc_frac_null = projs[1] / max(norm(dc), 1e-300)

    # ---- observables --------------------------------------------------------
    for a in 1:n_obs
        O_direct = real(expval(obs_mps[a], rho_ref_chi))
        # (B): coefficients from chi, expectation values from the QPU surrogate.
        O_dmpf_hw = sum(c_chi[j] * O_kj[a, j] for j in 1:r)
        # Control: the same combination done ENTIRELY classically at this chi.
        # Should track err_direct, not err_dmpf_hw -- if it does not, the gain
        # is coming from the fit rather than from the hybrid structure.
        O_dmpf_cl = sum(c_chi[j] * real(expval(obs_mps[a], rhos_chi[j])) for j in 1:r)

        ed  = abs(O_direct  - O_exact[a])
        ehw = abs(O_dmpf_hw - O_exact[a])
        ecl = abs(O_dmpf_cl - O_exact[a])

        push!(obs_rows, @sprintf("%s,%d,%.8e,%.10f,%.10f,%.8e,%.10f,%.8e,%.10f,%.8e,%.8e,%.8e,%.6f,%.6f",
                                 obs_names[a], chi, er.eps, O_exact[a],
                                 O_direct, ed, O_dmpf_hw, ehw, O_dmpf_cl, ecl,
                                 err_proj[a], err_best_single[a],
                                 ed / max(ehw, 1e-300), ed / max(ecl, 1e-300)))
    end

    # Headline line uses the middle-site Z.
    O_direct_mid = real(expval(obs_mps[mid], rho_ref_chi))
    O_dmpf_mid   = sum(c_chi[j] * O_kj[mid, j] for j in 1:r)
    ed_mid  = abs(O_direct_mid - O_exact[mid])
    ehw_mid = abs(O_dmpf_mid   - O_exact[mid])

    # ||O||_F = 2^(n/2) for any Pauli string, so this Cauchy-Schwarz bound is
    # loose by that factor. Recorded because its SCALING in chi is the thing to
    # compare with err_dmpf, not its magnitude.
    bound = sqrt(2.0^n) * sqrt(max(dcNdc, 0.0))

    push!(coef_rows, @sprintf("%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.6e,%.6f,%.1f,%.1f,%s,%s",
                              chi, er.eps, eps_cand, relerr_N,
                              maximum(abs.(dc)), dcNdc, sqrt(max(dcNdc, 0.0)), bound,
                              sol_chi.E_mpf, sol_exact.E_mpf,
                              minimum(sol_chi.eigvals), sol_chi.cond, dc_frac_null,
                              time_ref, time_cand,
                              join([@sprintf("%.6f", c) for c in corrs], ","),
                              join([@sprintf("%.6e", p) for p in projs], ",")))

    @printf("  %4d | %.2e | %.2e | %.2e | %.2e | %+.4f | %.4e | %.2e | %.2f\n",
            chi, er.eps, relerr_N, maximum(abs.(dc)), dcNdc,
            minimum(corrs), ed_mid, ehw_mid, ed_mid / max(ehw_mid, 1e-300))
    flush(stdout)
end

write("step1_observables$sfx.csv",  join(obs_rows,  "\n") * "\n")
write("step1_coefficients$sfx.csv", join(coef_rows, "\n") * "\n")

println()
println("wrote step1_observables$sfx.csv, step1_coefficients$sfx.csv")
println()
println("="^100)
println("HOW TO READ THIS")
println("="^100)
println("  ratio_hw > 1 at small chi   -> the hybrid beats the purely classical")
println("                                 simulation at matched bond dimension.")
println("                                 This is the headline. Plot err_direct and")
println("                                 err_dmpf_hw against eps_chi on log axes;")
println("                                 different SLOPES are the real result,")
println("                                 a constant offset is not.")
println()
println("  err_dmpf_hw flattening at err_proj is EXPECTED and correct: err_proj is")
println("  the floor set by the candidate family, not by the tensor network. If the")
println("  floor is hit while err_direct is still falling, the answer is to redesign")
println("  ks (Step 4), not to raise chi.")
println()
println("  corr_ref_kj near 1          -> the truncation errors on rho_ref and rho_kj")
println("                                 are the SAME error, cancelling in Delta.")
println("                                 This is the mechanism; without it the")
println("                                 outcome is accidental and will not survive")
println("                                 longer t or larger gamma.")
println()
println("  dc_frac_null near 1         -> the coefficient error points along the")
println("                                 near-null direction of N, where it does")
println("                                 least damage (L-MPF Appendix D).")
println()
println("  dc_max >> sqrt(dc'Ndc)      -> quantitative proof that max|dc| was the")
println("                                 wrong figure of merit all along.")

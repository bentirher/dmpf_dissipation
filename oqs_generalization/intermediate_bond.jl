# =============================================================================
# intermediate_bond.jl -- where does the N-route's error actually come from?
#
# CHECK 2 §3 FOUND, at n=6, chi=192:
#     Phi's one-shot tail beyond index 192   = 1.26e-03  (0.13%)
#     the sweep's actual |dN22|/N22          = 1.08e-01  (10.8%)
# an 86x gap. Phi IS compressible at 192; the middle-out recursion does not get
# there. So the error is ACCUMULATED through the contraction rather than set by
# the final object's rank.
#
# READ THIS BEFORE INTERPRETING THE RESULT. Decoupling the caps does NOT by
# itself save anything: if the recursion runs at maxdim_int=256 you pay the cost
# of 256, whatever you compress to afterwards. This script is a DIAGNOSTIC that
# splits the error in two, plus a test of one thing that could be a real fix.
#
#   ARM A -- maxdim_int (recursion) vs maxdim_fin (final compression).
#     Vary each with the other fixed.
#       error tracks maxdim_int, flat in maxdim_fin
#           -> the error is accumulated. The final object is cheap to store but
#              expensive to build, and raising maxdim_int is the only lever,
#              i.e. no saving -- the recursion needs the bond dimension it needs.
#       error tracks maxdim_fin, flat in maxdim_int
#           -> the recursion can run CHEAP and compress at the end. A real win.
#       both matter -> read the 2-D table for the cheapest adequate corner.
#
#   ARM B -- error-controlled vs rank-controlled truncation.
#     The sweeps so far ran cutoff=1e-14 AND maxdim=chi, so maxdim always binds
#     and truncation is RANK-limited: every step throws away whatever does not
#     fit, regardless of how much weight that is. Setting a loose cutoff with
#     maxdim at the ceiling instead makes truncation ERROR-limited, and the bond
#     dimension adapts to the actual spectrum at each step. Check 2 measured
#     Phi's effective rank at 1e-2 as 132 -- bounded in n -- so an error-limited
#     schedule may reach that rank naturally while a rank-limited one at 192
#     does not. This is the arm that could actually reduce cost.
#
# Environment: N_QUBITS, GAMMA, TVAL, K0, ORDER, ARM (A|B|both)
# =============================================================================
import Distributions, Random
using LinearAlgebra, Printf
using ITensors, ITensorMPS
include("trotter_error_gram.jl")
include("open_optimization_problem.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n     = parse(Int,     getenv("N_QUBITS", 6))
gamma = parse(Float64, getenv("GAMMA",    0.05))
t     = parse(Float64, getenv("TVAL",     3.0))
k0    = parse(Int,     getenv("K0",       48))
order = parse(Int,     getenv("ORDER",    2))
arm   = getenv("ARM", "both")
ks    = [3, 8]

chi_ceiling = theoretical_max_bond_dim(n)
chi_mps     = 4^min(n ÷ 2, n - n ÷ 2)

Random.seed!(1234)
J      = rand(Distributions.Uniform(1/4, 3/4), n - 1)
gammas = fill(gamma, n)
lsites = liouville_siteinds(n)
rho0   = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

@printf("intermediate_bond  n=%d gamma=%.3f t=%.1f ks=%s k0=%d order=%d\n",
        n, gamma, t, string(ks), k0, order)
@printf("MPO ceiling = %d   MPS ceiling (oracle) = %d\n\n", chi_ceiling, chi_mps)

# ---- ground truth: the oracle AT its ceiling (never above -- check 1b) -------
N_oracle = validate_N_against_direct(n, J, gammas, t, ks, k0, lsites, rho0;
                                     cutoff=0.0, maxdim=chi_mps, order=order,
                                     order_ref=order, dissipation=true)
c_exact = trotter_error_coefficients(N_oracle).coeffs
u = [float(k)^(-order) - float(k0)^(-order) for k in ks]
c_free = [1.0 1.0; u[1] u[2]] \ [1.0, 0.0]
signal = maximum(abs.(c_exact .- c_free))
@printf("oracle c1 = %+.10f   N22 = %.8e\n", c_exact[1], N_oracle[2,2])
@printf("free   c1 = %+.10f   signal = %.4e\n\n", c_free[1], signal)
flush(stdout)

# =============================================================================
# build_Phi_pair with the two caps separated.
# Identical to build_Phi_pair except: the recursion (and the defect MPOs it is
# built from) use maxdim_int, and the returned Phi is compressed to maxdim_fin
# only at the very end.
# =============================================================================
function build_Phi_pair_split(n, J, gammas, t, k_i::Int, k_j::Int, k0::Int,
                              lsites::LiouvilleSites;
                              cutoff=1e-14, maxdim_int=128, maxdim_fin=-1,
                              maxdim_G=-1, order::Int=2, order_ref::Int=2,
                              dissipation::Bool=true, exact_delta::Bool=true)
    @assert k0 % k_i == 0 && k0 % k_j == 0
    maxdim_fin = maxdim_fin < 0 ? maxdim_int : maxdim_fin
    mG = maxdim_G < 0 ? maxdim_int : maxdim_G
    @assert maxdim_int <= theoretical_max_bond_dim(n)

    dt_i, dt_j = t / k_i, t / k_j
    Li = step_defect_MPO(n, J, gammas, dt_i, k0 ÷ k_i, lsites, cutoff, maxdim_int;
                         order=order, order_ref=order_ref, dissipation=dissipation,
                         exact_delta=exact_delta)
    Rj = step_defect_MPO(n, J, gammas, dt_j, k0 ÷ k_j, lsites, cutoff, maxdim_int;
                         order=order, order_ref=order_ref, dissipation=dissipation,
                         exact_delta=exact_delta)
    Ai_dag, Bi_dag, deltai_dag = op_dag(Li.A), op_dag(Li.B), op_dag(Li.delta)

    G::MaybeMPO   = identity_liouville_mpo(lsites)
    Xi::MaybeMPO  = nothing
    Psi::MaybeMPO = nothing
    Phi::MaybeMPO = nothing
    time_i = 0.0; time_j = 0.0

    while (time_i < t - 1e-12) || (time_j < t - 1e-12)
        if (time_j <= time_i) && (time_j < t - 1e-12)
            Phi_new = _add(_rmul(Phi, Rj.A; cutoff=cutoff, maxdim=maxdim_int),
                           _rmul(Xi,  Rj.delta; cutoff=cutoff, maxdim=maxdim_int);
                           cutoff=cutoff, maxdim=maxdim_int)
            Psi_new = _add(_rmul(Psi, Rj.A; cutoff=cutoff, maxdim=maxdim_int),
                           _rmul(G,   Rj.delta; cutoff=cutoff, maxdim=maxdim_int);
                           cutoff=cutoff, maxdim=maxdim_int)
            Xi_new  = _rmul(Xi, Rj.B; cutoff=cutoff, maxdim=maxdim_int)
            G_new   = _rmul(G,  Rj.B; cutoff=cutoff, maxdim=mG)
            Phi, Psi, Xi, G = Phi_new, Psi_new, Xi_new, G_new
            time_j += dt_j
        elseif time_i < t - 1e-12
            Phi_new = _add(_lmul(Ai_dag,     Phi; cutoff=cutoff, maxdim=maxdim_int),
                           _lmul(deltai_dag, Psi; cutoff=cutoff, maxdim=maxdim_int);
                           cutoff=cutoff, maxdim=maxdim_int)
            Xi_new  = _add(_lmul(Ai_dag,     Xi;  cutoff=cutoff, maxdim=maxdim_int),
                           _lmul(deltai_dag, G;   cutoff=cutoff, maxdim=maxdim_int);
                           cutoff=cutoff, maxdim=maxdim_int)
            Psi_new = _lmul(Bi_dag, Psi; cutoff=cutoff, maxdim=maxdim_int)
            G_new   = _lmul(Bi_dag, G;   cutoff=cutoff, maxdim=mG)
            Phi, Psi, Xi, G = Phi_new, Psi_new, Xi_new, G_new
            time_i += dt_i
        end
    end

    chi_built = Phi === nothing ? 0 : middle_bond_dim(Phi)
    if Phi !== nothing && maxdim_fin < chi_built
        Phi = copy(Phi)
        truncate!(Phi; cutoff=0.0, maxdim=maxdim_fin)   # the ONLY final compression
    end
    return (Phi=Phi, chi_built=chi_built,
            chi_final=Phi === nothing ? 0 : middle_bond_dim(Phi))
end

function gram_split(; cutoff, maxdim_int, maxdim_fin=-1, maxdim_G=-1)
    r = length(ks); N = zeros(Float64, r, r); cb = 0; cf = 0
    for i in 1:r, j in i:r
        res = build_Phi_pair_split(n, J, gammas, t, ks[i], ks[j], k0, lsites;
                                   cutoff=cutoff, maxdim_int=maxdim_int,
                                   maxdim_fin=maxdim_fin, maxdim_G=maxdim_G,
                                   order=order, order_ref=order, dissipation=true,
                                   exact_delta=true)
        v = real(_expect(res.Phi, rho0)); N[i,j] = v; N[j,i] = v
        cb = max(cb, res.chi_built); cf = max(cf, res.chi_final)
    end
    N .= 0.5 .* (N .+ N')
    return N, cb, cf
end

score(N) = (maximum(abs.(trotter_error_coefficients(N).coeffs .- c_exact)),
            abs(N[2,2] / N_oracle[2,2] - 1))

# =============================================================================
# ARM A
# =============================================================================
function armA()
    println("="^78)
    println("ARM A : maxdim_int (recursion) vs maxdim_fin (final compression)")
    println("="^78)
    rows = ["maxdim_int,maxdim_fin,chi_built,chi_final,c1,dc,dN22_rel,signal,time_s"]
    ints = filter(<=(chi_ceiling), [64, 128, 192, 256])
    @printf("  %10s %10s %8s %12s %11s %10s\n",
            "maxdim_int","maxdim_fin","chi_fin","dc","dc/signal","dN22")
    for mi in ints, mf in filter(<=(mi), [48, 96, 128, 192, 256])
        el = @elapsed ((N, cb, cf) = gram_split(cutoff=1e-14, maxdim_int=mi, maxdim_fin=mf))
        dc, d22 = score(N)
        @printf("  %10d %10d %8d %12.3e %11.2f %10.2e   %5.0f s\n",
                mi, mf, cf, dc, dc/signal, d22, el)
        push!(rows, @sprintf("%d,%d,%d,%d,%.10f,%.6e,%.6e,%.6e,%.1f",
                             mi, mf, cb, cf,
                             trotter_error_coefficients(N).coeffs[1], dc, d22, signal, el))
        write("intermediate_bond_A.csv", join(rows, "\n") * "\n"); flush(stdout)
    end
    println("""
  READ DOWN a maxdim_int block: if dc is flat in maxdim_fin, the final
  compression is FREE and all the error is accumulated in the recursion.
  READ ACROSS blocks at fixed maxdim_fin: that is the size of the accumulation.
""")
end

# =============================================================================
# ARM B
# =============================================================================
function armB()
    println("="^78)
    println("ARM B : error-controlled (cutoff) vs rank-controlled (maxdim)")
    println("="^78)
    rows = ["mode,cutoff,maxdim,chi_reached,c1,dc,dN22_rel,signal,time_s"]
    @printf("  %-18s %10s %10s %12s %11s %10s\n",
            "mode","cutoff","chi built","dc","dc/signal","dN22")
    for md in filter(<=(chi_ceiling), [96, 128, 192])
        el = @elapsed ((N, cb, cf) = gram_split(cutoff=1e-14, maxdim_int=md))
        dc, d22 = score(N)
        @printf("  %-18s %10.0e %10d %12.3e %11.2f %10.2e   %5.0f s\n",
                "rank-limited", 1e-14, cb, dc, dc/signal, d22, el)
        push!(rows, @sprintf("rank,%.1e,%d,%d,%.10f,%.6e,%.6e,%.6e,%.1f",
                             1e-14, md, cb,
                             trotter_error_coefficients(N).coeffs[1], dc, d22, signal, el))
        write("intermediate_bond_B.csv", join(rows, "\n") * "\n"); flush(stdout)
    end
    for ct in [1e-6, 1e-7, 1e-8, 1e-9]
        el = @elapsed ((N, cb, cf) = gram_split(cutoff=ct, maxdim_int=chi_ceiling))
        dc, d22 = score(N)
        @printf("  %-18s %10.0e %10d %12.3e %11.2f %10.2e   %5.0f s\n",
                "error-limited", ct, cb, dc, dc/signal, d22, el)
        push!(rows, @sprintf("error,%.1e,%d,%d,%.10f,%.6e,%.6e,%.6e,%.1f",
                             ct, chi_ceiling, cb,
                             trotter_error_coefficients(N).coeffs[1], dc, d22, signal, el))
        write("intermediate_bond_B.csv", join(rows, "\n") * "\n"); flush(stdout)
    end
    println("""
  COMPARE AT EQUAL 'chi built'. If an error-limited run reaches the same bond
  dimension as a rank-limited one but with a much smaller dc, then the schedule
  -- not the bond dimension -- was the problem, and that IS a cost saving.
  WARNING: maxdim is set to the CEILING in the error-limited arm, so if the
  cutoff is too loose the bond dimension can run away. Watch 'chi built'.
""")
end

arm in ("A", "both") && armA()
arm in ("B", "both") && armB()
println("wrote intermediate_bond_A.csv, intermediate_bond_B.csv")

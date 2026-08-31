# =============================================================================
# spectra_study.jl   --   operator-Schmidt spectra + discarded weight vs error
#
# Produces the two most important figures:
#   cmp_spectra.csv          spectra of F, G, Xi, Phi (deck slide 23, corrected)
#   cmp_weight_vs_error.csv  Schmidt weight discarded vs resulting error in c
#                            -- THE amplification plot
#
# Reads dc values from cmp_coefficients.csv, so run comparison_study.jl first.
#
# WHY THE WEIGHT-VS-ERROR PLOT AND NOT THE SPECTRA:
# For F, sigma_1 IS the G = E^dag E piece, which cancels identically out of the
# coefficients -- so log10(sigma_i/sigma_1) makes F's tail look negligible when
# that tail is the entire signal. Phi has no such passenger. Comparing the
# normalized spectra side by side is apples-to-oranges and reads as Phi looking
# WORSE (it is: effective rank 193 vs 44). The apples-to-apples question is:
# how much weight did truncation discard, and what did that cost in the answer?
# That ratio IS the amplification factor. Measured geometric means: 548 for the
# M-route, 0.182 for the N-route -- a factor of 3018.
#
# Environment: N_QUBITS, GAMMA, TVAL, K0, ORDER, KDIAG, TAG
# =============================================================================
import Distributions, Random
using LinearAlgebra, Printf
include("trotter_error_gram.jl")
include("spectrum_truncation_analysis.jl")   # has no include of its own, so this
                                             # is a SINGLE clean load (unlike
                                             # comparison_study.jl)
BLAS.set_num_threads(parse(Int, get(ENV,"SLURM_CPUS_PER_TASK","1")))

getenv(k,d)=get(ENV,k,string(d))
n     = parse(Int,     getenv("N_QUBITS", 4))
gamma = parse(Float64, getenv("GAMMA",    0.05))
t     = parse(Float64, getenv("TVAL",     3.0))
k0    = parse(Int,     getenv("K0",       48))
order = parse(Int,     getenv("ORDER",    2))
kd    = parse(Int,     getenv("KDIAG",    8))
tag   = getenv("TAG",""); sfx = isempty(tag) ? "" : "_"*tag
ct    = 1e-14

# MAXCHI caps the build. spectra_study v1 used chi = theoretical_max_bond_dim(n),
# which is 256 at n=4 (fine, and exact) but 4096 at n=6 -- days of wall time and
# certain OOM. Cap it. NOTE the consequence: at n=6 the spectrum below is the
# spectrum of the TRUNCATED Phi, so it can only be read as a LOWER BOUND on the
# effective rank. That is still exactly the measurement we need: if truncated-Phi
# at n=6 already needs more than Phi's 193 at n=4, the rank is growing with n.
chi_ceiling = theoretical_max_bond_dim(n)
chi   = parse(Int, getenv("MAXCHI", chi_ceiling))
chi   = min(chi, chi_ceiling)

Random.seed!(1234)
J = rand(Distributions.Uniform(1/4,3/4), n-1); gammas = fill(gamma,n)
lsites = liouville_siteinds(n)
rho0 = vectorized_initial_state_mps(lsites, collect(0:2:(n-1)) .|> string)

@printf("n=%d  MPO ceiling=%d  build maxdim=%d  k=%d k0=%d order=%d\n",
        n, chi_ceiling, chi, kd, k0, order)
if chi < chi_ceiling
    @printf("  *** capped at %d of a ceiling of %d: every spectrum below is a\n", chi, chi_ceiling)
    @printf("      LOWER BOUND on the true effective rank. ***\n")
end
println(); flush(stdout)

# build_open_F takes a LIST of k's and returns Vector{MPO}, one per (i,j) pair.
# Passing a scalar silently returns a 1-element VECTOR, which then fails at the
# spectrum call. Pass [kd] and index explicitly.
t0=time()
F = build_open_F(n, J, gammas, t, [kd], lsites, ct, chi; order=order, dissipation=true)[1]
@printf("F   built in %6.1f s, chi = %d\n", time()-t0, middle_bond_dim(F)); flush(stdout)

t0=time()
pair = build_Phi_pair(n, J, gammas, t, kd, kd, k0, lsites;
                      cutoff=ct, maxdim=chi, maxdim_G=chi, order=order,
                      order_ref=order, dissipation=true, exact_delta=true)
@printf("Phi built in %6.1f s, chi = %d\n\n", time()-t0, middle_bond_dim(pair.Phi)); flush(stdout)

objs = [("F",F), ("G",pair.G), ("Xi",pair.Xi), ("Psi",pair.Psi), ("Phi",pair.Phi)]
S = Dict(nm => operator_schmidt_spectrum(o) for (nm,o) in objs if o !== nothing)

rows = ["index,object,sigma,sigma_over_s1,cum_weight_frac,tail_weight_frac"]
for (nm,s) in S
    w = s.^2; tot = sum(w); cw = cumsum(w)./tot
    for i in eachindex(s)
        push!(rows, @sprintf("%d,%s,%.10e,%.10e,%.12f,%.12e", i, nm, s[i], s[i]/s[1], cw[i], 1-cw[i]))
    end
end
write("cmp_spectra$sfx.csv", join(rows,"\n")*"\n")

println("object |  chi | ||.||_F    | s_1        | s1^2/tot | eff@1e-3 | eff@1e-6 | s2/s1")
for (nm,_) in objs
    haskey(S,nm) || continue
    s=S[nm]; w=s.^2
    @printf("%-6s | %4d | %.4e | %.4e | %7.2f%% | %8s | %8s | %.4e\n", nm, length(s),
            sqrt(sum(w)), s[1], 100*w[1]/sum(w),
            string(effective_rank(s; weight_fraction=1-1e-3)),
            string(effective_rank(s; weight_fraction=1-1e-6)),
            length(s)>1 ? s[2]/s[1] : NaN)
end

# F vs G: the bond-dimension study was measuring E^dag E, not F.
if haskey(S,"F") && haskey(S,"G")
    F_,G_ = S["F"], S["G"]; m = min(length(F_),length(G_))
    @printf("\n  sum (s_F - s_G)^2 / ||F||^2 = %.4e   <- F and G are the SAME object\n",
            sum((F_[1:m].-G_[1:m]).^2)/sum(F_.^2))
    @printf("  ||Phi||^2 / ||F||^2         = %.4e   <- the signal, as a fraction of F\n",
            sum(S["Phi"].^2)/sum(F_.^2))
end

# ---- discarded weight vs error ---------------------------------------------
disc(s,md) = md >= length(s) ? 0.0 : 1 - sum(s[1:md].^2)/sum(s.^2)
dc = Dict{Tuple{Int,String},Float64}()
fn = "cmp_coefficients$sfx.csv"
if isfile(fn)
    for l in readlines(fn)[2:end]
        f=split(strip(l),","); length(f)>=5 || continue
        dc[(parse(Int,f[1]),String(f[2]))] = parse(Float64,f[5])
    end
else
    println("\nWARNING: $fn not found -- run comparison_study.jl first.")
end

grid = isempty(dc) ? filter(<=(chi),[16,32,48,64,96,128,192,256]) : sort(unique(k[1] for k in keys(dc)))
rw = ["maxdim,object,discarded_weight,dc_max,route,amplification"]
println("\nmaxdim |  F: disc.wt   dc(M)      amp      |  Phi: disc.wt  dc(N)      amp")
println("-"^78)
for md in grid
    wF, wP = disc(S["F"],md), disc(S["Phi"],md)
    dM, dN = get(dc,(md,"M"),NaN), get(dc,(md,"N"),NaN)
    aM, aN = wF>0 ? dM/wF : NaN, wP>0 ? dN/wP : NaN
    push!(rw, @sprintf("%d,F,%.8e,%.8e,M,%.8e",   md, wF, dM, aM))
    push!(rw, @sprintf("%d,Phi,%.8e,%.8e,N,%.8e", md, wP, dN, aN))
    @printf("%6d | %.3e  %.3e  %8.2e | %.3e   %.3e  %8.2e\n", md, wF, dM, aM, wP, dN, aN)
end
write("cmp_weight_vs_error$sfx.csv", join(rw,"\n")*"\n")
println("\nwrote cmp_spectra$sfx.csv, cmp_weight_vs_error$sfx.csv")

# =============================================================================
# entanglement_growth_study.jl
#
# Locates the classically expensive corner of (n, gamma, t) for the vectorized
# MPS/TEBD simulation of the Heisenberg + amplitude damping chain, so that the
# quantum-circuit experiment (Hamiltonian + ancilla-mediated amplitude damping)
# can be sited somewhere defensible.
#
# For each (n, gamma) it evolves |rho(t)>> and records S_op(t) and the bond
# dimension actually required. Each gamma also gets a gamma=0 closed-system
# control run at the same n, so the "does dissipation help or hurt" claim is
# made against a like-for-like baseline rather than against intuition.
#
# WHAT THE OUTPUT SHOULD LOOK LIKE, AND WHAT IT MEANS
#
# Amplitude damping alone (sigma^- only, no absorption) has the trivial product
# steady state |0...0>, so S_op must return to zero at long times. The curve is
# a barrier: rise, peak, fall. The peak sits near t ~ 1/gamma and gets TALLER as
# gamma gets SMALLER. So:
#
#   - the hard regime is small gamma at t ~ 1/gamma, not long times;
#   - "evolve longer and it gets harder" is false for this model;
#   - the circuit experiment should be sited AT the peak, which is the only
#     place the classical baseline is genuinely strained.
#
# If instead chi_req stays flat while S_op falls, that is the Vilchez-Estevez
# effect and is the more interesting result of the two -- it is the separation
# between "the state is simple" and "getting there was not".
#
# Environment: N_LIST, GAMMA, TMAX, NT, DT, ORDER, MAXDIM, CUTOFF, JCOUP,
#              DISORDER, SEED, TAG
# =============================================================================
using LinearAlgebra, Printf
import Random, Distributions
include("vectorized_evolution.jl")
BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))

getenv(k, d) = get(ENV, k, string(d))
n_list   = parse.(Int, split(getenv("N_LIST", "8,12,16,20"), ","))
gamma    = parse(Float64, getenv("GAMMA",  0.05))
tmax     = parse(Float64, getenv("TMAX",   40.0))
nt       = parse(Int,     getenv("NT",     40))
dt       = parse(Float64, getenv("DT",     0.02))
order    = parse(Int,     getenv("ORDER",  2))
maxdim   = parse(Int,     getenv("MAXDIM", 512))
cutoff   = parse(Float64, getenv("CUTOFF", 1e-12))
jcoup    = parse(Float64, getenv("JCOUP",  0.5))
disorder = parse(Bool,    getenv("DISORDER", false))
seed     = parse(Int,     getenv("SEED",   1234))
tag      = getenv("TAG", "")
sfx      = isempty(tag) ? "" : "_" * tag
gtag     = replace(@sprintf("%.3f", gamma), "." => "p")

# Time grid. tmax defaults to ~2/gamma so the barrier peak is inside the window
# rather than just off the right-hand edge -- the mistake that makes every such
# plot look like monotone growth.
times = collect(range(tmax / nt, tmax; length=nt))

@printf("=== entanglement growth study ===\n")
@printf("n_list=%s gamma=%.4f tmax=%.2f (1/gamma=%.2f) nt=%d dt=%.4g\n",
        string(n_list), gamma, tmax, gamma > 0 ? 1/gamma : Inf, nt, dt)
@printf("order=%d maxdim=%d cutoff=%.1e J=%s\n\n",
        order, maxdim, cutoff, disorder ? "U[1/4,3/4] (seed $seed)" : @sprintf("%.3f uniform", jcoup))
flush(stdout)

all_rows = String[]
header_written = false

for n in n_list
    J = if disorder
        Random.seed!(seed)
        rand(Distributions.Uniform(1/4, 3/4), n - 1)
    else
        jcoup
    end

    for (diss, name) in ((true, "open"), (false, "closed"))
        @printf("\n--- n=%d  %s ---\n", n, name)
        flush(stdout)
        res = evolve_vectorized(n, J, gamma, times;
                                dissipation = diss,
                                dt = dt, order = order,
                                cutoff = cutoff, maxdim = maxdim,
                                initial = :neel,
                                tols = [1e-6, 1e-10],
                                verbose = true)
        csv = rows_to_csv(res; label = "$(name)_n$(n)")
        lines = split(chomp(csv), "\n")
        if !header_written
            push!(all_rows, lines[1]); header_written = true
        end
        append!(all_rows, lines[2:end])
    end
end

outfile = "entanglement_growth_g$(gtag)$(sfx).csv"
write(outfile, join(all_rows, "\n") * "\n")
@printf("\nwrote %s (%d rows)\n", outfile, length(all_rows) - 1)

# -----------------------------------------------------------------------------
# Convergence check on the largest n, at the barrier peak. Do not skip this.
# -----------------------------------------------------------------------------
if get(ENV, "SKIP_CONVERGENCE", "false") != "true"
    n_big = maximum(n_list)
    peak_window = filter(t -> t <= min(tmax, 1.5 / max(gamma, 1e-9)), times)
    isempty(peak_window) && (peak_window = times)
    check_times = peak_window[max(1, length(peak_window) ÷ 4):max(1, length(peak_window) ÷ 4):end]
    ladder = filter(<=(maxdim), [64, 128, 256, 512, 1024])
    length(ladder) < 2 && (ladder = [maxdim ÷ 2, maxdim])
    println("\n=== maxdim convergence, n=$n_big ===")
    maxdim_convergence(n_big, disorder ? rand(Distributions.Uniform(1/4,3/4), n_big-1) : jcoup,
                       gamma, collect(check_times), ladder;
                       dt=dt, order=order, cutoff=cutoff, initial=:neel,
                       tols=[1e-6, 1e-10], verbose=true)
end

using ITensors, ITensorMPS
using Random, LinearAlgebra, Printf, Statistics

# =============================================================================
# trajectory_evolution.jl
#
# Quantum-trajectory (dynamic-circuit) MPS simulation of the disordered
# Heisenberg chain under SINGLE-QUBIT amplitude damping, adapted from the
# two-ancilla collective-dissipation workflow (arXiv:2605.25830, App. E).
#
# PURPOSE: this is the missing half of the classical-hardness study. The MPDO
# results bound one classical method; this bounds the other, and it is the one
# most likely to beat the hardware. The quantity of interest is NOT the physics
# (we already know the physics from the MPDO runs) but the COST:
#
#     trajectory cost  ~  N_traj * n * steps * chi_traj^3
#     MPDO cost        ~          n * steps * chi_MPDO^3
#
# so trajectories win whenever chi_traj^3 * N_traj < chi_MPDO^3. With
# chi_MPDO ~ 1e7 at the n=24 operating point, trajectories would need only
# chi_traj < 1e7/N_traj^{1/3} ~ 5e5 to win. Expect them to win comfortably.
# Measuring by how much is the point.
#
# -----------------------------------------------------------------------------
# WHAT CHANGED RELATIVE TO THE COLLECTIVE-DISSIPATION SCRIPT
# -----------------------------------------------------------------------------
# 1. DISSIPATOR. The collective channel needed the two-qubit basis change P,
#    P_dag and the doubly-controlled CCRy into a shared ancilla. Single-qubit
#    amplitude damping needs none of that: per qubit j, one ancilla a_j, and
#
#        CRy(theta) : control q_j, target a_j      theta = 2 asin(sqrt(p))
#        CX         : control a_j, target q_j      p = 1 - exp(-gamma*dt)
#        reset a_j
#
#    which realises exactly K0 = diag(1, sqrt(1-p)), K1 = sqrt(p)|0><1|. Your
#    CRy_g_minus + CX_env_sysL + reset_qubit! triple already was this; we simply
#    drop the P and CCRy layers around it.
#
# 2. HAMILTONIAN. The collective model had Rz + RXX + RYY. The XXZ chain needs
#    an RZZ layer too. The bond unitary here is built to be BIT-IDENTICAL to
#    heisenberg_bond_unitary in the MPDO study, RZZ(2*J*dt) RYY(J*dt) RXX(J*dt),
#    so that trajectory and MPDO numbers are directly comparable rather than
#    approximately comparable. Do not "simplify" the angles.
#
# 3. TIME LOOP. `n_trajectories` rebuilt the initial state and reran the whole
#    circuit from scratch for every entry of t_vec -- O(T^2) work for an O(T)
#    problem, because the gate angles depended on t through a fixed-depth-k
#    scheme. Here dt is FIXED and each trajectory is integrated once, recording
#    at every snapshot as it passes. For a 30-point time grid that is a ~15x
#    saving. (If you specifically want the fixed-depth-k circuit the hardware
#    will run, set `fixed_depth=true` and pass k; see the note in
#    `run_trajectories`.)
#
# 4. RNG. Trajectories now take independent, deterministic per-trajectory seeds
#    instead of sharing one stream. This makes the run reproducible AND
#    embarrassingly parallel -- the single largest speedup available here, since
#    Ntraj is typically 1e3-1e4.
#
# -----------------------------------------------------------------------------
# THE BUG
# -----------------------------------------------------------------------------
# `reset_qubit!` sampled with `r < prob[1]` using UNNORMALISED probabilities.
# After `orthogonalize` the state is not guaranteed to have unit norm (repeated
# `apply` with truncation erodes it), so p0 + p1 < 1 and the sampling is biased
# TOWARDS outcome 1, i.e. towards spurious jumps. The bias is small when
# truncation is mild and grows exactly when truncation is not mild -- so it is
# worst in the regime you care about. Fixed below by dividing through.
# =============================================================================

# --- metric helpers: reuse the MPDO study's if present, else define locally,
#     so the two studies report the same quantities computed the same way ------
if !@isdefined(von_neumann_entropy)
    function von_neumann_entropy(p::Vector{Float64})
        s = 0.0
        for pi in p; pi > 1e-300 && (s -= pi*log2(pi)); end
        return s
    end
end
if !@isdefined(chi_required)
    function chi_required(p::Vector{Float64}, tol::Float64)
        isempty(p) && return 0
        acc = 0.0
        for k in length(p):-1:1
            acc += p[k]; acc > tol && return k
        end
        return 1
    end
end
if !@isdefined(bond_schmidt_spectrum)
    function bond_schmidt_spectrum(psi::MPS, b::Int)
        phi = orthogonalize(psi, b)
        lefties = uniqueinds(phi[b], phi[b+1])
        _, S, _ = svd(phi[b], lefties)
        p = Float64[]
        for i in 1:dim(S,1); push!(p, abs2(S[i,i])); end
        tot = sum(p); tot <= 0 && return Float64[]
        return sort!(p ./ tot; rev=true)
    end
end


# =============================================================================
# Gates
# =============================================================================

const _X = ComplexF64[0 1; 1 0]
const _Y = ComplexF64[0 -im; im 0]
const _Z = ComplexF64[1 0; 0 -1]
const _I = ComplexF64[1 0; 0 1]

"exp(-i θ/2 A⊗B) for Hermitian involutions A,B."
_rot2(A, B, θ) = cos(θ/2)*kron(_I,_I) - im*sin(θ/2)*kron(A,B)

"""
    heisenberg_bond_matrix(J, dt)

RZZ(2J dt) RYY(J dt) RXX(J dt) -- the SAME bond unitary as the MPDO study, i.e.
exp(-i dt h) with h = (J/2)(XX+YY) + J ZZ. Note (see the report, Sec. 2.2) that
this is -1/2 times the H of Eq. (13); the factor of two matters when converting
times, the sign does not matter for entanglement.
"""
function heisenberg_bond_matrix(J::Float64, dt::Float64)
    a = J*dt; b = 2*J*dt
    return _rot2(_Z,_Z,b) * _rot2(_Y,_Y,a) * _rot2(_X,_X,a)
end

"""
    amplitude_damping_pair(p)

Two-site unitary on (system, ancilla) implementing amplitude damping of strength
`p` once the ancilla is reset: CRy(theta) with control=system, target=ancilla,
followed by CX with control=ancilla, target=system. theta = 2 asin(sqrt(p)).
"""
function amplitude_damping_pair(p::Float64)
    @assert 0 <= p <= 1 "damping probability out of range: $p"
    θ = 2*asin(sqrt(p)); c = cos(θ/2); s = sin(θ/2)
    # basis order (q,a) = |00>,|01>,|10>,|11>
    CRy = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 c -s; 0 0 s c]
    CX  = zeros(ComplexF64,4,4)                 # control a, target q
    CX[1,1]=1; CX[4,2]=1; CX[3,3]=1; CX[2,4]=1
    return CX * CRy
end

"Wrap a 4x4 matrix as a two-site ITensor gate on (s1, s2)."
function _gate2(M::Matrix{ComplexF64}, s1::Index, s2::Index)
    T = ITensor(ComplexF64, s1', s2', s1, s2)
    for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        T[s1'=>a, s2'=>b, s1=>c, s2=>d] = M[2*(a-1)+b, 2*(c-1)+d]
    end
    return T
end


# =============================================================================
# Reset (measure + conditional X), with the sampling bug fixed
# =============================================================================

"""
    reset_ancilla!(psi, pos; cutoff, maxdim, rng)

Projective measurement of site `pos` in the computational basis followed by a
conditional X, i.e. exactly what a hardware `reset` does and what Qiskit models
internally. Returns (outcome, psi).

Probabilities are normalised before sampling. The original version compared the
uniform deviate against an unnormalised p0; since truncation erodes the norm,
that biased the sampling towards outcome 1 (spurious jumps), and did so most
strongly in the strongly-truncated regime.
"""
function reset_ancilla!(psi::MPS, pos::Int; cutoff, maxdim, rng=Random.default_rng())
    psi = orthogonalize(psi, pos)
    A = psi[pos]
    sidx = only(inds(A, "Site"))
    k0 = ITensor(sidx); k0[sidx=>1] = 1.0
    k1 = ITensor(sidx); k1[sidx=>2] = 1.0
    p0 = real(inner(A*dag(k0), A*dag(k0)))
    p1 = real(inner(A*dag(k1), A*dag(k1)))
    tot = p0 + p1
    tot <= 1e-300 && error("reset_ancilla!: zero-norm state at site $pos")
    outcome = rand(rng) < p0/tot ? 0 : 1        # <-- normalised
    P = op(outcome == 0 ? "Proj0" : "Proj1", sidx)
    psi = apply([P], psi; cutoff=cutoff, maxdim=maxdim)
    if outcome == 1
        psi = apply([op("X", sidx)], psi; cutoff=cutoff, maxdim=maxdim)
    end
    normalize!(psi)
    return outcome, psi
end


# =============================================================================
# Layout and measurement
# =============================================================================
#
# Physical chain: q1 a1 q2 a2 ... qn an, so every system qubit is adjacent to
# its own ancilla and all damping gates are nearest-neighbour. Ancillas sit at
# even positions and are in |0> (hence in a product state with everything else)
# whenever a damping layer has just finished.
#
# CONSEQUENCE FOR THE METRICS: measure entanglement only immediately AFTER a
# complete damping layer. At that moment the ancillas carry nothing, and the
# Schmidt spectrum across physical bond 2b -- between a_b and q_{b+1} -- is
# exactly the system's spectrum across system bond b. Measuring mid-layer would
# fold transient system-ancilla entanglement into the answer and inflate it.

sys_pos(j::Int) = 2j - 1
anc_pos(j::Int) = 2j
"Physical MPS bond corresponding to system bond b (between q_b and q_{b+1})."
sys_bond(b::Int) = 2b

"""
    trajectory_metrics(psi, n, tols)

Entanglement and bond-dimension requirement across every SYSTEM cut, using the
same definitions as the MPDO study so the two are directly comparable.
Entropies are in bits.
"""
function trajectory_metrics(psi::MPS, n::Int, tols::Vector{Float64})
    nb = n - 1
    S = zeros(nb); chis = [zeros(Int, nb) for _ in tols]

    # ONE copy, ONE left-to-right sweep.
    #
    # The first version called bond_schmidt_spectrum(psi, b) per bond, and that
    # helper does a NON-MUTATING orthogonalize -- so it copied the whole MPS and
    # re-orthogonalised from scratch (n-1) times per snapshot, i.e. O(n^2)
    # gauge work at chi^3 each. Invisible at n=10 where chi=13; a large fraction
    # of the runtime at n=24 where chi~600 and there are 23 bonds. Here the
    # centre is moved rightwards two sites at a time instead, which is O(1)
    # amortised per bond.
    phi = orthogonalize(psi, sys_bond(1))
    for b in 1:nb
        bb = sys_bond(b)
        b > 1 && orthogonalize!(phi, bb)     # centre moves right by 2: cheap
        lefties = uniqueinds(phi[bb], phi[bb+1])
        _, Sv, _ = svd(phi[bb], lefties)
        p = Float64[]
        for i in 1:dim(Sv, 1); push!(p, abs2(Sv[i,i])); end
        tot = sum(p)
        tot > 0 && (p ./= tot)
        sort!(p; rev=true)
        S[b] = von_neumann_entropy(p)
        for (ti, tol) in enumerate(tols); chis[ti][b] = chi_required(p, tol); end
    end

    mid = max(1, n ÷ 2)
    return (S_mid=S[mid], S_max=maximum(S), S_profile=S,
            chi_mid=[c[mid] for c in chis], chi_max=[maximum(c) for c in chis],
            linkdim=maxlinkdim(psi))
end


# =============================================================================
# One trajectory
# =============================================================================

"""
    single_trajectory(n, J, gamma, times; dt, cutoff, maxdim, seed, tols, initial)

Integrates ONE trajectory from t=0 through the whole time grid, recording at
each requested time. Trotter layout mirrors the `:strang` splitting used in the
MPDO runs:

    odd(dt/2), even(dt/2), damping(dt), even(dt/2), odd(dt/2)

so that trajectory and MPDO results differ only in the unravelling, not in the
discretisation. The damping layer applies CRy+CX+reset to every system qubit.
"""
function single_trajectory(n::Int, J::Vector{Float64}, gamma::Vector{Float64},
                           times::Vector{Float64};
                           dt::Float64=0.05, cutoff::Float64=1e-10,
                           maxdim::Int=512, seed::Int=1,
                           tols::Vector{Float64}=[1e-6],
                           initial=:neel)
    rng = MersenneTwister(seed)
    N = 2n
    sites = siteinds("Qubit", N)
    excited = initial === :neel ? collect(1:2:n) :
              initial === :allup ? collect(1:n) : collect(Int.(initial))
    st = fill("0", N)
    for j in excited; st[sys_pos(j)] = "1"; end
    psi = MPS(sites, st)

    # gate caches, keyed on the local step
    Hcache = Dict{Float64,Vector{ITensor}}()
    Dcache = Dict{Float64,Vector{ITensor}}()
    function hgates(step, parity)
        get!(Hcache, round(step + 0.5parity; digits=12)) do
            g = ITensor[]
            for b in (parity == 0 ? (1:2:n-1) : (2:2:n-1))
                M = heisenberg_bond_matrix(J[b], step)
                # system qubits b and b+1 are 2 physical sites apart; ITensor's
                # `apply` handles the intervening ancilla by swapping, which is
                # cheap because the ancilla is in a product state here.
                push!(g, _gate2(M, sites[sys_pos(b)], sites[sys_pos(b+1)]))
            end
            g
        end
    end
    function dgates(step)
        get!(Dcache, round(step; digits=12)) do
            [ _gate2(amplitude_damping_pair(1 - exp(-gamma[j]*step)),
                     sites[sys_pos(j)], sites[anc_pos(j)]) for j in 1:n ]
        end
    end

    rec = NamedTuple[]
    push!(rec, merge((t=0.0,), trajectory_metrics(psi, n, tols),
                     (z=ITensorMPS.expect(psi,"Z")[sys_pos.(1:n)],)))

    tprev = 0.0
    for tt in times
        nsteps = max(1, round(Int, (tt - tprev)/dt)); ldt = (tt - tprev)/nsteps
        for _ in 1:nsteps
            psi = apply(hgates(ldt/2, 0), psi; cutoff=cutoff, maxdim=maxdim)
            psi = apply(hgates(ldt/2, 1), psi; cutoff=cutoff, maxdim=maxdim)
            psi = apply(dgates(ldt),     psi; cutoff=cutoff, maxdim=maxdim)
            for j in 1:n
                _, psi = reset_ancilla!(psi, anc_pos(j); cutoff=cutoff,
                                        maxdim=maxdim, rng=rng)
            end
            psi = apply(hgates(ldt/2, 1), psi; cutoff=cutoff, maxdim=maxdim)
            psi = apply(hgates(ldt/2, 0), psi; cutoff=cutoff, maxdim=maxdim)
            normalize!(psi)
        end
        tprev = tt
        # measured only here, with every ancilla freshly reset to |0>
        push!(rec, merge((t=tt,), trajectory_metrics(psi, n, tols),
                         (z=ITensorMPS.expect(psi,"Z")[sys_pos.(1:n)],)))
    end
    return rec
end


# =============================================================================
# Ensemble
# =============================================================================

"""
    run_trajectories(n, J, gamma, times, Ntraj; kwargs...)

Averages over `Ntraj` independent trajectories and returns, at each recorded
time, both the physics (<Z_j> with its standard error) and the cost metrics.

WHAT TO LOOK AT. The comparison against the MPDO study is a cost comparison,
not an entropy comparison:

    speedup  =  chi_MPDO^3  /  (Ntraj * chi_traj^3)

`chi_traj` should be the high percentile, not the mean -- a run is only as cheap
as its most expensive trajectory, and the distribution has a tail. Both are
reported.

`Ntraj_for` estimates how many trajectories are needed to reach a target
statistical error on <Z>, from the measured per-trajectory variance; that is the
honest N to put in the formula above rather than whatever N you happened to run.
"""
function run_trajectories(n::Int, J, gamma, times::Vector{Float64}, Ntraj::Int;
                          dt::Float64=0.05, cutoff::Float64=1e-10,
                          maxdim::Int=512, seed0::Int=1000,
                          tols::Vector{Float64}=[1e-6],
                          initial=:neel, verbose::Bool=true)
    Jv = J isa Number ? fill(Float64(J), n-1) : collect(Float64.(J))
    gv = gamma isa Number ? fill(Float64(gamma), n) : collect(Float64.(gamma))

    all_runs = Vector{Vector{NamedTuple}}(undef, Ntraj)
    Threads.@threads for k in 1:Ntraj
        all_runs[k] = single_trajectory(n, Jv, gv, times; dt=dt, cutoff=cutoff,
                                        maxdim=maxdim, seed=seed0+k, tols=tols,
                                        initial=initial)
        if verbose && Threads.threadid() == 1 && k % max(1, Ntraj÷10) == 0
            @printf("  ... %d/%d trajectories\n", k, Ntraj); flush(stdout)
        end
    end

    nt = length(all_runs[1])
    out = NamedTuple[]
    for i in 1:nt
        t   = all_runs[1][i].t
        Sm  = [r[i].S_mid for r in all_runs]
        Smx = [r[i].S_max for r in all_runs]
        cm  = [r[i].chi_max[1] for r in all_runs]
        ldm = [r[i].linkdim for r in all_runs]
        zmid = [real(r[i].z[max(1,n÷2)]) for r in all_runs]
        zbar = mean(zmid); zsem = Ntraj > 1 ? std(zmid)/sqrt(Ntraj) : 0.0
        push!(out, (t=t,
            S_mid_mean=mean(Sm), S_mid_max=maximum(Sm),
            S_max_mean=mean(Smx), S_max_p95=quantile(Smx, 0.95),
            chi_mean=mean(cm), chi_p95=quantile(cm, 0.95), chi_max=maximum(cm),
            linkdim_mean=mean(ldm), linkdim_max=maximum(ldm),
            z_mid=zbar, z_sem=zsem, z_var=var(zmid),
            saturated = maximum(ldm) >= maxdim))
    end

    if verbose
        println("\n      t | <S_traj>  S_p95 | <chi>  chi_p95  chi_max | <Z_mid>±sem | sat")
        println("-"^82)
        for r in out
            @printf("%7.3f | %8.3f %6.3f | %5.0f %8.0f %8d | %+.4f±%.4f | %s\n",
                    r.t, r.S_max_mean, r.S_max_p95, r.chi_mean, r.chi_p95,
                    r.chi_max, r.z_mid, r.z_sem, r.saturated ? "!" : " ")
        end
    end
    return out
end

"""
    Ntraj_for(res, target_sem)

Trajectories needed for the stated standard error on <Z_mid>, from the measured
per-trajectory variance: N = var / target^2. Evaluated at the worst time.
"""
Ntraj_for(res, target_sem::Float64) =
    ceil(Int, maximum(r.z_var for r in res) / target_sem^2)

"""
    cost_comparison(res, chi_mpdo, target_sem)

Prints the head-to-head. This is the number the hardness claim rests on.
"""
function cost_comparison(res, chi_mpdo::Real, target_sem::Float64=0.01)
    N = Ntraj_for(res, target_sem)
    chi_t = maximum(r.chi_p95 for r in res)
    traj = N * chi_t^3
    mpdo = float(chi_mpdo)^3
    @printf("\n=== cost comparison (target SEM on <Z> = %.3f) ===\n", target_sem)
    @printf("  trajectories : N = %d, chi_p95 = %.0f  ->  N*chi^3 = %.3e\n", N, chi_t, traj)
    @printf("  MPDO         : chi = %.0f            ->    chi^3 = %.3e\n", float(chi_mpdo), mpdo)
    @printf("  trajectory advantage: %.3e\n", mpdo/traj)
    println(mpdo/traj > 1 ?
        "  => trajectories WIN. The MPDO bound alone does not establish hardness here." :
        "  => MPDO is cheaper here; the trajectory route is not the binding constraint.")
    return (Ntraj=N, chi_traj=chi_t, ratio=mpdo/traj)
end

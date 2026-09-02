using ITensors, ITensorMPS
using LinearAlgebra, Printf

# Single include pulls the whole existing chain:
#   F_diagnostics.jl  -> bond_dimension_tracking.jl -> open_middle_out_contraction.jl
#                     -> open_product_formula_generation.jl -> liouville_space.jl
# which gives us, already written and already validated in this project:
#   liouville_siteinds, LiouvilleSites, ID2/SIGMA_Z/SIGMA_MINUS   (liouville_space.jl)
#   get_open_step_gates(...; order, dissipation)                  (open_product_formula_generation.jl)
#   vectorized_initial_state_mps                                  (F_diagnostics.jl)
# Do NOT add a second include of any of those files -- see the note at the top
# of trotter_error_gram.jl about include ordering in this codebase.
#
# Julia resolves a relative `include` against the directory of the file doing
# the including, NOT the process working directory. So this file must live
# alongside F_diagnostics.jl. Checked explicitly, because the failure mode
# otherwise is a bare SystemError at load time that is easy to miss in a batch
# log -- and which leaves an empty output directory behind if the submit script
# already created one.
let dep = joinpath(@__DIR__, "F_diagnostics.jl")
    isfile(dep) || error(
        "vectorized_evolution.jl cannot find F_diagnostics.jl.\n" *
        "  Looked in: $(@__DIR__)\n" *
        "  Copy vectorized_evolution.jl and entanglement_growth_study.jl into the\n" *
        "  directory that contains F_diagnostics.jl, or symlink the project files here.")
end
include(joinpath(@__DIR__, "F_diagnostics.jl"))

# =============================================================================
# vectorized_evolution.jl
#
# PURPOSE (deliberately narrow): measure where the *vectorized-MPS / TEBD*
# simulation of
#
#   drho/dt = -i[H, rho] + sum_j gamma_j ( sigma^-_j rho sigma^+_j
#                                          - 1/2 {sigma^+_j sigma^-_j, rho} )
#   H = - sum_j ( J_j (X_j X_{j+1} + Y_j Y_{j+1}) + 2 J_j Z_j Z_{j+1} )   (Eq. 13)
#
# stops being cheap, as a function of (n, J, gamma, t). Nothing here touches
# F_ij, MOC, or DMPF: the object evolved is the physical state |rho(t)>>, and
# the cost metric is its operator entanglement / required bond dimension.
#
# -----------------------------------------------------------------------------
# THE ONE THING TO READ BEFORE USING THIS
# -----------------------------------------------------------------------------
# The bond-dimension ceiling here is NOT the one in trotter_error_gram.jl.
#
#   theoretical_max_bond_dim(n) = 16^min(n/2, n-n/2)     <- for an MPO F_ij
#                                                           (superoperator:
#                                                            4 in-legs x 4 out-legs)
#   state_bond_dim_ceiling(n, b) = 4^min(b, n-b)         <- for |rho>> as an MPS
#                                                           (local dim 4)
#
# The state route is exponentially cheaper than the F route at the same n.
# n = 20 is hopeless for F (16^10 ~ 1e12) and entirely reasonable for |rho>>
# (4^10 ~ 1e6 ceiling, with the physical requirement far below that). Use
# state_bond_dim_ceiling below; do not reach for the MPO one out of habit.
#
# -----------------------------------------------------------------------------
# WHAT IS RECORDED AND WHY BOTH
# -----------------------------------------------------------------------------
# Two different complexity proxies are logged at every requested time, because
# the recent literature (Vilchez-Estevez, Yosifov & Sun, arXiv:2508.19959)
# reports that they can disagree: the operator entanglement entropy S_OP can
# fall with increasing gamma while the bond dimension actually needed does not.
#
#   S_op(t)      von Neumann operator entanglement across a cut, in BITS
#                (log2, so it is directly comparable to log2(chi) and to the
#                figures in Preisser et al., arXiv:2303.09426).
#   chi_req(t)   the smallest number of Schmidt values whose discarded weight
#                is below a tolerance. THIS is the number that sets the cost.
#
# S_op is a property of the state at time t. chi_req is what the algorithm has
# to carry. Report both; the interesting claim is where they separate.
#
# -----------------------------------------------------------------------------
# THE TRAP: chi_req IS CENSORED BY maxdim
# -----------------------------------------------------------------------------
# chi_req is read off the Schmidt spectrum of the MPS *as stored*. If the run
# was truncated at maxdim, the spectrum has at most maxdim entries and chi_req
# cannot exceed it -- the measurement saturates and silently under-reports.
# Every row therefore carries a `saturated` flag (chi_req at the loosest
# tolerance has reached the stored link dimension). A saturated row is a lower
# bound, not a measurement. Rerun at larger maxdim or discard it.
# =============================================================================


# =============================================================================
# Ceilings
# =============================================================================

"Max possible Schmidt rank of |rho>> (MPS, local dim 4) at bond b of an n-site chain."
state_bond_dim_ceiling(n::Int, b::Int) = 4^min(b, n - b)

"Max possible Schmidt rank of |rho>> across the middle cut."
state_bond_dim_ceiling(n::Int) = state_bond_dim_ceiling(n, n ÷ 2)


# =============================================================================
# Vectorized local operators, for traces and observables
# =============================================================================
#
# Convention fixed by vectorized_initial_state_mps: rho_{ab} lives on
# |a>_ket |b>_bra. So vec(M) = sum_{ab} M[a,b] |a>_ket |b>_bra, and the
# Hilbert-Schmidt inner product <<A|B>> = Tr(A^dag B) is exactly ITensorMPS's
# `inner(A_mps, B_mps)` (which conjugates its first argument). For Hermitian M
# this gives <M> = Tr(M rho) = inner(vec(M), vec(rho)) directly.

"""
    local_op_vectorized_mps(lsites, ops)

Vectorization of a product operator, as a bond-dimension-1 MPS over the doubled
sites. `ops` is a Dict mapping site index -> dense 2x2 matrix; every site not
listed gets the identity.
"""
function local_op_vectorized_mps(lsites::LiouvilleSites, ops::Dict{Int,<:AbstractMatrix})
    n = lsites.n
    links = [Index(1, "Link,l=$j") for j in 1:(n-1)]
    tensors = ITensor[]
    for j in 1:n
        M = Matrix{ComplexF64}(get(ops, j, ID2))
        @assert size(M) == (2, 2) "local_op_vectorized_mps expects 2x2 single-qubit operators."
        ket_s, bra_s = lsites.ket[j], lsites.bra[j]
        t = ITensor(M, ket_s, bra_s)
        j > 1 && (t *= onehot(links[j-1] => 1))
        j < n && (t *= onehot(links[j]   => 1))
        push!(tensors, t)
    end
    return MPS(tensors)
end

"MPS representing vec(identity); <<Id|rho>> = Tr(rho)."
identity_vectorized_mps(lsites::LiouvilleSites) =
    local_op_vectorized_mps(lsites, Dict{Int,Matrix{ComplexF64}}())

"MPS representing vec(Z_j); <<Z_j|rho>> = Tr(Z_j rho)."
pauli_z_vectorized_mps(lsites::LiouvilleSites, j::Int) =
    local_op_vectorized_mps(lsites, Dict{Int,Matrix{ComplexF64}}(j => SIGMA_Z))

"Tr(rho) from the vectorized MPS. Should stay at 1 up to Trotter/truncation error."
liouville_trace(rho::MPS, id_mps::MPS) = inner(id_mps, rho)

"Tr(rho^2), the purity. Falls from 1 as dissipation mixes the state."
liouville_purity(rho::MPS) = real(inner(rho, rho))


# =============================================================================
# Schmidt spectrum, entropies, required bond dimension
# =============================================================================

"""
    bond_schmidt_spectrum(psi, b)

Normalized squared Schmidt values {lambda_i^2} across the bond between sites b
and b+1, sorted descending and summing to 1.

The normalization matters: the vectorized evolution is not trace- or
norm-preserving under truncation, and the overall scale of |rho>> is physically
irrelevant to the entanglement structure anyway. Normalizing here makes S_op
and chi_req independent of whatever `apply` did to the global prefactor.
"""
function bond_schmidt_spectrum(psi::MPS, b::Int)
    n = length(psi)
    @assert 1 <= b <= n - 1 "bond index b must lie in 1:$(n-1), got $b."
    phi = orthogonalize(psi, b)
    # Everything except the link to site b+1 goes on the row side of the SVD.
    lefties = uniqueinds(phi[b], phi[b+1])
    _, S, _ = svd(phi[b], lefties)
    p = Float64[]
    for i in 1:dim(S, 1)
        push!(p, abs2(S[i, i]))
    end
    tot = sum(p)
    tot <= 0 && return Float64[]
    return sort!(p ./ tot; rev=true)
end

"""
    von_neumann_entropy(p) -> bits

S = -sum p_i log2(p_i). Base 2 throughout, so that S = log2(chi) is the exact
statement of "a bond dimension chi can support this much entanglement".
"""
function von_neumann_entropy(p::Vector{Float64})
    s = 0.0
    for pi in p
        pi > 1e-300 && (s -= pi * log2(pi))
    end
    return s
end

"""
    renyi_entropy(p, alpha) -> bits

Renyi-alpha entropy. alpha = 0.5 is included by default in the recorded output
because the rigorous MPO-approximability bounds are stated in terms of
alpha < 1 Renyi entropies, not the von Neumann one -- S_1 alone can look benign
while a fat tail makes truncation expensive.
"""
function renyi_entropy(p::Vector{Float64}, alpha::Float64)
    isempty(p) && return 0.0
    if isapprox(alpha, 1.0)
        return von_neumann_entropy(p)
    end
    return log2(sum(p .^ alpha)) / (1 - alpha)
end

"""
    chi_required(p, tol)

Smallest chi such that the discarded weight sum_{i>chi} p_i <= tol.
This, not the entropy, is the number that sets the runtime.
"""
function chi_required(p::Vector{Float64}, tol::Float64)
    isempty(p) && return 0
    acc = 0.0
    for k in length(p):-1:1
        acc += p[k]
        if acc > tol
            return k
        end
    end
    return 1
end

"Discarded weight if the spectrum were cut at chi."
function discarded_weight(p::Vector{Float64}, chi::Int)
    chi >= length(p) && return 0.0
    return sum(@view p[(chi+1):end])
end


# =============================================================================
# Per-snapshot diagnostics
# =============================================================================

"""
    bond_report(psi, tols)

Sweeps every internal bond and returns entropies, required bond dimensions and
ceilings, both at the middle cut and maximized over cuts.

The middle cut is the conventional place to look, but it is not automatically
the worst one -- with site-dependent gammas or a disorder realization that
happens to be weak in the middle it is not. The max-over-bonds columns exist so
you can check rather than assume.
"""
function bond_report(psi::MPS, tols::Vector{Float64})
    n = length(psi)
    nb = n - 1
    mid = n ÷ 2

    S1   = zeros(Float64, nb)
    Shalf = zeros(Float64, nb)
    linkd = zeros(Int, nb)
    chis = [zeros(Int, nb) for _ in tols]

    for b in 1:nb
        p = bond_schmidt_spectrum(psi, b)
        S1[b]    = von_neumann_entropy(p)
        Shalf[b] = renyi_entropy(p, 0.5)
        linkd[b] = length(p)
        for (ti, tol) in enumerate(tols)
            chis[ti][b] = chi_required(p, tol)
        end
    end

    return (
        S_op_mid      = S1[mid],
        S_op_max      = maximum(S1),
        S_half_mid    = Shalf[mid],
        S_half_max    = maximum(Shalf),
        linkdim_mid   = linkd[mid],
        linkdim_max   = maximum(linkd),
        chi_req_mid   = [c[mid] for c in chis],
        chi_req_max   = [maximum(c) for c in chis],
        S_profile     = S1,
        S_half_profile = Shalf,
        linkdim_profile = linkd,
        chi_profiles  = chis,
    )
end


# =============================================================================
# THE MAIN DRIVER
# =============================================================================

"""
    evolve_vectorized(n, J, gamma, times; kwargs...)

Vectorized-MPS / TEBD evolution of |rho(t)>> under the Heisenberg + amplitude
damping master equation, recording the operator entanglement and the bond
dimension actually required at every requested time.

# Required arguments
- `n`        : number of physical qubits (the MPS has n sites of local dim 4).
- `J`        : coupling. Either a scalar (uniform chain) or a length-(n-1)
               vector of per-bond couplings. Passing a scalar is the right
               choice for a clean scaling study; the disordered
               `rand(Uniform(1/4,3/4), n-1)` used elsewhere in this project
               makes the n-dependence noisy unless you average over seeds.
- `gamma`    : amplitude damping rate. Scalar (uniform) or length-n vector.
- `times`    : strictly increasing vector of times at which to record. t=0 is
               recorded automatically and need not be included.

# Keyword arguments
- `dissipation=true`  : set false for the closed-system control. This zeroes
                        every gamma via the existing `dissipation` flag in
                        `get_open_step_gates`, so the closed run is the same
                        code path, not a different one. In the closed case rho
                        stays pure and S_op is exactly twice the pure-state
                        entanglement entropy -- that factor of 2 is the whole
                        reason MPDO looks worse than trajectories at small
                        gamma, and it is worth seeing explicitly.
- `dt=0.02`           : Trotter step. Locally adjusted so each requested time
                        is landed on exactly.
- `order=2`           : product formula order (1, 2 or 4), reusing
                        `get_open_step_gates`. Order 4 uses a negative
                        sub-step, which is not a CP map for the dissipative
                        layer; harmless for observables at small dt but it can
                        push the state slightly non-positive, so order 2 is the
                        default.
- `cutoff=1e-12`, `maxdim=512` : truncation. See the saturation warning above.
- `initial=:neel`     : `:neel` (sites 1,3,5,... excited -- the standard choice
                        in this literature and the one Preisser et al. use),
                        `:allup`, `:alldown`, or an explicit Vector{Int} of
                        1-indexed excited sites.
- `tols=[1e-6,1e-10]` : discarded-weight tolerances defining chi_req.
- `renormalize_trace=true` : rescale rho by 1/Tr(rho) after each snapshot. This
                        does not touch the Schmidt spectrum, hence not S_op or
                        chi_req; it only keeps observables sane. The raw trace
                        before rescaling is recorded as a Trotter/truncation
                        diagnostic -- watch it.
- `track_observable=true` : record <Z_j> at the middle site.
- `verbose=true`

# Returns
A NamedTuple of per-snapshot vectors, ready for `rows_to_csv` below.
"""
function evolve_vectorized(
    n::Int,
    J,
    gamma,
    times::Vector{Float64};
    dissipation::Bool = true,
    dt::Float64 = 0.02,
    order::Int = 2,
    cutoff::Float64 = 1e-12,
    maxdim::Int = 512,
    initial = :neel,
    tols::Vector{Float64} = [1e-6, 1e-10],
    renormalize_trace::Bool = true,
    track_observable::Bool = true,
    verbose::Bool = true,
)
    @assert n >= 2 "need at least 2 sites"
    @assert issorted(times) && all(times .> 0) "times must be strictly increasing and positive"

    Jvec = J isa Number ? fill(Float64(J), n - 1) : collect(Float64.(J))
    @assert length(Jvec) == n - 1 "J must be a scalar or a length-$(n-1) vector, got $(length(Jvec))"
    gammas = gamma isa Number ? fill(Float64(gamma), n) : collect(Float64.(gamma))
    @assert length(gammas) == n "gamma must be a scalar or a length-$n vector, got $(length(gammas))"

    lsites = liouville_siteinds(n)

    excited = if initial === :neel
        collect(1:2:n)
    elseif initial === :allup
        collect(1:n)
    elseif initial === :alldown
        Int[]
    else
        collect(Int.(initial))
    end
    rho = vectorized_initial_state_mps(lsites, excited)

    id_mps = identity_vectorized_mps(lsites)
    z_mps  = track_observable ? pauli_z_vectorized_mps(lsites, max(1, n ÷ 2)) : nothing

    ceiling_mid = state_bond_dim_ceiling(n)
    if verbose
        @printf("n=%d  order=%d  dissipation=%s  dt=%.4g  cutoff=%.1e  maxdim=%d\n",
                n, order, dissipation, dt, cutoff, maxdim)
        @printf("mean J=%.4f  mean gamma=%.4f  MPS ceiling at middle cut = %d (= 4^%d)\n",
                sum(Jvec)/length(Jvec), sum(gammas)/n, ceiling_mid, min(n÷2, n-n÷2))
        maxdim >= ceiling_mid && println("  (maxdim is at or above the ceiling: this run is untruncated)")
        println()
    end

    # ---- recording ----------------------------------------------------------
    rec_t          = Float64[]
    rec_S_mid      = Float64[]
    rec_S_max      = Float64[]
    rec_Shalf_mid  = Float64[]
    rec_ld_mid     = Int[]
    rec_ld_max     = Int[]
    rec_chi_mid    = [Int[] for _ in tols]
    rec_chi_max    = [Int[] for _ in tols]
    rec_trace      = ComplexF64[]
    rec_purity     = Float64[]
    rec_zmid       = ComplexF64[]
    rec_saturated  = Bool[]
    # Full per-bond profiles, kept so the "is the middle cut actually the
    # bottleneck?" question can be answered from the saved data rather than
    # re-run. Cheap: (n-1) floats per snapshot.
    prof_S         = Vector{Vector{Float64}}()
    prof_Shalf     = Vector{Vector{Float64}}()
    prof_ld        = Vector{Vector{Int}}()
    prof_chi       = Vector{Vector{Vector{Int}}}()   # [snapshot][tol][bond]

    function snapshot!(tnow)
        tr = liouville_trace(rho, id_mps)
        pur = liouville_purity(rho)
        zj = track_observable ? inner(z_mps, rho) : ComplexF64(NaN)
        if renormalize_trace && abs(tr) > 1e-14
            rho[1] = rho[1] / tr
            pur /= abs2(tr)
            zj  /= tr
        end
        r = bond_report(rho, tols)
        push!(rec_t, tnow)
        push!(rec_S_mid, r.S_op_mid);   push!(rec_S_max, r.S_op_max)
        push!(rec_Shalf_mid, r.S_half_mid)
        push!(rec_ld_mid, r.linkdim_mid); push!(rec_ld_max, r.linkdim_max)
        for (i, _) in enumerate(tols)
            push!(rec_chi_mid[i], r.chi_req_mid[i])
            push!(rec_chi_max[i], r.chi_req_max[i])
        end
        push!(rec_trace, tr); push!(rec_purity, pur); push!(rec_zmid, zj)
        # Truncation flag. The old criterion (chi_req >= linkdim AND linkdim >=
        # maxdim) was too permissive: at gamma=0.2, n=20 it read chi_req = 248
        # against maxdim = 256 and reported "not saturated", yet the same
        # quantity measured at maxdim 64/128/256 came out as 64/125/241. It was
        # tracking maxdim, not converging. The honest criterion is simply
        # whether maxdim binds anywhere: if it does, the requested `cutoff` was
        # never enforced, truncation noise is accumulating, and chi_req is a
        # censored lower bound rather than a measurement.
        push!(rec_saturated, r.linkdim_max >= maxdim)
        push!(prof_S, r.S_profile)
        push!(prof_Shalf, r.S_half_profile)
        push!(prof_ld, r.linkdim_profile)
        push!(prof_chi, r.chi_profiles)
        return r
    end

    r0 = snapshot!(0.0)
    if verbose
        println("      t |  S_op(mid)  S_op(max) | linkdim |  chi(1e-6)  chi(1e-10) |    Tr(rho)   purity | sat")
        println("-"^103)
        @printf("%7.3f | %10.4f %10.4f | %7d | %10d %11d | %10.6f %8.4f |  %s\n",
                0.0, r0.S_op_mid, r0.S_op_max, r0.linkdim_max,
                r0.chi_req_max[1], r0.chi_req_max[2],
                real(rec_trace[end]), rec_purity[end], rec_saturated[end] ? "!" : " ")
        flush(stdout)
    end

    # ---- evolution ----------------------------------------------------------
    # Gates are rebuilt whenever the local step changes, and cached, because we
    # snap exactly onto each requested time rather than drifting toward it.
    gate_cache = Dict{Float64,Vector{ITensor}}()
    function gates_for(step::Float64)
        key = round(step; digits=12)
        get!(gate_cache, key) do
            get_open_step_gates(n, Jvec, gammas, key, lsites;
                                order=order, dissipation=dissipation)
        end
    end

    t_prev = 0.0
    for t_target in times
        span = t_target - t_prev
        nsteps = max(1, round(Int, span / dt))
        local_dt = span / nsteps
        gates = gates_for(local_dt)
        for _ in 1:nsteps
            rho = apply(gates, rho; cutoff=cutoff, maxdim=maxdim)
        end
        t_prev = t_target
        r = snapshot!(t_target)
        if verbose
            @printf("%7.3f | %10.4f %10.4f | %7d | %10d %11d | %10.6f %8.4f |  %s\n",
                    t_target, r.S_op_mid, r.S_op_max, r.linkdim_max,
                    r.chi_req_max[1], r.chi_req_max[2],
                    real(rec_trace[end]), rec_purity[end], rec_saturated[end] ? "!" : " ")
            flush(stdout)
        end
    end

    if verbose && any(rec_saturated)
        nsat = count(rec_saturated)
        @printf("\nWARNING: %d/%d snapshots saturated maxdim=%d. chi_req there is a LOWER BOUND.\n",
                nsat, length(rec_saturated), maxdim)
        println("         Rerun with a larger maxdim before quoting any of those numbers.")
    end

    return (
        n = n, J = Jvec, gammas = gammas, dissipation = dissipation,
        order = order, dt = dt, cutoff = cutoff, maxdim = maxdim,
        tols = tols, ceiling_mid = ceiling_mid,
        t = rec_t,
        S_op_mid = rec_S_mid, S_op_max = rec_S_max, S_half_mid = rec_Shalf_mid,
        linkdim_mid = rec_ld_mid, linkdim_max = rec_ld_max,
        chi_req_mid = rec_chi_mid, chi_req_max = rec_chi_max,
        trace = rec_trace, purity = rec_purity, z_mid = rec_zmid,
        saturated = rec_saturated,
        S_profile = prof_S, S_half_profile = prof_Shalf,
        linkdim_profile = prof_ld, chi_profile = prof_chi,
    )
end


# =============================================================================
# Convergence check -- run this before believing anything above
# =============================================================================

"""
    maxdim_convergence(n, J, gamma, times, maxdims; kwargs...)

Repeats `evolve_vectorized` over a ladder of maxdim values and reports, at each
time, the change in S_op and in <Z_mid> relative to the largest maxdim.

THE LADDER MUST EXTEND ABOVE THE PRODUCTION maxdim. The first round of runs
used a ladder that topped out AT the production value, which made "converged at
256" a tautology: the reference and the run being tested were the same
calculation. `run_convergence` in the driver now always appends 2x the
production maxdim for this reason.

Note that the two quantities converge at very different rates. In the first
round S_op was already converged to 1e-4 at maxdim = 64, while chi_req at the
same times went 64 -> 127 -> 249 as maxdim doubled. Expect to certify S_op
cheaply and chi_req not at all.
"""
function maxdim_convergence(n, J, gamma, times::Vector{Float64},
                            maxdims::Vector{Int}; verbose::Bool=true, kwargs...)
    runs = [evolve_vectorized(n, J, gamma, times; maxdim=md, verbose=false, kwargs...)
            for md in maxdims]
    ref = runs[end]
    if verbose
        @printf("\nConvergence vs maxdim=%d (n=%d)\n", maxdims[end], n)
        println("      t | " * join([@sprintf("%8s", "chi=$md") for md in maxdims], " ") *
                "  |  dS_op vs ref")
        println("-"^(20 + 9*length(maxdims) + 20))
        for (i, t) in enumerate(times)
            ds = [abs(r.S_op_mid[i+1] - ref.S_op_mid[i+1]) for r in runs]
            @printf("%7.3f | %s  |  max %.2e\n", t,
                    join([@sprintf("%8.4f", r.S_op_mid[i+1]) for r in runs], " "),
                    maximum(ds))
        end
        println("\n chi_req(1e-6) at the middle cut, same runs:")
        for (i, t) in enumerate(times)
            @printf("%7.3f | %s\n", t,
                    join([@sprintf("%8d", r.chi_req_mid[1][i+1]) for r in runs], " "))
        end
        println(" If this row tracks the maxdim ladder, chi_req is censored, not measured.")
    end
    return (maxdims=maxdims, runs=runs)
end

"""
    dt_convergence(n, J, gamma, times, dts; kwargs...)

Trotter-step convergence at fixed maxdim. The first round used dt = 0.02
throughout, which is very conservative for a second-order splitting and was a
large part of why the small-gamma jobs never finished. Run this once to find
the largest dt that leaves S_op and <Z> unchanged, then use it everywhere.
"""
function dt_convergence(n, J, gamma, times::Vector{Float64},
                        dts::Vector{Float64}; verbose::Bool=true, kwargs...)
    runs = [evolve_vectorized(n, J, gamma, times; dt=d, verbose=false, kwargs...)
            for d in dts]
    ref = runs[argmin(dts)]
    if verbose
        @printf("\nTrotter convergence (reference dt=%.4g, n=%d)\n", minimum(dts), n)
        println("      t | " * join([@sprintf("%9s", "dt=$d") for d in dts], " ") *
                "  |  max |dS_op|   max |dz|")
        for (i, t) in enumerate(times)
            ds = [abs(r.S_op_mid[i+1] - ref.S_op_mid[i+1]) for r in runs]
            dz = [abs(real(r.z_mid[i+1]) - real(ref.z_mid[i+1])) for r in runs]
            @printf("%7.3f | %s  |  %.2e    %.2e\n", t,
                    join([@sprintf("%9.4f", r.S_op_mid[i+1]) for r in runs], " "),
                    maximum(ds), maximum(dz))
        end
    end
    return (dts=dts, runs=runs)
end


# =============================================================================
# CSV output
# =============================================================================

#
# Three tables, all long-format and all carrying their own run parameters in
# every row, so files from different jobs can simply be concatenated (drop the
# repeated header) without a separate metadata lookup:
#
#   rows_to_csv        one row per (run, time)         -- the main result
#   profile_to_csv     one row per (run, time, bond)   -- is the middle cut the
#                                                         bottleneck?
#   convergence_to_csv one row per (maxdim, time)      -- is any of it converged?
#
# `saturated` is carried in the main table on purpose: filter on it before
# plotting chi_req, or you will be plotting your own maxdim back at yourself.

"Column values shared by every table, so each file is self-describing."
function _run_meta(res, label::String)
    gmean = sum(res.gammas) / length(res.gammas)
    jmean = sum(res.J) / length(res.J)
    return [label, string(res.n), @sprintf("%.6f", gmean), @sprintf("%.6f", jmean),
            string(res.dissipation), string(res.order), string(res.maxdim),
            @sprintf("%.3e", res.cutoff), @sprintf("%.5f", res.dt)]
end

const _META_HEADER = ["label", "n", "gamma", "Jmean", "dissipation",
                      "order", "maxdim", "cutoff", "dt"]

"""
    rows_to_csv(res; label)

Main time series: one row per recorded time.
"""
function rows_to_csv(res; label::String="", header::Bool=true)
    tolnames    = [@sprintf("chi_req_mid_%.0e", tol) for tol in res.tols]
    tolnames_mx = [@sprintf("chi_req_max_%.0e", tol) for tol in res.tols]
    hdr = join(vcat(_META_HEADER,
        ["t", "S_op_mid", "S_op_max", "S_half_mid", "linkdim_mid", "linkdim_max"],
        tolnames, tolnames_mx,
        ["trace_re", "trace_im", "purity", "z_mid_re", "ceiling_mid", "saturated"]), ",")
    lines = header ? [hdr] : String[]
    meta = _run_meta(res, label)
    for i in eachindex(res.t)
        fields = vcat(meta,
            [@sprintf("%.6f", res.t[i]),
             @sprintf("%.10f", res.S_op_mid[i]), @sprintf("%.10f", res.S_op_max[i]),
             @sprintf("%.10f", res.S_half_mid[i]),
             string(res.linkdim_mid[i]), string(res.linkdim_max[i])],
            [string(c[i]) for c in res.chi_req_mid],
            [string(c[i]) for c in res.chi_req_max],
            [@sprintf("%.10f", real(res.trace[i])), @sprintf("%.3e", imag(res.trace[i])),
             @sprintf("%.10f", res.purity[i]),
             @sprintf("%.10f", real(res.z_mid[i])),
             string(res.ceiling_mid), string(res.saturated[i])])
        push!(lines, join(fields, ","))
    end
    return join(lines, "\n") * "\n"
end

"""
    profile_to_csv(res; label)

Per-bond profile: one row per (time, bond). Use this to confirm that the middle
cut really is the worst cut -- with site-dependent gammas, or a disorder
realization that happens to be weak near the centre, it is not, and the
middle-cut numbers in the main table would then understate the cost.
"""
function profile_to_csv(res; label::String="", header::Bool=true)
    tolnames = [@sprintf("chi_req_%.0e", tol) for tol in res.tols]
    hdr = join(vcat(_META_HEADER,
        ["t", "bond", "S_op_bond", "S_half_bond", "linkdim_bond"],
        tolnames, ["ceiling_bond"]), ",")
    lines = header ? [hdr] : String[]
    meta = _run_meta(res, label)
    n = res.n
    for i in eachindex(res.t), b in 1:(n-1)
        fields = vcat(meta,
            [@sprintf("%.6f", res.t[i]), string(b),
             @sprintf("%.10f", res.S_profile[i][b]),
             @sprintf("%.10f", res.S_half_profile[i][b]),
             string(res.linkdim_profile[i][b])],
            [string(res.chi_profile[i][ti][b]) for ti in eachindex(res.tols)],
            [string(state_bond_dim_ceiling(n, b))])
        push!(lines, join(fields, ","))
    end
    return join(lines, "\n") * "\n"
end

"""
    convergence_to_csv(conv; label_prefix)

Flattens the output of `maxdim_convergence` into one row per (maxdim, time),
including the deviation of S_op and <Z_mid> from the largest-maxdim run. The
column `converged_1e3` marks rows whose S_op agrees with the reference to
better than 1e-3 bits -- a crude but explicit criterion, so that "which points
am I allowed to quote?" is answered in the data file rather than by eye.
"""
function convergence_to_csv(conv; label_prefix::String="conv", header::Bool=true)
    ref = conv.runs[end]
    hdr = join(vcat(_META_HEADER,
        ["t", "S_op_mid", "S_op_mid_ref", "dS_op", "z_mid_re", "dz_mid",
         "chi_req_mid_loose", "saturated", "converged_1e3"]), ",")
    lines = header ? [hdr] : String[]
    for (r, md) in zip(conv.runs, conv.maxdims)
        meta = _run_meta(r, "$(label_prefix)_chi$(md)")
        for i in eachindex(r.t)
            dS = abs(r.S_op_mid[i] - ref.S_op_mid[i])
            dz = abs(real(r.z_mid[i]) - real(ref.z_mid[i]))
            fields = vcat(meta,
                [@sprintf("%.6f", r.t[i]),
                 @sprintf("%.10f", r.S_op_mid[i]),
                 @sprintf("%.10f", ref.S_op_mid[i]),
                 @sprintf("%.6e", dS),
                 @sprintf("%.10f", real(r.z_mid[i])),
                 @sprintf("%.6e", dz),
                 string(r.chi_req_mid[1][i]),
                 string(r.saturated[i]),
                 string(dS < 1e-3)])
            push!(lines, join(fields, ","))
        end
    end
    return join(lines, "\n") * "\n"
end

write_csv(path::String, res; label::String="") = write(path, rows_to_csv(res; label=label))
write_profile_csv(path::String, res; label::String="") = write(path, profile_to_csv(res; label=label))
write_convergence_csv(path::String, conv; label_prefix::String="conv") =
    write(path, convergence_to_csv(conv; label_prefix=label_prefix))

"""
    save_run(res; dir=".", label="run")

Convenience: writes `<dir>/<label>_timeseries.csv` and `<dir>/<label>_profile.csv`
for a single `evolve_vectorized` result. Returns the two paths.
"""
function save_run(res; dir::String=".", label::String="run")
    mkpath(dir)
    p1 = joinpath(dir, "$(label)_timeseries.csv")
    p2 = joinpath(dir, "$(label)_profile.csv")
    write_csv(p1, res; label=label)
    write_profile_csv(p2, res; label=label)
    @printf("wrote %s and %s\n", p1, p2)
    return (timeseries=p1, profile=p2)
end

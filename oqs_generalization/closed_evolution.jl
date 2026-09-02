using ITensors, ITensorMPS
using LinearAlgebra, Printf

# LOAD ORDER: this file reuses helpers defined elsewhere and does NOT include
# them itself, to avoid a second include of the project chain (see the note at
# the top of trotter_error_gram.jl). It needs
#   heisenberg_bond_unitary, _matrix_to_gate   from the project chain, and
#   bond_schmidt_spectrum, von_neumann_entropy, renyi_entropy, chi_required
#                                              from vectorized_evolution.jl
# so vectorized_evolution.jl must be included BEFORE this file. Checked here,
# because the alternative failure mode is an UndefVarError several minutes into
# a batch job.
for _sym in (:heisenberg_bond_unitary, :_matrix_to_gate, :bond_schmidt_spectrum,
             :von_neumann_entropy, :renyi_entropy, :chi_required)
    isdefined(@__MODULE__, _sym) || error(
        "closed_evolution.jl requires `$_sym`, which is not defined yet.\n" *
        "  Include vectorized_evolution.jl BEFORE closed_evolution.jl.")
end

# =============================================================================
# closed_evolution.jl -- the gamma = 0 baseline, done properly
#
# WHY THIS FILE EXISTS
#
# The first round of runs computed the closed-system control by setting
# `dissipation=false` in the Liouville-space evolution, i.e. by propagating
# |rho>> as an MPS of local dimension 4. That was wasteful and, at n >= 12,
# simply wrong: with maxdim=256 the purity Tr(rho^2) climbed to 1.69 at n=12,
# 4.19 at n=16 and 24.2 at n=20. For a state that is pure by construction the
# purity is identically 1, so those runs were unphysical past t ~ 2.5 and the
# closed curves at n >= 12 had to be discarded.
#
# The fix is to stop simulating a pure state as a density matrix. For gamma = 0
# the state stays pure, so we evolve |psi> directly:
#
#   * local dimension 2 instead of 4,
#   * bond dimension chi instead of chi^2 for the same physical accuracy,
#   * purity is exactly 1 by construction and cannot be violated,
#   * energy conservation becomes available as a sharp truncation diagnostic.
#
# and we then RECONSTRUCT the operator-space quantities exactly, rather than
# measuring them. For rho = |psi><psi| the operator Schmidt decomposition
# across a cut is the outer product of the state's Schmidt decomposition with
# itself, so if the pure state has normalized squared Schmidt values {p_i},
# the vectorized density matrix has exactly {p_i p_j}. Therefore
#
#   S_op       = 2 * S_vN                    (exactly, not approximately)
#   S_op^(1/2) = 2 * S^(1/2)_vN              (same argument, Renyi factorizes)
#   chi_op(tol) = |{(i,j) kept}| from sorting the product spectrum
#
# The last line is the useful one: it gives the bond dimension an MPDO code
# WOULD have needed, computed from a calculation that never paid the MPDO cost.
# That is a converged number where the direct route gave a censored one.
#
# GATE CONVENTION -- worth one look before trusting the energy diagnostic.
# The bond gate is RZZ(beta)*RYY(alpha)*RXX(alpha) with alpha = J*dt and
# beta = 2*J*dt, i.e. exp(-i*dt*h) with
#     h_j = (J_j/2)(X_j X_{j+1} + Y_j Y_{j+1}) + J_j Z_j Z_{j+1}.
# Note this is -1/2 times the H of Eq. 13 in main.pdf (opposite overall sign,
# and half the magnitude). The sign is irrelevant for entanglement, but the
# factor of 2 means the effective time here is half what a naive reading of
# Eq. 13 would suggest -- relevant if you are converting to the tJ units used
# in Preisser et al. The energy MPO below deliberately matches the GATES, not
# the paper, so that conservation is a real test of what is being simulated.
# =============================================================================


# -----------------------------------------------------------------------------
# Gates (pure-state versions of the Liouville layers, same splitting)
# -----------------------------------------------------------------------------

function closed_odd_layer(n::Int, J::Vector{Float64}, dt::Float64, s)
    gates = ITensor[]
    for j in 1:2:n-1
        U = heisenberg_bond_unitary(J[j]*dt, 2*J[j]*dt)
        push!(gates, _matrix_to_gate(Matrix{ComplexF64}(U), [s[j], s[j+1]]))
    end
    return gates
end

function closed_even_layer(n::Int, J::Vector{Float64}, dt::Float64, s)
    gates = ITensor[]
    n <= 2 && return gates
    for j in 2:2:n-1
        U = heisenberg_bond_unitary(J[j]*dt, 2*J[j]*dt)
        push!(gates, _matrix_to_gate(Matrix{ComplexF64}(U), [s[j], s[j+1]]))
    end
    return gates
end

"Same Trotter splitting as get_open_step_gates, with the dissipator layer absent."
function closed_step_gates(n::Int, J::Vector{Float64}, dt::Float64, s; order::Int=2)
    if order == 1
        return vcat(closed_odd_layer(n, J, dt, s), closed_even_layer(n, J, dt, s))
    elseif order == 2
        return vcat(closed_odd_layer(n, J, dt/2, s),
                    closed_even_layer(n, J, dt, s),
                    closed_odd_layer(n, J, dt/2, s))
    elseif order == 4
        p1 = 1 / (4 - 4^(1/3)); p2 = 1 - 4p1
        return vcat(closed_step_gates(n, J, p1*dt, s; order=2),
                    closed_step_gates(n, J, p1*dt, s; order=2),
                    closed_step_gates(n, J, p2*dt, s; order=2),
                    closed_step_gates(n, J, p1*dt, s; order=2),
                    closed_step_gates(n, J, p1*dt, s; order=2))
    end
    error("order must be 1, 2 or 4, got $order")
end

"MPO for the Hamiltonian the GATES actually implement (see header note)."
function closed_hamiltonian_mpo(n::Int, J::Vector{Float64}, s)
    os = OpSum()
    for j in 1:n-1
        os += J[j]/2, "X", j, "X", j+1
        os += J[j]/2, "Y", j, "Y", j+1
        os += J[j],   "Z", j, "Z", j+1
    end
    return MPO(os, s)
end


# -----------------------------------------------------------------------------
# Exact operator-space spectrum from the pure-state spectrum
# -----------------------------------------------------------------------------

"""
    operator_spectrum_from_pure(p) -> sorted {p_i p_j}

For rho = |psi><psi|, the operator Schmidt values across a cut are the pairwise
products of the state's Schmidt values. Exact, not an approximation.

Truncated to the largest `cap` entries when the outer product would be huge;
`cap` is chosen well above any chi we would ever quote, and the discarded mass
is reported so the truncation can be checked rather than assumed.
"""
function operator_spectrum_from_pure(p::Vector{Float64}; cap::Int=200_000)
    isempty(p) && return Float64[], 0.0
    q = vec(p * p')                     # all products p_i * p_j
    sort!(q; rev=true)
    if length(q) > cap
        lost = sum(@view q[(cap+1):end])
        return q[1:cap], lost
    end
    return q, 0.0
end


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

"""
    evolve_closed(n, J, times; kwargs...)

Unitary TEBD on a pure state, returning the SAME NamedTuple schema as
`evolve_vectorized` so that `rows_to_csv`, `profile_to_csv` and `save_run` work
on it unchanged. `gamma` is absent by construction; the returned `gammas` field
is a vector of zeros and `dissipation` is `false`, so the CSV rows slot straight
into the combined table alongside the open runs.

All operator-space columns (S_op, chi_req, linkdim) are the EXACT MPDO values
implied by the pure state, not measurements of an MPDO simulation.

`maxdim` and `cutoff` here apply to the WAVEFUNCTION. The equivalent MPDO bond
dimension is its square, which is why this route reaches n = 20-24 where the
Liouville-space closed run could not.
"""
function evolve_closed(
    n::Int, J, times::Vector{Float64};
    dt::Float64 = 0.05,
    order::Int = 2,
    cutoff::Float64 = 1e-12,
    maxdim::Int = 512,
    initial = :neel,
    tols::Vector{Float64} = [1e-6, 1e-10],
    verbose::Bool = true,
)
    @assert n >= 2
    Jvec = J isa Number ? fill(Float64(J), n-1) : collect(Float64.(J))
    @assert length(Jvec) == n-1

    s = siteinds("Qubit", n)
    excited = initial === :neel  ? collect(1:2:n) :
              initial === :allup ? collect(1:n)   :
              initial === :alldown ? Int[]        : collect(Int.(initial))
    psi = MPS(s, [j in excited ? "1" : "0" for j in 1:n])

    Hmpo = closed_hamiltonian_mpo(n, Jvec, s)
    E0 = real(inner(psi', Hmpo, psi))

    ceiling_mid = 4^min(n÷2, n - n÷2)   # MPDO ceiling, for schema compatibility

    rec_t=Float64[]; rec_S=Float64[]; rec_Smax=Float64[]; rec_Sh=Float64[]
    rec_ld=Int[]; rec_ldmax=Int[]
    rec_chi_mid=[Int[] for _ in tols]; rec_chi_max=[Int[] for _ in tols]
    rec_tr=ComplexF64[]; rec_pur=Float64[]; rec_z=ComplexF64[]; rec_sat=Bool[]
    prof_S=Vector{Vector{Float64}}(); prof_Sh=Vector{Vector{Float64}}()
    prof_ld=Vector{Vector{Int}}(); prof_chi=Vector{Vector{Vector{Int}}}()
    rec_dE=Float64[]

    function snapshot!(tnow)
        nb = n-1; mid = n÷2
        S1=zeros(nb); Sh=zeros(nb); ld=zeros(Int,nb)
        chis=[zeros(Int,nb) for _ in tols]
        for b in 1:nb
            p = bond_schmidt_spectrum(psi, b)          # pure-state spectrum
            q, _ = operator_spectrum_from_pure(p)      # exact MPDO spectrum
            S1[b] = von_neumann_entropy(q)
            Sh[b] = renyi_entropy(q, 0.5)
            ld[b] = length(q)
            for (ti,tol) in enumerate(tols); chis[ti][b] = chi_required(q, tol); end
        end
        nrm = norm(psi)
        E   = real(inner(psi', Hmpo, psi)) / max(nrm^2, 1e-300)
        z   = ITensorMPS.expect(psi, "Z"; sites=max(1,n÷2))

        push!(rec_t,tnow); push!(rec_S,S1[mid]); push!(rec_Smax,maximum(S1))
        push!(rec_Sh,Sh[mid]); push!(rec_ld,ld[mid]); push!(rec_ldmax,maximum(ld))
        for ti in eachindex(tols)
            push!(rec_chi_mid[ti], chis[ti][mid]); push!(rec_chi_max[ti], maximum(chis[ti]))
        end
        push!(rec_tr, ComplexF64(nrm^2))
        push!(rec_pur, 1.0)              # exact by construction, not measured
        push!(rec_z, ComplexF64(z))
        # The honest truncation flag for a pure-state run is energy drift, not
        # a bond-dimension comparison: chi here is the wavefunction's, and its
        # ceiling is 2^min(b,n-b), not the MPDO one.
        push!(rec_dE, abs(E - E0) / max(abs(E0), 1e-12))
        push!(rec_sat, abs(E - E0) / max(abs(E0), 1e-12) > 1e-6)
        push!(prof_S,S1); push!(prof_Sh,Sh); push!(prof_ld,ld); push!(prof_chi,chis)
        return (S=S1[mid], Smax=maximum(S1), chi=maximum(chis[1]), dE=rec_dE[end])
    end

    r0 = snapshot!(0.0)
    if verbose
        @printf("closed n=%d  dt=%.3g order=%d  psi maxdim=%d  (MPDO-equivalent %d)\n",
                n, dt, order, maxdim, maxdim^2)
        println("      t |  S_op(mid)  S_op(max) |  chi_op(1e-6) |   dE/E   | flag")
        println("-"^72)
        @printf("%7.3f | %10.4f %10.4f | %13d | %.2e |  \n", 0.0, r0.S, r0.Smax, r0.chi, r0.dE)
        flush(stdout)
    end

    gate_cache = Dict{Float64,Vector{ITensor}}()
    t_prev = 0.0
    for t_target in times
        span = t_target - t_prev
        nsteps = max(1, round(Int, span/dt))
        ldt = span/nsteps
        gates = get!(gate_cache, round(ldt; digits=12)) do
            closed_step_gates(n, Jvec, round(ldt; digits=12), s; order=order)
        end
        for _ in 1:nsteps
            psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
            normalize!(psi)
        end
        t_prev = t_target
        r = snapshot!(t_target)
        if verbose
            @printf("%7.3f | %10.4f %10.4f | %13d | %.2e | %s\n",
                    t_target, r.S, r.Smax, r.chi, r.dE, r.dE > 1e-6 ? "!" : " ")
            flush(stdout)
        end
    end

    if verbose && any(rec_sat)
        @printf("\nWARNING: energy drifted by >1e-6 relative at %d/%d snapshots.\n",
                count(rec_sat), length(rec_sat))
        println("         Those points are truncation-limited; raise maxdim before quoting them.")
    end

    return (
        n=n, J=Jvec, gammas=zeros(n), dissipation=false,
        order=order, dt=dt, cutoff=cutoff, maxdim=maxdim,
        tols=tols, ceiling_mid=ceiling_mid,
        t=rec_t, S_op_mid=rec_S, S_op_max=rec_Smax, S_half_mid=rec_Sh,
        linkdim_mid=rec_ld, linkdim_max=rec_ldmax,
        chi_req_mid=rec_chi_mid, chi_req_max=rec_chi_max,
        trace=rec_tr, purity=rec_pur, z_mid=rec_z, saturated=rec_sat,
        S_profile=prof_S, S_half_profile=prof_Sh,
        linkdim_profile=prof_ld, chi_profile=prof_chi,
        energy_drift=rec_dE, E0=E0,
    )
end

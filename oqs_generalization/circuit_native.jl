using ITensors, ITensorMPS
using LinearAlgebra, Printf, Statistics, Random

# =============================================================================
# circuit_native.jl  --  STEP 1 of the hardware-aware hardness study
#
# Simulates EXACTLY the circuit in hardware_aware_circuit.py / naive_adc_circuit.py,
# rather than the continuous-time master equation. The object of study is now the
# circuit itself: a k-step, first-order Trotterisation of the XXZ chain with
# ancilla-mediated amplitude damping.
#
# -----------------------------------------------------------------------------
# PARAMETERISATION: GATE ANGLES, NOT (J, dt)
# -----------------------------------------------------------------------------
# Your Qiskit circuit emits, per bond per Trotter step,
#
#     rxx(alpha), ryy(alpha), rzz(2*alpha)     with alpha = 2*J_i*(t/k)
#
# The MPDO and trajectory code used elsewhere in this project built the same
# block as heisenberg_bond_unitary(J*dt, 2*J*dt) -- i.e. HALF the angle, hence
# half the effective coupling. Every time quoted in the earlier study therefore
# corresponds to t/2 of yours.
#
# To make that class of error impossible, everything here is parameterised by
# the RXX ANGLE theta directly:
#
#     theta = 2 * J * t / k        (read off your circuit)
#
# and never by (J, dt). Likewise the damping is parameterised by the per-step
# jump probability
#
#     p = 1 - exp(-gamma * t / k)
#
# which is what your `cry(2*asin(sqrt(p)))` actually implements. If you change
# J, t or k in Qiskit, convert to (theta, p) and pass those.
#
# -----------------------------------------------------------------------------
# TROTTER ORDER
# -----------------------------------------------------------------------------
# Your circuit is FIRST ORDER: odd layer, even layer, damping. The earlier study
# used a symmetric (Strang) splitting. These are different circuits and generate
# entanglement at different rates, so the old numbers do not transfer. First
# order is the right choice here: the point is no longer to approximate the
# Lindbladian, and it costs half the depth.
#
# -----------------------------------------------------------------------------
# WHAT IS AND IS NOT SIMULATED
# -----------------------------------------------------------------------------
# The SWAP shuttling in hardware_aware_circuit.py is exact and does not change
# the channel: after the swap, ancillas[c] carries the old q_{i+1} data while
# system_qubits[i+1] holds |0> and serves as the ancilla for both damping events,
# with a reset between them. We therefore simulate the LOGICAL circuit and omit
# the SWAPs. They matter for hardware depth and for the transpiler, not for the
# entanglement of the simulated state.
# =============================================================================

if !@isdefined(von_neumann_entropy)
    error("circuit_native.jl needs the metric helpers.\n" *
          "  include(\"vectorized_evolution.jl\") before this file.")
end


# =============================================================================
# Initial product states
# =============================================================================
#
# WHY THIS MATTERS MORE THAN IT LOOKS. At theta = pi/2 the bond gate is locally
# equivalent to iSWAP, which on a COMPUTATIONAL BASIS state is just a
# permutation with phases (|01> <-> |10>, phases on |00> and |11>). Neel is a
# basis state, so the circuit shuffles it and the state stays EXACTLY product:
# measured S_op = 0 at n=8 where the MPDO is exact. From |+>^n the same circuit
# reaches 4.000 bits at n=8 -- the maximum at the half cut.
#
# That integer is a warning, though: at theta = pi/2 the gates RXX(pi/2),
# RYY(pi/2), RZZ(pi) are all CLIFFORD and |+>^n is a stabilizer state, so the
# closed circuit is Gottesman-Knill simulable however entangled it looks. Use
# :plus to understand the structure, not to site a hardness claim.
#
#   :neel     |1010...>   basis state, the default
#   :single   |100...>    matches the notebook validation
#   :plus     |+++...>    one Hadamard layer on hardware; Clifford at theta=pi/2
#   :random   Haar-random single-qubit states, seeded; non-stabilizer by
#             construction, and the honest stress test

"Single-qubit kets for a product initial state, as a length-n vector of 2-vectors."
function initial_kets(n::Int, spec; seed::Int=20260916)
    if spec === :neel
        return [isodd(j) ? ComplexF64[0,1] : ComplexF64[1,0] for j in 1:n]
    elseif spec === :single
        return [j == 1 ? ComplexF64[0,1] : ComplexF64[1,0] for j in 1:n]
    elseif spec === :plus
        return [ComplexF64[1,1]/sqrt(2) for _ in 1:n]
    elseif spec === :random
        rng = MersenneTwister(seed)
        return [(v = ComplexF64.(randn(rng,2), randn(rng,2)); v/norm(v)) for _ in 1:n]
    elseif spec isa AbstractVector{<:Integer}
        return [j in spec ? ComplexF64[0,1] : ComplexF64[1,0] for j in 1:n]
    end
    error("initial state must be :neel, :single, :plus, :random or a vector of excited sites")
end

"|rho(0)>> for an arbitrary product state, as an MPS over the doubled sites."
function vectorized_product_mps(lsites::LiouvilleSites, kets::Vector{Vector{ComplexF64}})
    n = lsites.n
    links = [Index(1, "Link,l=$j") for j in 1:(n-1)]
    ts = ITensor[]
    for j in 1:n
        rho_j = kets[j] * kets[j]'          # |s><s|
        T = ITensor(Matrix{ComplexF64}(rho_j), lsites.ket[j], lsites.bra[j])
        j > 1 && (T *= onehot(links[j-1] => 1))
        j < n && (T *= onehot(links[j]   => 1))
        push!(ts, T)
    end
    return MPS(ts)
end

"|psi(0)> for an arbitrary product state on the interleaved system/ancilla chain."
function product_mps_with_ancillas(sites, n::Int, kets::Vector{Vector{ComplexF64}})
    N = 2n
    links = [Index(1, "Link,l=$j") for j in 1:(N-1)]
    ts = ITensor[]
    for m in 1:N
        v = isodd(m) ? kets[(m+1)÷2] : ComplexF64[1,0]   # ancillas start in |0>
        T = ITensor(v, sites[m])
        m > 1 && (T *= onehot(links[m-1] => 1))
        m < N && (T *= onehot(links[m]   => 1))
        push!(ts, T)
    end
    return MPS(ts)
end


# =============================================================================
# Gates, in circuit units
# =============================================================================

"""
    circuit_bond_matrix(theta)

RZZ(2*theta) RYY(theta) RXX(theta) -- exactly the three-rotation block your
circuit emits per bond, with theta the rxx angle.

Entangling power (operator entanglement across the bond, max 2 bits):
    theta   0.05   0.25   0.79   1.25   pi/2   2.36   pi
    bits   0.012   0.55   1.81   1.91   2.00   1.81   0.00
It is MAXIMALLY entangling at theta = pi/2 and trivial at theta = pi.
"""
circuit_bond_matrix(theta::Float64) =
    Matrix{ComplexF64}(heisenberg_bond_unitary(theta, 2*theta))

"Per-step jump probability from your cry angle: p = sin^2(angle/2)."
p_from_cry_angle(angle::Float64) = sin(angle/2)^2

"Your circuit's mapping from physics units to circuit units."
theta_of(J::Real, t::Real, k::Integer) = 2*J*t/k
p_of(gamma::Real, t::Real, k::Integer) = 1 - exp(-gamma*t/k)


# =============================================================================
# ROUTE A -- MPDO (vectorised density matrix)
# =============================================================================

"""
    circuit_mpdo(n, theta, p, k; cutoff, maxdim, excited, tols, verbose)

Applies k Trotter steps of your circuit to |rho>> as an MPS over doubled sites,
recording operator entanglement and required bond dimension AFTER EVERY STEP.

`excited` is a vector of 1-indexed sites starting in |1>. Your notebook uses
`excited=["0"]`, i.e. qubit 0 only -> pass `[1]` here. Note that a single
excitation on an otherwise polarised chain is far less entangling than a Neel
state; `:neel` is available and is the better choice for a hardness study.
"""
function circuit_mpdo(n::Int, theta::Float64, p::Float64, k::Int;
                      cutoff::Float64=1e-12, maxdim::Int=1024,
                      excited=:neel, tols::Vector{Float64}=[1e-6,1e-10],
                      dissipation::Bool=true, snap_every::Int=1,
                      verbose::Bool=true)
    # snap_every: bond_report costs a full gauge sweep plus (n-1) SVDs. For a
    # parameter sweep only the final step is wanted, so pass snap_every=k.
    lsites = liouville_siteinds(n)
    rho = vectorized_product_mps(lsites, initial_kets(n, excited))
    idm = identity_vectorized_mps(lsites)

    U = circuit_bond_matrix(theta)
    odd  = vcat([unitary_channel_gates(U, [j, j+1], lsites) for j in 1:2:n-1]...)
    even = n > 2 ? vcat([unitary_channel_gates(U, [j, j+1], lsites) for j in 2:2:n-1]...) : ITensor[]
    # amplitude_damping_gate takes (gamma, dt); exp(-gamma*dt) = 1-p, so feeding
    # gamma=1, dt=-log(1-p) reproduces the circuit's p exactly.
    damp = dissipation && p > 0 ?
        [amplitude_damping_gate(1.0, -log(1-p), j, lsites) for j in 1:n] : ITensor[]

    rec = NamedTuple[]
    function snap!(step)
        tr = liouville_trace(rho, idm)
        abs(tr) > 1e-14 && (rho[1] = rho[1]/tr)
        r = bond_report(rho, tols)
        push!(rec, (step=step, S_op_mid=r.S_op_mid, S_op_max=r.S_op_max,
                    chi_mid=r.chi_req_mid[1], chi_max=r.chi_req_max[1],
                    linkdim=r.linkdim_max, trace=tr,
                    saturated=r.linkdim_max >= maxdim))
        return rec[end]
    end
    snap!(0)

    if verbose
        @printf("MPDO  n=%d  theta=%.4f  p=%.4f  k=%d  maxdim=%d  (ceiling %d)\n",
                n, theta, p, k, maxdim, state_bond_dim_ceiling(n))
        println(" step | S_op(mid)  S_op(max) | chi(1e-6) | linkdim |  Tr(rho)  | sat")
    end
    for s in 1:k
        rho = apply(odd,  rho; cutoff=cutoff, maxdim=maxdim)
        !isempty(even) && (rho = apply(even, rho; cutoff=cutoff, maxdim=maxdim))
        !isempty(damp) && (rho = apply(damp, rho; cutoff=cutoff, maxdim=maxdim))
        (s % snap_every == 0 || s == k) || continue
        r = snap!(s)
        verbose && @printf("%5d | %9.4f %10.4f | %9d | %7d | %8.6f | %s\n",
            s, r.S_op_mid, r.S_op_max, r.chi_max, r.linkdim, real(r.trace),
            r.saturated ? "!" : " ")
    end
    return rec
end

"""
    mpdo_dense(rho, lsites) -> Matrix

Contracts the vectorised MPDO into a dense 2^n x 2^n density matrix. Only for
small n (n <= 10 is 1024 x 1024); it exists so the fidelity mode can compute a
TRACE distance, which needs eigenvalues and cannot be got from MPS contractions.
"""
function mpdo_dense(rho::MPS, ls::LiouvilleSites)
    n = ls.n
    n > 12 && error("mpdo_dense: n=$n is too large to densify")
    # Densifying deliberately builds a 2n-index tensor (16 at n=8), past
    # ITensor's order-warning threshold of 14. The warning is correct in general
    # and meaningless here, and it printed a stack trace per contraction --
    # hundreds of lines per theta point in task 11. Silence it for this block only.
    # The macro form is the one ITensor's own warning message recommends, so it
    # is certain to exist; it evaluates the block with the warning off and
    # restores the previous threshold afterwards.
    T = ITensors.@disable_warn_order begin
        TT = rho[1]
        for j in 2:length(rho); TT *= rho[j]; end
        TT
    end
    A = Array(T, ls.ket..., ls.bra...)
    d = 2^n
    return reshape(A, d, d)
end

"""
    mpdo_run(n, theta, p, k; ...) -> (rho, lsites)

Applies the k-step circuit and returns the final vectorised state. Shared by the
observable and sweep helpers so a circuit is never run more than once per point.
"""
function mpdo_run(n::Int, theta::Float64, p::Float64, k::Int;
                  cutoff::Float64=1e-12, maxdim::Int=1024,
                  excited=:neel, dissipation::Bool=true, lsites=nothing)
    # `lsites` lets two runs SHARE one set of Index objects. Without it every
    # call makes fresh indices, and any operation relating the two states --
    # inner(rho, rho_ref) for a distance, or one site operator applied to both --
    # has nothing to contract and throws. That is what broke the HS-distance
    # version of trotter_infidelity: the older code only ever compared numbers
    # extracted from self-contained runs, so it never noticed.
    lsites = isnothing(lsites) ? liouville_siteinds(n) : lsites
    rho = vectorized_product_mps(lsites, initial_kets(n, excited))
    U = circuit_bond_matrix(theta)
    odd  = vcat([unitary_channel_gates(U, [j, j+1], lsites) for j in 1:2:n-1]...)
    even = n > 2 ? vcat([unitary_channel_gates(U, [j, j+1], lsites) for j in 2:2:n-1]...) : ITensor[]
    damp = dissipation && p > 0 ?
        [amplitude_damping_gate(1.0, -log(1-p), j, lsites) for j in 1:n] : ITensor[]
    for _ in 1:k
        rho = apply(odd, rho; cutoff=cutoff, maxdim=maxdim)
        !isempty(even) && (rho = apply(even, rho; cutoff=cutoff, maxdim=maxdim))
        !isempty(damp) && (rho = apply(damp, rho; cutoff=cutoff, maxdim=maxdim))
    end
    return rho, lsites
end

"""
    mpdo_expectation_Z_all(n, theta, p, k; ...) -> Vector of <Z_j>, all sites

One circuit run, every site. Calling the single-site version in a loop reruns
the whole circuit n times for data that is already there.
"""
function mpdo_expectation_Z_all(n::Int, theta::Float64, p::Float64, k::Int;
                                cutoff::Float64=1e-12, maxdim::Int=1024,
                                excited=:neel, dissipation::Bool=true, lsites=nothing)
    rho, lsites = mpdo_run(n, theta, p, k; cutoff=cutoff, maxdim=maxdim,
                           excited=excited, dissipation=dissipation, lsites=lsites)
    tr = inner(identity_vectorized_mps(lsites), rho)
    return [real(inner(pauli_z_vectorized_mps(lsites, j), rho) / tr) for j in 1:n]
end

"""
    mpdo_expectation_Z(n, theta, p, k, site; ...) -> <Z_site> after k steps

Your notebook plots <(I - Z_0)/2>, the excited population of qubit 0, which is
(1 - <Z_1>)/2 in 1-indexed Julia terms.
"""
mpdo_expectation_Z(n::Int, theta::Float64, p::Float64, k::Int, site::Int; kw...) =
    mpdo_expectation_Z_all(n, theta, p, k; kw...)[site]


# =============================================================================
# STEP 2 helpers: hardness and faithfulness as functions of the gate angle
# =============================================================================

"""
    circuit_cost_point(n, theta, p, k; ...)

MPDO cost of ONE circuit: operator entanglement and required bond dimension
after the final Trotter step, plus the per-step trace as a truncation check.
"""
function circuit_cost_point(n::Int, theta::Float64, p::Float64, k::Int;
                            cutoff::Float64=1e-12, maxdim::Int=1024,
                            excited=:neel, tols::Vector{Float64}=[1e-6,1e-10])
    rec = circuit_mpdo(n, theta, p, k; cutoff=cutoff, maxdim=maxdim,
                       excited=excited, tols=tols, snap_every=k, verbose=false)
    fin = rec[end]
    return (S_op=fin.S_op_max, chi=fin.chi_max, linkdim=fin.linkdim,
            trace=real(fin.trace), saturated=fin.saturated,
            S_path=[r.S_op_max for r in rec], chi_path=[r.chi_max for r in rec])
end

"""
    trotter_infidelity(n, theta, p, k; k_ref, ...)

How far the k-step circuit is from the CONTINUOUS-TIME master equation it
nominally approximates, measured as max_j |<Z_j>_circuit - <Z_j>_exact|.

The reference is the same physical evolution resolved with `k_ref` steps
instead of k. Holding the physics fixed while changing k means holding
(J*t, gamma*t) fixed, i.e.

    theta_ref = theta * k / k_ref      since theta = 2*J*t/k
    p_ref     = 1 - (1-p)^(k/k_ref)    since (1-p)^k = exp(-gamma*t)

so no separate J or gamma is needed -- the circuit parameters alone determine
the master equation they are a discretisation of.

This is the axis to put hardness against. Both grow with theta, so there is a
real trade-off and it should be shown rather than asserted.
"""
function trotter_infidelity(n::Int, theta::Float64, p::Float64, k::Int;
                            k_ref::Int=1000, cutoff::Float64=1e-12,
                            maxdim::Int=1024, excited=:neel)
    ls = liouville_siteinds(n)          # ONE index set, shared by both runs
    rho , _ = mpdo_run(n, theta, p, k; cutoff=cutoff, maxdim=maxdim,
                       excited=excited, lsites=ls)
    rhor, _ = mpdo_run(n, theta*k/k_ref, 1-(1-p)^(k/k_ref), k_ref;
                       cutoff=cutoff, maxdim=maxdim, excited=excited, lsites=ls)
    idm = identity_vectorized_mps(ls)
    rho  = rho  / inner(idm, rho)          # normalise both traces to 1
    rhor = rhor / inner(idm, rhor)

    z  = [real(inner(pauli_z_vectorized_mps(ls, j), rho))  for j in 1:n]
    zr = [real(inner(pauli_z_vectorized_mps(ls, j), rhor)) for j in 1:n]
    dz = maximum(abs.(z .- zr))

    # HILBERT-SCHMIDT DISTANCE, ||rho - rho_ref||_2, expanded as
    #     <<rho|rho>> + <<ref|ref>> - 2 Re <<rho|ref>>
    # all three of which are plain MPS inner products.
    #
    # WHY THIS AND NOT JUST max|dZ|. The <Z> measure depends on the SCALE of the
    # observable, which depends on the initial state: from Neel every <Z_j>
    # starts at +/-1, from a random product state they start spread over
    # [-1,1] and are typically smaller. The Neel run reported infidelity 0.137
    # at theta=0.95 and the random run 0.062 -- but part of that gap is the
    # observable shrinking, not the circuit becoming more faithful. The HS
    # distance is a property of the states and is comparable across initial
    # states. Normalised by ||rho_ref||_2 so it is dimensionless.
    hs2 = real(inner(rho,rho)) + real(inner(rhor,rhor)) - 2*real(inner(rho,rhor))
    hs  = sqrt(max(hs2, 0.0)) / sqrt(max(real(inner(rhor,rhor)), 1e-300))

    # TRACE DISTANCE, the one with an operational meaning: (1/2)||rho-rho_ref||_1
    # is the maximum probability of telling the two apart by ANY measurement, and
    # it is bounded in [0,1]. The normalised HS distance above is not bounded
    # that way and, because it divides by sqrt(purity), it blows up as the state
    # mixes -- the n=8 run reported ~1.2 while single-site observables agreed to
    # 6%. Needs eigenvalues, hence the densification, hence small n only.
    td = NaN
    if n <= 10
        D = mpdo_dense(rho, ls) - mpdo_dense(rhor, ls)
        td = 0.5 * sum(abs, eigvals(Hermitian((D + D')/2)))
    end

    return dz, hs, td, z, zr
end


# =============================================================================
# ROUTE B -- trajectories (pure state + ancilla + reset), same circuit
# =============================================================================
#
# Layout q1 a1 q2 a2 ... as in trajectory_evolution.jl. The damping block is
# CRy(2 asin(sqrt(p))) from system to ancilla, CX back, reset -- identical to
# your `cry / cx / reset` triple. We use one ancilla per system qubit rather
# than your shuttled n/2, because the SWAPs are exact and change nothing about
# the state's entanglement.

function circuit_trajectory(n::Int, theta::Float64, p::Float64, k::Int;
                            cutoff::Float64=1e-10, maxdim::Int=1024,
                            excited=:neel, seed::Int=1,
                            tols::Vector{Float64}=[1e-6],
                            dissipation::Bool=true)
    rng = MersenneTwister(seed)
    sites = siteinds("Qubit", 2n)
    psi = product_mps_with_ancillas(sites, n, initial_kets(n, excited))

    U = circuit_bond_matrix(theta)
    gA = [_gate2(U, sites[sys_pos(b)], sites[sys_pos(b+1)]) for b in 1:2:n-1]
    gB = n > 2 ? [_gate2(U, sites[sys_pos(b)], sites[sys_pos(b+1)]) for b in 2:2:n-1] : ITensor[]
    gD = dissipation && p > 0 ?
        [_gate2(amplitude_damping_pair(p), sites[sys_pos(j)], sites[anc_pos(j)]) for j in 1:n] :
        ITensor[]

    rec = [merge((step=0,), trajectory_metrics(psi, n, tols),
                 (z=ITensorMPS.expect(psi,"Z")[sys_pos.(1:n)],))]
    for s in 1:k
        psi = apply(gA, psi; cutoff=cutoff, maxdim=maxdim)
        !isempty(gB) && (psi = apply(gB, psi; cutoff=cutoff, maxdim=maxdim))
        if !isempty(gD)
            psi = apply(gD, psi; cutoff=cutoff, maxdim=maxdim)
            for j in 1:n
                _, psi = reset_ancilla!(psi, anc_pos(j); cutoff=cutoff,
                                        maxdim=maxdim, rng=rng)
            end
        end
        normalize!(psi)
        push!(rec, merge((step=s,), trajectory_metrics(psi, n, tols),
                         (z=ITensorMPS.expect(psi,"Z")[sys_pos.(1:n)],)))
    end
    return rec
end

"""
    circuit_trajectory_ensemble(n, theta, p, k, Ntraj; ...)

Averages `circuit_trajectory` and returns per-step cost statistics in the same
form as the earlier study (chi mean with an error bar, the Jensen third moment,
and measured wall time), so the two are directly comparable.
"""
function circuit_trajectory_ensemble(n::Int, theta::Float64, p::Float64, k::Int,
                                     Ntraj::Int; cutoff::Float64=1e-10,
                                     maxdim::Int=1024, excited=:neel,
                                     seed0::Int=1000, dissipation::Bool=true,
                                     site::Int=max(1, n ÷ 2),
                                     verbose::Bool=true)
    # `site` is which qubit <Z> is reported for, and it defaults to the MIDDLE
    # site because that is what the earlier (Neel-initialised) study wanted.
    # That default silently broke the first circuit validation: the driver
    # compared this against the MPDO at site 1 with only q0 excited, and the two
    # sites have completely different curves -- 150 sigma of apparent
    # disagreement from a pure bookkeeping mismatch. The full per-site vector is
    # now returned as `z_all` so the comparison can never be ambiguous again.
    @assert 1 <= site <= n "site must lie in 1:$n, got $site"
    runs = Vector{Vector{NamedTuple}}(undef, Ntraj); wall = zeros(Ntraj)
    Threads.@threads for i in 1:Ntraj
        wall[i] = @elapsed runs[i] =
            circuit_trajectory(n, theta, p, k; cutoff=cutoff, maxdim=maxdim,
                               excited=excited, seed=seed0+i, tols=[1e-6],
                               dissipation=dissipation)
    end
    out = NamedTuple[]
    for s in 1:(k+1)
        S  = [r[s].S_max for r in runs]
        c  = float.([r[s].chi_max[1] for r in runs])
        z  = [real(r[s].z[site]) for r in runs]
        zall = [mean(real(r[s].z[j]) for r in runs) for j in 1:n]
        zallsem = [Ntraj > 1 ? std([real(r[s].z[j]) for r in runs])/sqrt(Ntraj) : 0.0
                   for j in 1:n]
        sd = Ntraj > 1 ? std(c) : 0.0
        push!(out, (step=s-1, S_mean=mean(S), S_p95=quantile(S,0.95),
                    chi_mean=mean(c), chi_std=sd,
                    chi_sem = Ntraj>1 ? sd/sqrt(Ntraj) : 0.0,
                    chi3_mean=mean(c.^3), chi_p95=quantile(c,0.95), chi_max=maximum(c),
                    site=site, z_mid=mean(z),
                    z_sem = Ntraj>1 ? std(z)/sqrt(Ntraj) : 0.0,
                    z_var=var(z), z_all=zall, z_all_sem=zallsem,
                    saturated=maximum(float.([r[s].linkdim for r in runs])) >= maxdim,
                    # The flag above fires when the STORED bond dimension hits the
                    # cap. With a fat spectral tail that can happen while the
                    # REPORTED chi (at tolerance 1e-6) is far below it, in which
                    # case only weight < 1e-6 was discarded and the number stands.
                    # This one asks the question that matters: is the reported
                    # chi itself pressed against the cap?
                    censored=maximum(c) >= 0.9*maxdim))
    end
    if verbose
        @printf("\nTRAJ  n=%d theta=%.4f p=%.4f k=%d Ntraj=%d | wall %.1f s/traj, %.2f core-h\n",
                n, theta, p, k, Ntraj, mean(wall), sum(wall)/3600)
        @printf(" <Z> reported for SITE %d of %d\n", site, n)
        println(" step |  <S>   S_p95 |  <chi>±sem  chi_p95  chi_max | <Z_site>±sem | sat")
        for r in out
            @printf("%5d | %6.3f %6.3f | %7.1f±%-5.1f %7.0f %8.0f | %+.4f±%.4f | %s\n",
                r.step, r.S_mean, r.S_p95, r.chi_mean, r.chi_sem, r.chi_p95,
                r.chi_max, r.z_mid, r.z_sem, r.saturated ? "!" : " ")
        end
    end
    return (series=out, walltime=wall)
end

using ITensors, ITensorMPS
using Random, LinearAlgebra, Printf, Statistics

# =============================================================================
# dephasing_evolution.jl
#
# Dephasing analogue of trajectory_evolution.jl. Same chain, same Trotter
# splitting, same metrics, same cost accounting -- only the dissipator changes:
#
#     amplitude damping   L_j = sqrt(gamma) sigma^-_j
#     dephasing           L_j = sqrt(gamma_phi/2) Z_j
#
# -----------------------------------------------------------------------------
# THE ONE THING THAT IS GENUINELY DIFFERENT, AND WHY THIS FILE IS LONGER
# -----------------------------------------------------------------------------
# For amplitude damping the unravelling is essentially forced: CRy+CX+reset is
# the physical dilation, the jump takes |1> -> |0>, and every sensible Kraus
# decomposition looks like that one. chi_traj is a property of the channel.
#
# For dephasing it is NOT forced, and the two natural unravellings differ by
# orders of magnitude in cost while producing the IDENTICAL density matrix:
#
#   :projective   K0 = diag(1, sqrt(1-lam)), K1 = sqrt(lam)|1><1|,
#                 lam = 1 - exp(-2 gamma_phi dt).
#                 Implemented as CRy(theta) into a per-site ancilla + reset,
#                 theta = 2 asin(sqrt(lam)). NO CX back -- that gate exists in
#                 the amplitude-damping circuit only to move population, and
#                 dephasing moves none. This is continuous weak monitoring of
#                 Z_j, so it actively disentangles.
#
#   :pauli        K0 = sqrt(1-q) I, K1 = sqrt(q) Z, q = (1-exp(-gamma_phi dt))/2.
#                 BOTH Kraus operators are unitary, so every trajectory is a
#                 pure unitary circuit. A single-site Z changes no bond
#                 dimension whatsoever; since Z commutes with ZZ and
#                 anticommutes with XX and YY, the entire channel is a sign
#                 randomisation of the Trotter angles. Expect chi_traj to track
#                 the NOISELESS chain and to saturate maxdim.
#
#   :gaussian     Rz(eta_j), eta ~ N(0, 2 gamma_phi dt). Exactly the same
#                 channel as :pauli, and exactly what the hardware would do
#                 (virtual Z, zero pulses). Included because it is the arm the
#                 experiment actually implements.
#
# A classical competitor picks whichever unravelling is cheapest, so the honest
# trajectory cost is the MINIMUM over unravellings, not whichever one you ran.
# For amplitude damping that distinction was invisible. Here it is the result.
# Both arms are therefore measured and `unravelling_comparison` reports the min.
#
# -----------------------------------------------------------------------------
# MATCHING THE NOISE STRENGTH AGAINST THE AMPLITUDE-DAMPING LINE
# -----------------------------------------------------------------------------
# "Same gamma" is not a comparison, it is a coincidence of symbols. The two
# channels are matched here on T2, i.e. on the decay of the transverse Pauli
# components, which is what actually feeds operator entanglement:
#
#     amplitude damping   D_X = D_Y = exp(-gamma   dt / 2)
#     dephasing           D_X = D_Y = exp(-gamma_phi dt)
#
# so gamma_phi = gamma/2 reproduces the same transverse damping per step. Use
# `noise_diagnostics` to print D_X, D_Z and the contraction coefficient c(N) of
# arXiv:2606.00474 for both channels side by side, so the matching in any given
# run is auditable rather than asserted. Note c >= 1/3 for dephasing at every
# strength (D_Z = 1 always), which is the structural reason to expect it to be
# the harder channel.
# =============================================================================

# --- metric helpers, identical definitions to the MPDO and AD studies ---------
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

RZZ(2J dt) RYY(J dt) RXX(J dt). BIT-IDENTICAL to the amplitude-damping and MPDO
studies. Do not "simplify" the angles -- the comparability of every number in
this file to the AD numbers rests on this being the same unitary.
"""
function heisenberg_bond_matrix(J::Float64, dt::Float64)
    a = J*dt; b = 2*J*dt
    return _rot2(_Z,_Z,b) * _rot2(_Y,_Y,a) * _rot2(_X,_X,a)
end

"""
    phase_damping_pair(lam)

Two-site unitary on (system, ancilla) realising, once the ancilla is reset,
K0 = diag(1, sqrt(1-lam)), K1 = sqrt(lam)|1><1|. This is CRy(theta) with
control = system, target = ancilla, theta = 2 asin(sqrt(lam)), and nothing else.

Contrast `amplitude_damping_pair`, which is CX * CRy: the CX transports the
excitation to the ground state. Dephasing transports nothing, so the dilation is
one gate cheaper -- on hardware, 1 native RZZ instead of 2 CZ + 1 CZ.
"""
function phase_damping_pair(lam::Float64)
    @assert 0 <= lam <= 1 "dephasing probability out of range: $lam"
    θ = 2*asin(sqrt(lam)); c = cos(θ/2); s = sin(θ/2)
    # basis order (q,a) = |00>,|01>,|10>,|11>
    return ComplexF64[1 0 0 0; 0 1 0 0; 0 0 c -s; 0 0 s c]
end

"Wrap a 4x4 matrix as a two-site ITensor gate on (s1, s2)."
function _gate2(M::Matrix{ComplexF64}, s1::Index, s2::Index)
    T = ITensor(ComplexF64, s1', s2', s1, s2)
    for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        T[s1'=>a, s2'=>b, s1=>c, s2=>d] = M[2*(a-1)+b, 2*(c-1)+d]
    end
    return T
end

"Wrap a 2x2 matrix as a one-site ITensor gate on s."
function _gate1(M::Matrix{ComplexF64}, s::Index)
    T = ITensor(ComplexF64, s', s)
    for a in 1:2, b in 1:2; T[s'=>a, s=>b] = M[a,b]; end
    return T
end

"Rz(eta) = exp(-i eta Z / 2). On hardware this is a frame update: free."
_rz(η::Float64) = ComplexF64[exp(-im*η/2) 0; 0 exp(im*η/2)]


# =============================================================================
# Noise diagnostics: make the AD-vs-dephasing matching auditable
# =============================================================================

"""
    noise_diagnostics(gamma_phi, gamma_ad, dt)

Pauli-transfer-matrix damping coefficients and the contraction coefficient
c(N) = (t_X^2+t_Y^2+t_Z^2+D_X^2+D_Y^2+D_Z^2)/3 of arXiv:2606.00474, printed for
both channels at the same dt. Two things to check on every run:

  * D_X should agree between the channels -- that is the matching.
  * c(dephasing) should be >= 1/3 and c(AD) should also be >= 1/3, but AD
    approaches 1/3 only at total damping while dephasing sits at
    (2 exp(-2 gamma_phi dt) + 1)/3 for all strengths. Below 1/3 is the regime
    where an O(1) operator-entanglement plateau is provable; neither channel
    gets there, which is why this study has to be run rather than cited.
"""
function noise_diagnostics(gamma_phi::Float64, gamma_ad::Float64, dt::Float64)
    dφ = exp(-gamma_phi*dt)                       # D_X = D_Y for dephasing
    cφ = (2dφ^2 + 1.0)/3
    p  = 1 - exp(-gamma_ad*dt)
    dA, dZ = sqrt(1-p), 1-p                       # D_X = D_Y, D_Z for AD
    cA = (2dA^2 + dZ^2 + p^2)/3
    @printf("  channel      D_X=D_Y     D_Z        t_Z       c(N)\n")
    @printf("  dephasing    %.6f    %.6f   %.6f   %.6f\n", dφ, 1.0, 0.0, cφ)
    @printf("  amp.damping  %.6f    %.6f   %.6f   %.6f\n", dA, dZ, p, cA)
    @printf("  transverse damping mismatch: %.3e  (0 => T2-matched)\n", abs(dφ-dA))
    return (D_X_deph=dφ, c_deph=cφ, D_X_ad=dA, c_ad=cA)
end


# =============================================================================
# Reset (measure + conditional X), normalised sampling -- the AD study's fix
# =============================================================================

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
    outcome = rand(rng) < p0/tot ? 0 : 1        # normalised
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
# :projective needs one ancilla per system qubit and uses the AD study's layout
#   q1 a1 q2 a2 ... qn an   (stride 2, N = 2n)
# so that chi and wall time are both directly comparable with the AD runs.
#
# :pauli and :gaussian need no ancilla at all -- the Kraus operators are
# unitary. Those arms run on the bare chain (stride 1, N = n). chi remains
# comparable; wall time does NOT, since the MPS is half as long. Compare chi.

stride_of(unravel::Symbol) = unravel === :projective ? 2 : 1

"""
    trajectory_metrics(psi, n, tols, stride)

Entanglement and bond-dimension requirement across every SYSTEM cut, one copy
and one left-to-right sweep. With stride 2 the physical bond 2b sits between
a_b and q_{b+1}; because the metrics are only ever taken with every ancilla
freshly reset to |0>, that spectrum is exactly the system spectrum across
system bond b. With stride 1 the physical and system bonds coincide.
"""
function trajectory_metrics(psi::MPS, n::Int, tols::Vector{Float64}, str::Int)
    nb = n - 1
    S = zeros(nb); chis = [zeros(Int, nb) for _ in tols]

    phi = orthogonalize(psi, str)
    for b in 1:nb
        bb = str*b
        b > 1 && orthogonalize!(phi, bb)
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
    single_trajectory(n, J, gamma_phi, times; unravel, dt, cutoff, maxdim, seed, ...)

Integrates ONE trajectory through the whole time grid. Splitting is identical to
the AD study:

    odd(dt/2), even(dt/2), dephasing(dt), even(dt/2), odd(dt/2)

`unravel` is :projective, :pauli or :gaussian (see the header). All three
reproduce the same Lindbladian; they differ only in trajectory cost, which is
the quantity being measured.
"""
function single_trajectory(n::Int, J::Vector{Float64}, gamma_phi::Vector{Float64},
                           times::Vector{Float64};
                           unravel::Symbol=:projective,
                           dt::Float64=0.05, cutoff::Float64=1e-10,
                           maxdim::Int=512, seed::Int=1,
                           tols::Vector{Float64}=[1e-6],
                           initial=:neel)
    unravel in (:projective, :pauli, :gaussian) ||
        error("unknown unravelling: $unravel")
    rng = MersenneTwister(seed)
    st_ = stride_of(unravel)
    N = st_ * n
    sysp(j::Int) = st_*(j-1) + 1
    ancp(j::Int) = 2j

    sites = siteinds("Qubit", N)
    excited = initial === :neel ? collect(1:2:n) :
              initial === :allup ? collect(1:n) : collect(Int.(initial))
    stlist = fill("0", N)
    for j in excited; stlist[sysp(j)] = "1"; end
    psi = MPS(sites, stlist)

    Hcache = Dict{Float64,Vector{ITensor}}()
    Dcache = Dict{Float64,Vector{ITensor}}()
    Zgates = [ _gate1(_Z, sites[sysp(j)]) for j in 1:n ]

    function hgates(step, parity)
        get!(Hcache, round(step + 0.5parity; digits=12)) do
            g = ITensor[]
            for b in (parity == 0 ? (1:2:n-1) : (2:2:n-1))
                M = heisenberg_bond_matrix(J[b], step)
                push!(g, _gate2(M, sites[sysp(b)], sites[sysp(b+1)]))
            end
            g
        end
    end
    # lam = 1 - exp(-2 gamma_phi dt): the coherence factor is sqrt(1-lam).
    function dgates(step)
        get!(Dcache, round(step; digits=12)) do
            [ _gate2(phase_damping_pair(1 - exp(-2*gamma_phi[j]*step)),
                     sites[sysp(j)], sites[ancp(j)]) for j in 1:n ]
        end
    end

    "One dephasing layer of duration `step`, in whichever unravelling."
    function dephasing_layer!(psi, step)
        if unravel === :projective
            psi = apply(dgates(step), psi; cutoff=cutoff, maxdim=maxdim)
            for j in 1:n
                _, psi = reset_ancilla!(psi, ancp(j); cutoff=cutoff,
                                        maxdim=maxdim, rng=rng)
            end
        elseif unravel === :pauli
            # q = (1 - exp(-gamma_phi dt))/2. Single-site unitaries: these do
            # not change any bond dimension. Applied one at a time so that a
            # trajectory with no jumps costs nothing at all.
            g = ITensor[]
            for j in 1:n
                q = 0.5*(1 - exp(-gamma_phi[j]*step))
                rand(rng) < q && push!(g, Zgates[j])
            end
            isempty(g) || (psi = apply(g, psi; cutoff=cutoff, maxdim=maxdim))
        else # :gaussian -- eta ~ N(0, 2 gamma_phi dt), i.e. the hardware's Rz
            g = [ _gate1(_rz(sqrt(2*gamma_phi[j]*step)*randn(rng)),
                         sites[sysp(j)]) for j in 1:n ]
            psi = apply(g, psi; cutoff=cutoff, maxdim=maxdim)
        end
        return psi
    end

    rec = NamedTuple[]
    push!(rec, merge((t=0.0,), trajectory_metrics(psi, n, tols, st_),
                     (z=ITensorMPS.expect(psi,"Z")[sysp.(1:n)],)))

    tprev = 0.0
    for tt in times
        nsteps = max(1, round(Int, (tt - tprev)/dt)); ldt = (tt - tprev)/nsteps
        for _ in 1:nsteps
            psi = apply(hgates(ldt/2, 0), psi; cutoff=cutoff, maxdim=maxdim)
            psi = apply(hgates(ldt/2, 1), psi; cutoff=cutoff, maxdim=maxdim)
            psi = dephasing_layer!(psi, ldt)
            psi = apply(hgates(ldt/2, 1), psi; cutoff=cutoff, maxdim=maxdim)
            psi = apply(hgates(ldt/2, 0), psi; cutoff=cutoff, maxdim=maxdim)
            normalize!(psi)
        end
        tprev = tt
        push!(rec, merge((t=tt,), trajectory_metrics(psi, n, tols, st_),
                         (z=ITensorMPS.expect(psi,"Z")[sysp.(1:n)],)))
    end
    return rec
end


# =============================================================================
# Ensemble
# =============================================================================

"""
    run_trajectories(n, J, gamma_phi, times, Ntraj; unravel, kwargs...)

Same reporting as the AD study: physics (<Z_j> with standard error) plus the
cost metrics, including chi_std/chi_sem for a real error bar and chi3_mean for
the correct ensemble cost sum_k chi_k^3 = N<chi^3> rather than N<chi>^3.
"""
function run_trajectories(n::Int, J, gamma_phi, times::Vector{Float64}, Ntraj::Int;
                          unravel::Symbol=:projective,
                          dt::Float64=0.05, cutoff::Float64=1e-10,
                          maxdim::Int=512, seed0::Int=1000,
                          tols::Vector{Float64}=[1e-6],
                          initial=:neel, verbose::Bool=true)
    Jv = J isa Number ? fill(Float64(J), n-1) : collect(Float64.(J))
    gv = gamma_phi isa Number ? fill(Float64(gamma_phi), n) : collect(Float64.(gamma_phi))

    all_runs = Vector{Vector{NamedTuple}}(undef, Ntraj)
    walltime = zeros(Ntraj)
    Threads.@threads for k in 1:Ntraj
        walltime[k] = @elapsed all_runs[k] =
            single_trajectory(n, Jv, gv, times; unravel=unravel, dt=dt,
                              cutoff=cutoff, maxdim=maxdim, seed=seed0+k,
                              tols=tols, initial=initial)
        if verbose && Threads.threadid() == 1 && k % max(1, Ntraj÷10) == 0
            @printf("  ... %d/%d trajectories (%s)\n", k, Ntraj, unravel); flush(stdout)
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
        cmf   = float.(cm)
        chisd = Ntraj > 1 ? std(cmf) : 0.0
        push!(out, (t=t, unravel=unravel,
            S_mid_mean=mean(Sm), S_mid_max=maximum(Sm),
            S_max_mean=mean(Smx), S_max_p95=quantile(Smx, 0.95),
            chi_mean=mean(cmf), chi_std=chisd,
            chi_sem = Ntraj > 1 ? chisd/sqrt(Ntraj) : 0.0,
            chi3_mean=mean(cmf.^3),
            chi_p95=quantile(cmf, 0.95), chi_max=maximum(cmf),
            linkdim_mean=mean(ldm), linkdim_max=maximum(ldm),
            z_mid=zbar, z_sem=zsem, z_var=var(zmid),
            saturated = maximum(ldm) >= maxdim))
    end

    if verbose
        @printf("\n  [%s] wall time per trajectory: mean %.1f s, max %.1f s, total %.1f core-h\n",
                unravel, mean(walltime), maximum(walltime), sum(walltime)/3600)
        println("\n      t | <S_traj>  S_p95 | <chi>±sem  chi_p95  chi_max | <Z_mid>±sem | sat")
        println("-"^82)
        for r in out
            @printf("%7.3f | %8.3f %6.3f | %6.0f±%-4.0f %7.0f %8.0f | %+.4f±%.4f | %s\n",
                    r.t, r.S_max_mean, r.S_max_p95, r.chi_mean, r.chi_sem,
                    r.chi_p95, r.chi_max, r.z_mid, r.z_sem, r.saturated ? "!" : " ")
        end
    end
    return (series=out, walltime=walltime)
end

"Trajectories needed for a target standard error on <Z_mid>, at the worst time."
Ntraj_for(res, target_sem::Float64) =
    ceil(Int, maximum(r.z_var for r in res) / target_sem^2)

"Cost of one arm at its chi peak: (Ntraj_needed, chi, N<chi^3>, N<chi>^3)."
function arm_cost(res, target_sem::Float64)
    N = Ntraj_for(res, target_sem)
    i = argmax([r.chi_mean for r in res])
    r = res[i]
    return (t=r.t, Ntraj=N, chi_mean=r.chi_mean, chi_sem=r.chi_sem,
            chi_p95=r.chi_p95, chi_max=r.chi_max, chi3=r.chi3_mean,
            cost_true = N*r.chi3_mean, cost_naive = N*r.chi_mean^3,
            inflation = r.chi_mean > 0 ? r.chi3_mean/r.chi_mean^3 : NaN,
            saturated = any(x.saturated for x in res))
end

"""
    unravelling_comparison(arms, chi_mpdo, target_sem; chi_ad_traj)

The headline table. `arms` is a Dict(:projective => res, :pauli => res, ...).

THREE COMPARISONS, in decreasing order of how much they matter:

 1. MIN OVER UNRAVELLINGS. A competitor is not obliged to use the unravelling
    you find convenient. The binding trajectory cost is the cheapest arm. If
    :pauli is 100x more expensive than :projective, that is not evidence of
    hardness -- it only means nobody would run :pauli.

 2. AGAINST MPDO. Needs a chi_mpdo measured for DEPHASING. The amplitude-damping
    fitted law (S_op ~ 1.39 t, chi ~ 6*2^(1.61 S)) must NOT be reused here: it
    encodes the AD operator-entanglement growth, and the whole hypothesis under
    test is that dephasing grows differently. Pass chi_mpdo <= 0 and this block
    is skipped with a warning rather than silently fabricated.

 3. AGAINST THE AMPLITUDE-DAMPING TRAJECTORY RUN at the same n. This is the
    "is dephasing actually harder" question, and it is a chi-to-chi comparison
    at matched transverse damping.
"""
function unravelling_comparison(arms::Dict, chi_mpdo::Real,
                                target_sem::Float64=0.01; chi_ad_traj::Real=0)
    costs = Dict{Symbol,Any}()
    @printf("\n=== unravelling comparison (target SEM on <Z> = %.3f) ===\n", target_sem)
    @printf("  %-11s %8s %10s %9s %11s %11s %s\n",
            "unravel","t_peak","<chi>","Ntraj","N<chi^3>","N<chi>^3","sat")
    for (k, res) in arms
        c = arm_cost(res, target_sem); costs[k] = c
        @printf("  %-11s %8.3f %10.1f %9d %11.3e %11.3e %s\n",
                k, c.t, c.chi_mean, c.Ntraj, c.cost_true, c.cost_naive,
                c.saturated ? "SATURATED" : "")
    end

    ks = collect(keys(costs))
    best = ks[argmin([costs[k].cost_true for k in ks])]
    bc = costs[best]
    @printf("\n  binding (cheapest) unravelling: %s at N<chi^3> = %.3e\n",
            best, bc.cost_true)
    if length(costs) > 1
        worst = ks[argmax([costs[k].cost_true for k in ks])]
        @printf("  unravelling gap: %s / %s = %.2e\n",
                worst, best, costs[worst].cost_true/bc.cost_true)
        println("  (a large gap means the expensive arm is irrelevant to hardness,")
        println("   not that the problem is hard. Quote the cheap arm.)")
    end
    if bc.saturated
        println("\n  WARNING: the binding arm hit maxdim. Its chi is a LOWER bound, so")
        println("           the reported trajectory cost is an UNDER-estimate and the")
        println("           hardness claim is correspondingly weaker. Rerun larger.")
    end

    ratio = NaN
    if chi_mpdo > 0
        mpdo = float(chi_mpdo)^3
        ratio = mpdo / bc.cost_true
        @printf("\n  MPDO (dephasing): chi = %.3e -> chi^3 = %.3e\n", float(chi_mpdo), mpdo)
        @printf("  trajectory advantage over MPDO: %.3e\n", ratio)
        println(ratio > 1 ?
            "  => trajectories WIN. MPDO chi alone does not establish hardness." :
            "  => MPDO is cheaper; trajectories are not the binding constraint.")
    else
        println("\n  CHI_MPDO not supplied. The MPDO comparison is SKIPPED -- it is not")
        println("  filled in from the amplitude-damping fitted law, because that law")
        println("  describes AD operator entanglement and reusing it here would assume")
        println("  the answer. Run the MPDO study with the dephasing dissipator.")
    end

    ad_ratio = NaN
    if chi_ad_traj > 0
        ad_ratio = bc.chi_mean / float(chi_ad_traj)
        @printf("\n  vs amplitude damping at the same n (T2-matched):\n")
        @printf("    chi_traj  dephasing %.1f   AD %.1f   ratio %.3f\n",
                bc.chi_mean, float(chi_ad_traj), ad_ratio)
        @printf("    cost ratio (chi^3): %.3f\n", ad_ratio^3)
        println(ad_ratio > 1 ?
            "  => dephasing is the HARDER channel for trajectories at this n." :
            "  => dephasing is EASIER here. The hypothesis fails at this n.")
    end

    return (costs=costs, binding=best, mpdo_ratio=ratio, ad_chi_ratio=ad_ratio)
end

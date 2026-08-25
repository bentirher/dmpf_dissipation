using ITensors, ITensorMPS
using LinearAlgebra
include("bond_dimension_tracking.jl")

# =============================================================================
# Vectorized initial state |rho(0)>> as an MPS over the doubled (ket,bra)
# Liouville sites
# =============================================================================
#
# We restrict to product initial states where each qubit starts either in
# the ground state |0> or the excited state |1>, i.e. rho(0) = ⊗_j |s_j><s_j|
# with s_j in {0,1}. `initially_excited` lists the (1-indexed, matching the
# rest of this codebase's site numbering) qubit positions that start in |1>;
# every other qubit starts in |0>. Example: initially_excited = [1, 4] means
# qubits 1 and 4 start excited, all others start in the ground state.
#
# Vectorized, rho_j = |s_j><s_j| has a single nonzero entry equal to 1 at the
# (bra=s_j, ket=s_j) position (see liouville_space.jl's bra-slow/ket-fast
# convention), i.e. |rho_j>> = |s_j>_ket ⊗ |s_j>_bra -- the SAME label s_j
# repeated on both the ket and bra leg, with no conjugation needed since
# computational-basis amplitudes are already real. This makes |rho(0)>> a
# simple product-state MPS over all 2n doubled-site indices, built the same
# way identity_liouville_mpo builds the identity MPO: one ITensor per
# physical site carrying both the ket and bra index, stitched together with
# trivial bond-dimension-1 links.

function vectorized_initial_state_mps(lsites::LiouvilleSites, initially_excited::Vector{Int})
    n = lsites.n
    @assert all(1 .<= initially_excited .<= n) "initially_excited entries must be valid 1-indexed site numbers in 1:$n."

    tensors = ITensor[]
    links = [Index(1, "Link,l=$j") for j in 1:(n-1)]

    for j in 1:n
        s = (j in initially_excited) ? 2 : 1   # ITensor state index: 1 <-> |0>, 2 <-> |1>
        ket_s = lsites.ket[j]
        bra_s = lsites.bra[j]
        t = state(ket_s, s) * state(bra_s, s)
        if j > 1
            t *= onehot(links[j-1] => 1)
        end
        if j < n
            t *= onehot(links[j] => 1)
        end
        push!(tensors, t)
    end

    return MPS(tensors)
end

# Convenience overload accepting string labels, matching the user-facing
# convention requested (e.g. initially_excited = ["0", "3"] for 0-indexed
# qubit labels "0" and "3"). Internally converts to the 1-indexed
# Vector{Int} convention used everywhere else in this codebase.
function vectorized_initial_state_mps(lsites::LiouvilleSites, initially_excited::Vector{String})
    one_indexed = [parse(Int, s) + 1 for s in initially_excited]
    return vectorized_initial_state_mps(lsites, one_indexed)
end

# =============================================================================
# Distance of an MPO to the identity, in Frobenius (Hilbert-Schmidt) norm
# =============================================================================
#
# ||F - Id||_F = sqrt(<<F-Id, F-Id>>) = sqrt(Tr[(F-Id)^dagger (F-Id)]).
#
# This is the same norm used throughout the project notes/draft (e.g. the
# Trotter-error norm ||rho-mu||_F, and the analytical/numerical results for
# ||F_ii - 1||_F derived for amplitude damping), so this diagnostic is
# directly comparable to those earlier results. We build D = F - Id as one
# MPO via ITensorMPS's `-` (the same `add`/truncation machinery as for MPS),
# then take its norm -- numerically more robust than separately computing
# inner(F,F), inner(F,Id), inner(Id,F), inner(Id,Id) and subtracting (which
# risks cancellation between large near-equal terms when F is close to Id).
#
# NOTE: subtraction/addition of MPOs can itself increase bond dimension (the
# bond dimension of a sum is, before truncation, the SUM of the two operands'
# bond dimensions), so we expose cutoff/maxdim here too and truncate the
# difference the same way every other MPO in this codebase is truncated.

function distance_to_identity(F::MPO, lsites::LiouvilleSites; cutoff::Float64=1e-12, maxdim::Int=400)
    Id = identity_liouville_mpo(lsites)
    D = +(F, -1 * Id; cutoff=cutoff, maxdim=maxdim)
    return sqrt(real(inner(D, D)))
end

# =============================================================================
# Expectation value <<rho(0)| F_ij |rho(0)>>  (open-system analogue of Eq. 14)
# =============================================================================
#
# Direct analogue of `inner(psi', F, psi)` used in the closed-system code
# (optimization_problem.jl) for ⟨ψ(0)|F_ij|ψ(0)⟩ -- here psi is the
# vectorized-state MPS built above, and the result is generally COMPLEX
# (unlike the closed pure-state case, where |⟨ψ(0)|Fij|ψ(0)⟩|^2 is what
# enters M_ij). Per Eq. 52 of main.pdf, M_ij(t) = <<rho(0)|F_ij|rho(0)>>
# directly, with no extra modulus-squared -- we therefore return the raw
# (complex) inner product and let the caller decide whether to report its
# real part, modulus, etc.

function expect_F(F::MPO, rho0::MPS)
    return inner(rho0', F, rho0)
end

# =============================================================================
# Full F(t) report: expectation-value matrix + distance-to-identity,
# mirroring Eqs. 14-16 of main.pdf
# =============================================================================
#
# Builds every F_ij for the given list of k's (via build_open_F), then for
# each one computes:
#   (a) <<rho(0)|F_ij|rho(0)>>             -- the open-system M_ij entry
#   (b) ||F_ij - Id||_F                    -- how far this MPO is from the
#                                              identity in Frobenius norm
#
# and assembles (a) into a matrix laid out exactly like Eq. 14 (rows/cols
# indexed by the same order as `ks`), so the diagonal vs. off-diagonal
# structure can be inspected directly. As noted in the project analysis,
# expect the DIAGONAL entries (i==j) to be visibly less than 1 once
# dissipation != 0 -- in sharp contrast to the closed-system case, where
# F_ii is exactly the identity and the diagonal is exactly 1 by construction
# (S^dagger S = Id for unitary S, which fails once S is a genuine CPTP map).

function F_report(
    n, J, gammas, t, ks, initially_excited, lsites::LiouvilleSites, cutoff, maxdim;
    order::Int=1, dissipation::Bool=true,
    dist_cutoff::Float64=1e-12, dist_maxdim::Int=400
)
    r = length(ks)
    Fs = build_open_F(n, J, gammas, t, ks, lsites, cutoff, maxdim; order=order, dissipation=dissipation)
    rho0 = vectorized_initial_state_mps(lsites, initially_excited)

    M = zeros(ComplexF64, r, r)
    distances = zeros(Float64, r, r)

    idx = 1
    for i in 1:r
        for j in 1:r
            F = Fs[idx]
            M[i, j] = expect_F(F, rho0)
            distances[i, j] = distance_to_identity(F, lsites; cutoff=dist_cutoff, maxdim=dist_maxdim)
            idx += 1
        end
    end

    return (M=M, distances=distances, Fs=Fs, rho0=rho0)
end

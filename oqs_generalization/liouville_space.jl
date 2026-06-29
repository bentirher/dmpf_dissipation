using ITensors, ITensorMPS
using LinearAlgebra

# =============================================================================
# Liouville-space site setup
# =============================================================================
#
# Convention: every physical qubit site `j` is represented, in the vectorized
# (Liouville) picture, by a *pair* of ordinary "Qubit" indices:
#
#   ket_sites[j]  -- the "row" leg, i.e. the index that ρ would carry on the
#                     ket side (transforms with K  in K ρ K^†)
#   bra_sites[j]  -- the "column" leg, i.e. the index that ρ would carry on
#                     the bra side (transforms with K* in K ρ K^†)
#
# This matches the standard vectorization convention used throughout the
# project (Eq. 43 of main.pdf, vec(AXB) = (B^T ⊗ A) vec(X)):
#
#   |ρ⟩⟩ = vec(ρ),     K ρ K^†  -->  (K* ⊗ K) |ρ⟩⟩
#
# So the "ket" leg of our doubled site carries the ordinary operator K, and
# the "bra" leg carries K* (note: NOT K^†, just the elementwise conjugate).
#
# We DO NOT use a single dimension-4 "super-site". Instead we keep ket/bra as
# two separate dimension-2 ITensor indices living at the same physical
# location, combined into one MPO tensor with 4 site indices
# (ket-in, ket-out, bra-in, bra-out). This makes it trivial to reuse all the
# existing ITensors/ITensorMPS MPO machinery (apply, contract, op_dag, SVD
# truncation, etc.) exactly as in the closed-system code, just running it
# *twice* (once for ket, once for bra) for unitary pieces, and coupling the
# two legs together only inside genuinely dissipative (non-unitary) gates.
#
# Bookkeeping: we store the ket and bra site indices in two separate vectors,
# `ket` and `bra`, both of type Vector{Index}, each carrying the tags
# "Qubit" (so the existing built-in op overloads for SiteType"Qubit" still
# dispatch correctly), plus "Ket"/"Bra" and a site-number tag to keep every
# index globally distinct.

struct LiouvilleSites
    n::Int
    ket::Vector{Index}
    bra::Vector{Index}
end

function liouville_siteinds(n::Int)
    # Built directly as fresh Index objects with exactly the tags needed
    # ("Qubit", "Site", and "Ket"/"Bra" plus a site-number tag), rather than
    # layering addtags on top of siteinds("Qubit", n) output. ITensor indices
    # support at most 4 tags; siteinds("Qubit", n) already attaches "Qubit",
    # "Site", and a "n=j" tag (3 tags), so adding a 4th ("Ket"/"Bra") would be
    # right at the documented ceiling and is avoided here for robustness --
    # we just build the index with the 3 tags we actually need.
    ket = [Index(2, "Qubit,Ket,n=$j") for j in 1:n]
    bra = [Index(2, "Qubit,Bra,n=$j") for j in 1:n]
    return LiouvilleSites(n, ket, bra)
end

# =============================================================================
# Pauli / ladder operators (dense 2x2), used to build Kraus operators
# =============================================================================

const ID2 = ComplexF64[1 0; 0 1]
const SIGMA_X = ComplexF64[0 1; 1 0]
const SIGMA_Y = ComplexF64[0 -im; im 0]
const SIGMA_Z = ComplexF64[1 0; 0 -1]

# Pauli lowering operator, sigma_- = |0><1| (with |0>=ground/computational "0").
# WARNING (see project notes): this is the operator that maps |1> -> |0>.
# Do NOT confuse with the raising operator |1><0|.
const SIGMA_MINUS = ComplexF64[0 1; 0 0]

# =============================================================================
# Liouville-space ("doubled") gate construction
# =============================================================================
#
# For a unitary gate U acting on some subset of *physical* sites, the
# corresponding Liouville-space gate acting on rho is
#
#   rho -> U rho U^†
#
# which, vectorized, is the operator  U* ⊗ U  acting on |rho⟩⟩  (ket legs
# transform with U, bra legs transform with U*, consistent with the
# liouville_siteinds convention above).
#
# We build this as a pair of ITensor gates: apply `U` itself on the ket legs
# and `conj(U)` on the bra legs of the same physical sites. Because U and
# conj(U) act on *different* ITensor indices (ket_sites vs bra_sites), this
# is just two ordinary ITensor "op" gates and can be applied independently
# with the standard `apply` function -- no need to build a single bigger
# tensor by hand.

# Build the ITensor gate list implementing rho -> U rho U^† for a given dense
# unitary matrix `Umat` acting on physical sites `js` (1 or 2 sites).
#
# Index convention (must match exactly how Umat was built):
#   - 1-site: Umat is an ordinary 2x2 matrix, basis order |0>,|1>.
#   - 2-site: Umat is built (see heisenberg_bond_unitary in
#     open_product_formula_generation.jl) as an ordinary 4x4 matrix in basis
#     order |00>,|01>,|10>,|11> with site `js[1]` as the *more significant*
#     bit, i.e. exactly the same convention as the existing RXX/RYY/RZZ
#     overloads in product_formula_generation.jl. To turn a 4x4 matrix `M`
#     written in that convention into a correct two-site ITensor gate, the
#     documented ITensors.jl recipe (see the "CX" example in the ITensors.jl
#     manual) is:
#         itensor(M, s2', s1', s2, s1)
#     i.e. the *second* site index comes first, for both the primed and
#     unprimed groups. We use exactly this recipe below -- do not replace it
#     with a generic op(M, s1, s2) call, which uses a different index
#     ordering convention and would silently produce the wrong gate.
function unitary_channel_gates(Umat::AbstractMatrix, js::Vector{Int}, lsites::LiouvilleSites)
    Uket = _matrix_to_gate(Matrix{ComplexF64}(Umat), [lsites.ket[j] for j in js])
    Ubra = _matrix_to_gate(Matrix{ComplexF64}(conj(Umat)), [lsites.bra[j] for j in js])
    return [Uket, Ubra]
end

# Helper: turn a dense 2^m x 2^m matrix (basis order with site 1 most
# significant) into a correctly-indexed m-site ITensor gate, for m in {1,2}.
function _matrix_to_gate(M::Matrix{ComplexF64}, inds::Vector{<:Index})
    if length(inds) == 1
        return itensor(M, inds[1]', inds[1])
    elseif length(inds) == 2
        s1, s2 = inds[1], inds[2]
        return itensor(M, s2', s1', s2, s1)
    else
        error("_matrix_to_gate only supports 1- or 2-site gates, got $(length(inds)) sites.")
    end
end

# =============================================================================
# Reserved for the upcoming Kraus-operator-based comparison
# =============================================================================
#
# NOT currently used by the direct/brute-force pipeline in
# open_product_formula_generation.jl, which instead builds dissipator gates
# by exponentiating the dense vectorized-Lindbladian generator directly (see
# single_qubit_dissipator_channel there). This function is kept here,
# unused for now, because it implements the alternative Kraus-based route to
# the SAME forward channel -- useful once we're ready to benchmark "exponentiate
# the dense generator" against "sum K_mu ⊗ conj(K_mu) over Kraus operators"
# against each other for bond-dimension/runtime, as planned next.
#
# Build the ITensor gate list implementing a single-site Kraus channel
# rho -> sum_mu K_mu rho K_mu^† on physical site j, given a list of Kraus
# matrices {K_mu} (2x2 each), via the standard isometry construction:
#
#   Stack the Kraus operators into one tall matrix and apply it as a single
#   rank-4 tensor that couples ket and bra legs (this is *not* a tensor
#   product of two independent gates, since a generic CP map cannot be
#   written as U ⊗ V -- it genuinely entangles ket and bra legs whenever it
#   is not unitary).
#
# Concretely we build the dense superoperator matrix
#   S = sum_mu  conj(K_mu) ⊗ K_mu     (4x4 for a single qubit)
# and turn it into a single ITensor with one ket-in/ket-out index pair and
# one bra-in/bra-out index pair, suitable for `apply`.
function kraus_channel_gate(Kraus_ops::Vector{<:AbstractMatrix}, j::Int, lsites::LiouvilleSites)
    d = size(Kraus_ops[1], 1)
    @assert d == 2 "kraus_channel_gate currently assumes single-qubit (d=2) Kraus operators."

    S = zeros(ComplexF64, d * d, d * d)
    for K in Kraus_ops
        S .+= kron(conj(K), K)
    end

    ket_s = lsites.ket[j]
    bra_s = lsites.bra[j]

    # S acts on the combined (bra,ket) "super-index" of dimension d*d in the
    # column-stacking convention vec(K rho K^†) = (K* ⊗ K) vec(rho), i.e. the
    # *first* (slower) tensor factor is bra, the *second* (faster) is ket.
    # We now reshape S into a 4-index ITensor with indices
    # (ket', bra', ket, bra) = (ket_s', bra_s', ket_s, bra_s)
    # acting as: out = S * in, i.e. ITensor with primed (output) and
    # unprimed (input) indices.
    Stensor = reshape(S, d, d, d, d)  # (ket_out, bra_out, ket_in, bra_in)

    g = ITensor(Stensor, ket_s', bra_s', ket_s, bra_s)
    return g
end

# =============================================================================
# Identity superoperator MPO
# =============================================================================

function identity_liouville_mpo(lsites::LiouvilleSites)
    n = lsites.n
    tensors = ITensor[]
    links = [Index(1, "Link,l=$j") for j in 1:(n-1)]

    for j in 1:n
        ket_s = lsites.ket[j]
        bra_s = lsites.bra[j]
        t = op("Id", ket_s) * op("Id", bra_s)
        if j > 1
            t *= onehot(links[j-1] => 1)
        end
        if j < n
            t *= onehot(links[j] => 1)
        end
        push!(tensors, t)
    end

    return MPO(tensors)
end

using ITensors, ITensorMPS
using LinearAlgebra
include("bond_dimension_tracking.jl")   # pulls in everything else via its include chain

# =============================================================================
# Example: open-system Heisenberg chain with single-qubit amplitude damping
# =============================================================================
#
# METHODOLOGY NOTE: the dissipator gates built here (via
# open_product_formula_generation.jl) use the DIRECT/brute-force route --
# dense exponentiation of the local vectorized-Lindbladian generator, no
# Kraus operators. This is the baseline to characterize before comparing
# against a Kraus-operator-based construction (see kraus_channel_gate in
# liouville_space.jl, not yet wired into the product formula).

n = 6                      # number of system qubits
J = fill(1.0, n - 1)       # uniform nearest-neighbor Heisenberg coupling
gammas = fill(0.05, n)     # uniform amplitude-damping rate on every qubit
t = 2.0                    # total evolution time
ks = [4, 6]                # Trotter step counts to compare (k_i, k_j)

cutoff = 1e-10
maxdim = 200

lsites = liouville_siteinds(n)

# --- Build the full set of F_ij (Eq. 50) for all pairs in ks, via middle-out ---
Fs = build_open_F(n, J, gammas, t, ks, lsites, cutoff, maxdim; order=2)
println("Built ", length(Fs), " F_ij components for ks = ", ks)
for (idx, F) in enumerate(Fs)
    println("  F[$idx]: middle bond dim = ", middle_bond_dim(F))
end

# --- Build F_ex,j (Eq. 51) using a fine reference k0 >> ks ---
k0 = 40
Fex = build_open_F_ex(n, J, gammas, t, ks, k0, lsites, cutoff, maxdim; order=2, order_ref=2)
println("\nBuilt ", length(Fex), " F_ex,j components against reference k0 = ", k0)
for (idx, F) in enumerate(Fex)
    println("  F_ex[$idx]: middle bond dim = ", middle_bond_dim(F))
end

# --- Track Schmidt-rank growth of F_ii layer by layer (the diagnostic the
#     project notes recommend running first, since the closed-system
#     near-identity argument is not expected to survive once gamma > 0) ---
k_track = 8
result = track_Fii_bond_dimension(
    n, J, gammas, t, k_track, lsites, cutoff, maxdim;
    order=2, track_exact_rank=true, rank_cutoff=1e-10
)

println("\nLayer-by-layer F_ii growth (n=$n, gamma=$(gammas[1]), t=$t, k=$k_track):")
println("layer | stored bond dim | exact operator-Schmidt rank")
for i in eachindex(result.layer)
    println(result.layer[i], "     | ", result.bond_dim[i], "       | ", result.schmidt_rank[i])
end

# Full bond-dimension profile of the final F_ii (all bonds, not just middle)
println("\nFull bond profile of final F_ii: ", full_bond_dimension_profile(result.F_final))

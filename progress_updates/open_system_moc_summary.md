# Open-System Dynamic Multi-Product Formulas: Implementation Summary

**Status as of end of conversation (2025-07)**
**Covers:** all `.jl` files currently in the project plus the `main.pdf` draft.

---

## 1. Overview and Goal

The project extends the closed-system Dynamic Multi-Product Formula (DMPF) algorithm of Robertson et al. (PRX Quantum 6, 020360, 2025) — implemented in the companion closed-system Julia files (`middle_out_contraction.jl`, `optimization_problem.jl`, etc.) — to **open quantum systems** governed by the Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) master equation:

$$\frac{d\rho}{dt} = -i[H,\rho] + \sum_i \gamma_i \left( L_i \rho L_i^\dagger - \frac{1}{2}\{L_i^\dagger L_i, \rho\} \right)$$

The specific physical system used throughout as a test case is:
- **Hamiltonian:** Heisenberg spin chain $H = -\sum_i J_i(X_iX_{i+1} + Y_iY_{i+1}) + 2J_i Z_iZ_{i+1}$, with $J_i$ drawn uniformly from $[1/4, 3/4]$ (same as closed-system paper).
- **Dissipators:** single-qubit amplitude-damping channels with $L_i = \sigma_- = |0\rangle\langle 1|$ (the Pauli-lowering operator, mapping the excited state $|1\rangle$ to the ground state $|0\rangle$) and rate $\gamma_i > 0$ on every site.
- **Initial state:** computational product state $|\psi(0)\rangle = |1\cdots 0 \cdots 1\cdots\rangle$, specified by a list of excited-qubit labels.

The goal is to compute the DMPF coefficients $\{c_j(t)\}$ that minimize $\|\rho(t) - \mu(t)\|_F^2$ where $\mu(t) = \sum_j c_j(t) \rho_{k_j}(t)$, and to quantify how the open-system nature changes the tractability of the underlying MPO objects compared to the closed system.

---

## 2. Mathematical Framework

### 2.1 Vectorization / Liouville Space

The central tool is the **column-stacking vectorization** (Eq. 43 of main.pdf, following Gyamfi, Eur. J. Phys. 41, 063002, 2020):

$$\text{vec}(AXB) = (B^T \otimes A)\,\text{vec}(X)$$

Under this mapping:
- The density matrix $\rho$ (size $2^n \times 2^n$) becomes a supervector $|\rho\rangle\rangle$ (size $4^n$).
- The Lindbladian superoperator becomes the vectorized Lindbladian matrix (Eq. 43):

$$\mathcal{L} = -i\mathbb{1}\otimes H + iH^T\otimes\mathbb{1} + \sum_i \gamma_i\left(\bar L_i \otimes L_i - \frac{1}{2}\mathbb{1}\otimes L_i^\dagger L_i - \frac{1}{2}(L_i^\dagger L_i)^T \otimes \mathbb{1}\right)$$

- The exact solution is $|\rho(t)\rangle\rangle = e^{t\mathcal{L}}|\rho(0)\rangle\rangle$.
- The product-formula approximation is $|\rho_k(t)\rangle\rangle = S(t/k)^k |\rho(0)\rangle\rangle$, where $S$ is the vectorized channel.

### 2.2 Why the Unitary Part Does Not Need Explicit Vectorization

For a unitary gate $U$, the channel $\rho \mapsto U\rho U^\dagger$ vectorizes as:

$$\text{vec}(U\rho U^\dagger) = \underbrace{(U^\dagger)^T}_{= \bar U} \otimes U \cdot \text{vec}(\rho) = (\bar U \otimes U)\,|\rho\rangle\rangle$$

This follows directly from applying $\text{vec}(AXB) = (B^T\otimes A)\text{vec}(X)$ with $A=U$, $B=U^\dagger$. Crucially, the result **factorizes** as a tensor product of two independent operators — $U$ acting on the "ket" leg, $\bar U$ acting on the "bra" leg. This means the vectorized unitary gate can be implemented as two separate, smaller ITensor gates without ever building the full $4\times4$ (or $16\times16$ for two-qubit bonds) Kronecker-product matrix explicitly. This is both mathematically exact and computationally cheaper.

This factorization is also confirmed by exponentiating the Liouvillian's unitary part: since $\mathbb{1}\otimes H$ and $H^T\otimes\mathbb{1}$ commute (they act on disjoint tensor factors), the exponential splits exactly as $e^{-it(\mathbb{1}\otimes H - H^T\otimes\mathbb{1})} = e^{itH^T}\otimes e^{-itH} = \bar U \otimes U$ (using $e^{itH^T} = e^{it\bar H} = \overline{e^{-itH}} = \bar U$ for Hermitian $H$).

**The dissipator cannot use this trick.** The dissipative part is a *sum* of terms — $\bar L\otimes L$, $\mathbb{1}\otimes L^\dagger L$, $(L^\dagger L)^T\otimes\mathbb{1}$ — whose sum does not factorize as $A_\text{bra}\otimes A_\text{ket}$ for any $A_\text{bra}$, $A_\text{ket}$. This is why the dissipator gate genuinely couples the ket and bra legs of the MPO tensor.

### 2.3 Gram Matrix and Overlap Vector in Liouville Space

The Frobenius inner product $\text{Tr}(A^\dagger B)$ becomes the ordinary Hilbert-space inner product $\langle\langle A|B\rangle\rangle$ under vectorization. This gives (Eqs. 46–53 of main.pdf):

$$M_{ij}(t) = \text{Tr}(\rho_{k_i}(t)\rho_{k_j}(t)) = \langle\langle\rho(0)|\underbrace{S^\dagger(t/k_i)^{k_i} S(t/k_j)^{k_j}}_{\mathbb{F}_{ij}}|\rho(0)\rangle\rangle$$

$$L_j(t) = \text{Tr}(\rho(t)\rho_{k_j}(t)) = \langle\langle\rho(0)|\underbrace{(e^{t\mathcal{L}})^\dagger S(t/k_j)^{k_j}}_{\mathbb{F}_{\text{ex},j}}|\rho(0)\rangle\rangle$$

**Key semantic difference from the closed system:** In the closed system, $M_{ij} = |\langle\psi(0)|F_{ij}|\psi(0)\rangle|^2$ uses a modulus-squared (because $M_{ij}$ there reduces to a squared amplitude). In the open system, $M_{ij} = \text{Re}\langle\langle\rho(0)|\mathbb{F}_{ij}|\rho(0)\rangle\rangle$ uses only the real part (no squared modulus), because $M_{ij} = \text{Tr}(\rho_{k_i}\rho_{k_j})$ is manifestly real as a trace of a product of two Hermitian operators. Any imaginary residual in the inner product is floating-point noise and is discarded.

### 2.4 Optimization Problem

The optimization problem (Eq. 40 of main.pdf) is **structurally identical** to the closed-system case:

$$\min_{\{c_j\}} \sum_{i,j} c_i M_{ij} c_j - 2\sum_j c_j L_j \quad \text{s.t.} \quad \sum_j c_j = 1$$

The constraint $\sum_j c_j = 1$ ensures trace-preservation of $\mu(t)$ (since each $\rho_{k_j}$ is trace-one). Note this does **not** guarantee positivity; $\mu(t)$ may not be a valid density matrix.

The solution via Lagrange multipliers yields the same $(r+1)\times(r+1)$ block linear system (Eq. 24):

$$\begin{pmatrix} M & -\mathbf{1} \\ \mathbf{1}^T & 0 \end{pmatrix} \begin{pmatrix} \vec c \\ \lambda \end{pmatrix} = \begin{pmatrix} \vec L \\ 1 \end{pmatrix}$$

The difference from closed system: $M$ and $L$ now use the Liouville-space formulas above; `dynamic_mpf_coefficients` itself is reused unchanged.

### 2.5 Error Expressions

The cost function expands as (Eq. 39):

$$\|\rho(t) - \mu(t)\|_F^2 = \text{Tr}(\rho^2(t)) + \vec{c}^T M \vec{c} - 2\vec{L}^T\vec{c}$$

where $\text{Tr}(\rho^2(t)) \leq 1$ for mixed states (equals 1 only in the closed/pure-state case). This term is **independent of $c_j$** and drops out of the optimization, but it must be included in error estimation.

- **DMPF error:** $E_\text{MPF} = \text{Tr}(\rho^2) + \vec{c}^T M_\text{ref} \vec{c} - 2\vec{L}_\text{ref}^T\vec{c}$
- **Single-Trotter errors:** $E_{k_j} = \text{Tr}(\rho^2) + \text{Tr}(\rho_{k_j}^2) - 2L_j = \text{Tr}(\rho^2) + M[j,j] - 2L_j$

Note: $\text{Tr}(\rho_{k_j}^2) = M[j,j]$ (the diagonal of the Gram matrix) comes for free from the computation already done. The purity $\text{Tr}(\rho^2(t))$ is computed as $\|S(t/k_\text{ref})^{k_\text{ref}}|\rho(0)\rangle\rangle\|^2$ by applying the $k_\text{ref}$-step forward channel MPO to $|\rho_0\rangle\rangle$ and norming the result — $O(\chi^2)$ cost, much cheaper than building another MPO contraction.

### 2.6 Adjoint vs. Inverse — A Critical Distinction

In the closed system, $S(t/k)$ is a unitary operator, so $S^\dagger = S^{-1}$ and the adjoint/inverse distinction is trivial. For a CPTP map (open system), **the adjoint $S^\dagger$ is not itself a physical channel**: the adjoint of a quantum channel cannot be implemented by any completely positive map, even probabilistically (no-go theorem). This has two consequences:

1. The tractability argument of the closed-system paper (that $\mathbb{F}_{ij} = S^\dagger S$ stays close to identity, giving small bond dimension) **does not transfer** to the open system. The failure of $S^\dagger S \approx \mathbb{1}$ is both expected and observed numerically (see Section 5).

2. The adjoint $S^\dagger$ must be built differently from the forward channel $S$. Specifically, for a product $S = A_1 A_2 \cdots A_m$:
$$S^\dagger = A_m^\dagger \cdots A_2^\dagger A_1^\dagger$$
For unitary gates, $A_k^\dagger$ is obtained by negating $dt$ (or equivalently by `swapprime(dag(.))` on the ITensor). For dissipative gates, $A_k^\dagger$ is the conjugate-transpose of the dense superoperator matrix — **not** a re-run with negated $dt$, since the dissipator channel is not unitary.

---

## 3. Code Architecture and File Descriptions

### 3.1 Include Chain

```
liouville_space.jl
  └── open_product_formula_generation.jl
        └── open_middle_out_contraction.jl
              └── bond_dimension_tracking.jl
                    └── F_diagnostics.jl
                          └── open_optimization_problem.jl
```

### 3.2 `liouville_space.jl`

**Purpose:** Site setup and basic gate primitives for Liouville space.

**Key design decision — dual-index site representation:** Each physical qubit $j$ is represented by two ITensor `Index` objects:
- `ket[j]` tagged `"Qubit,Ket,n=j"` — the "row" index, transforms with $U$ (or $K$).
- `bra[j]` tagged `"Qubit,Bra,n=$j"` — the "column" index, transforms with $\bar U$ (or $\bar K$).

Both carry the `"Qubit"` tag so that ITensor's built-in `op("Id", ...)` and similar dispatches still work. We do **not** use a single dimension-4 "super-site" because the two-separate-index approach lets unitary gates reuse all existing ITensors machinery directly (just applied twice — once per leg), and only dissipative gates need to couple the two legs.

**Tag construction warning:** ITensor supports at most 4 tags per `Index`. `siteinds("Qubit", n)` already attaches 3 tags (`"Qubit"`, `"Site"`, `"n=j"`), leaving no room for an additional `"Ket"`/`"Bra"` tag via `addtags`. We therefore construct the indices directly as `Index(2, "Qubit,Ket,n=$j")`, bypassing `siteinds` entirely. This is safe because the `"Qubit"` tag remains, allowing `op("Id", ket_s)` to dispatch correctly.

**`unitary_channel_gates(Umat, js, lsites)`:** Builds two ITensor gates implementing $\rho \mapsto U\rho U^\dagger$ — one for the ket leg (`U`) and one for the bra leg ($\bar U$). Uses the documented `itensor(M, s2', s1', s2, s1)` index ordering for two-site gates (NOT generic `op(M, s1, s2)` which has a different convention). This is the factorized vectorization implementation described in Section 2.2.

**`kraus_channel_gate(Kraus_ops, j, lsites)`:** Reserved for future Kraus-operator comparison. Builds the superoperator $S = \sum_\mu \bar K_\mu \otimes K_\mu$ as a single rank-4 tensor coupling ket and bra legs at site $j$. Currently unused by the main pipeline.

**`identity_liouville_mpo(lsites)`:** Builds the identity MPO $\mathbb{1}$ over the doubled site space, with trivial bond dimension-1 links. Used as the starting point for all `build_open_F*` functions.

### 3.3 `open_product_formula_generation.jl`

**Purpose:** Build the one-step channel MPO $S(\Delta t)$ and its adjoint $S^\dagger(\Delta t)$ for the Heisenberg+amplitude-damping system.

**Unitary part — `heisenberg_bond_unitary(alpha, beta)`:** Builds the dense 4×4 unitary per bond as $R_{ZZ}(\beta)R_{YY}(\alpha)R_{XX}(\alpha)$ with $\alpha = J_j \Delta t$, $\beta = 2J_j\Delta t$. This is identical to the closed-system code's gate convention.

**Dissipator part — direct/brute-force route:**

`single_qubit_dissipator_generator(L, gamma)` builds the dense 4×4 superoperator generator:

$$\mathcal{L}_\text{diss} = \gamma\left(\bar L \otimes L - \frac{1}{2}\mathbb{1}\otimes L^\dagger L - \frac{1}{2}(L^\dagger L)^T\otimes\mathbb{1}\right)$$

`single_qubit_dissipator_channel(L, gamma, dt)` exponentiates it: $S_\text{diss}(\Delta t) = e^{\Delta t\,\mathcal{L}_\text{diss}}$.

This is the **brute-force baseline** — equivalent to the Kraus-based route for amplitude damping (both produce the same $4\times4$ forward-channel matrix), but framed as a direct matrix exponential of the vectorized generator for clarity and generality. The distinction matters for the future Kraus comparison and for generalizing to Lindblad operators without closed-form Kraus decompositions.

**Index convention for dissipator gate — `dissipator_gate_from_matrix(S, j, lsites)`:**

The $4\times4$ matrix $S$ is in "bra-slow, ket-fast" convention (because $\bar L\otimes L$ has bra as the slow/outer factor in $\text{kron}(\bar L, L)$). Reshaping as `reshape(S, 2, 2, 2, 2)` gives a tensor with axis ordering `(ket_out, bra_out, ket_in, bra_in)`, and assigning `ITensor(Stensor, ket_s', bra_s', ket_s, bra_s)` uses the generic `ITensor(array, inds...)` constructor which assigns dimensions in given order directly (no hidden index reversal). This is consistent and verified.

**Product formula orders:**

- **Order 1:** odd unitary layer → even unitary layer → dissipator layer.
- **Order 2 (Strang):** odd(dt/2) → even(dt) → dissipator(dt) → odd(dt/2).
- **Order 4 (Suzuki-Trotter):** five applications of order-2 blocks with coefficients $p_1 = 1/(4-4^{1/3})$, $p_2 = 1-4p_1$.

**Adjoint construction:** `get_open_step_gates_dag` reverses the gate list and replaces each gate by its Hilbert-Schmidt adjoint:
- Unitary gates: `swapprime(dag(g), 0 => 1)` — equivalent to ITensors' standard MPO dagger recipe, valid because the dense matrix IS unitary.
- Dissipator gates: conjugate-transpose of the dense $4\times4$ matrix — NOT the same as running with $-\gamma$ or $-\Delta t$.

**Dissipation toggle:** A `dissipation::Bool=true` keyword is threaded through `get_open_step_gates`, `get_open_step_gates_dag`, `get_open_step_MPO`, `get_open_step_MPO_dag`, and all downstream functions. Setting `dissipation=false` zeroes out `gammas` internally, which causes `dissipator_layer_channel_gates` to produce empty layers (it already skips sites with $\gamma=0$), reducing the computation exactly to the closed-system unitary Heisenberg circuit.

### 3.4 `open_middle_out_contraction.jl`

**Purpose:** Build $\mathbb{F}_{ij}$ and $\mathbb{F}_{\text{ex},j}$ via the middle-out contraction algorithm.

**`op_dag(A::MPO)`:** `swapprime(dag(A), 0 => 1)` — unchanged from closed-system code. Works for the doubled-index MPO because `dag` conjugates all tensors and swaps all primed/unprimed index pairs found, regardless of how many pairs exist per site.

**`left_multiply` / `right_multiply`:** Direct analogues of the closed-system functions. `right_multiply(F, A)` is implemented as `op_dag(apply(op_dag(A), op_dag(F)))` to avoid index-convention issues, identically to the closed-system code.

**`build_open_F`:** Implements the two-clock middle-out contraction (Algorithm 1 of Robertson et al.) for the open system. At each step, either right-multiply by one more forward step $S_j$ (if the $j$-clock is behind) or left-multiply by one more adjoint step $S_i^\dagger$ (if the $i$-clock is behind). The per-step MPOs come from `open_product_formula_generation.jl`.

**`build_open_F_between_lists`:** Generalizes `build_open_F` to allow different lists on the left ($i$) and right ($j$) sides — used for $\mathbb{F}_{\text{ex},j}$ where the left side uses a fine reference $k_0$ and the right side uses the product-formula $k_j$.

**`build_open_F_ex`:** Thin wrapper around `build_open_F_between_lists` with `ks_left=[k0]`, exposing the reference evolution explicitly.

### 3.5 `bond_dimension_tracking.jl`

**Purpose:** Diagnostic tools for MPO bond dimension and operator-Schmidt rank.

**`middle_bond_dim(M::MPO)`:** Returns `linkdims(M)[n÷2]` — the stored bond dimension at the middle link. Cheap.

**`operator_schmidt_rank(M::MPO; cutoff)`:** Orthogonalizes $M$ to site `n÷2` via `orthogonalize!`, then SVDs the middle bond explicitly to count singular values above `cutoff`. Gives the exact rank of the MPO as currently stored (does not retroactively undo prior truncations). More expensive than `middle_bond_dim` but gives a true rank rather than an approximate one.

**`track_Fii_bond_dimension`:** Builds $\mathbb{F}_{ii} = S^\dagger(t/k)^k S(t/k)^k$ incrementally, recording middle bond dimension (and optionally the exact Schmidt rank) after each pair of forward+adjoint steps. The key observable: in the closed system, $F_{ii}$ stays at bond dimension 1 (it equals the identity exactly, which has rank 1). In the open system, $F_{ii}$ is NOT the identity and its bond dimension grows from the first step. The rate of this growth is the central quantity to measure.

**`full_bond_dimension_profile(M::MPO)`:** Returns `linkdims(M)` — the bond dimension at every internal link. Useful to confirm the middle bond is the bottleneck (as expected for a translationally-near-invariant chain) vs. edge bonds.

### 3.6 `F_diagnostics.jl`

**Purpose:** Build the vectorized initial state and compute $\langle\langle\rho(0)|\mathbb{F}_{ij}|\rho(0)\rangle\rangle$ and $\|\mathbb{F}_{ij} - \mathbb{1}\|_F$.

**`vectorized_initial_state_mps(lsites, initially_excited)`:**
- Accepts either `Vector{Int}` (1-indexed) or `Vector{String}` (0-indexed, e.g. `["0","3"]`).
- Builds the product-state MPS $|\rho(0)\rangle\rangle = \bigotimes_j |\rho_j\rangle\rangle$ where $|\rho_j\rangle\rangle = |s_j\rangle_\text{ket}\otimes|s_j\rangle_\text{bra}$ (same state label on both legs, real so no conjugation needed).
- Uses `state(idx, s)` with integer $s \in \{1,2\}$ — works on bare `Index` objects regardless of tags, confirmed by ITensors docs.
- Stitches trivial bond-dimension-1 links the same way as `identity_liouville_mpo`.

**`distance_to_identity(F, lsites)`:** Computes $\|\mathbb{F}-\mathbb{1}\|_F = \sqrt{\langle\langle\mathbb{F}-\mathbb{1}|\mathbb{F}-\mathbb{1}\rangle\rangle}$ via `+(F, -1*Id; cutoff, maxdim)` then `sqrt(real(inner(D,D)))`. Uses Frobenius norm for consistency with the rest of the project (both $\|\rho-\mu\|_F$ in the optimization and the analytical bounds in the project notes use Frobenius). Uses ITensorMPS's `+` overload with scalar coefficient rather than the `add(F, -1.0, Id; ...)` form (whose signature was not verified).

**`expect_F(F, rho0)`:** Returns `inner(rho0', F, rho0)` — complex-valued. The real part equals $M_{ij}$ (see Section 2.3); the imaginary part is numerical noise. Uses the same three-argument `inner(A, M, B)` pattern as the closed-system `optimization_problem.jl`.

**`F_report`:** Convenience function that builds all $\mathbb{F}_{ij}$ via `build_open_F`, computes the expectation-value matrix and distance-to-identity matrix, and returns both. Useful for exploratory analysis; the full optimization workflow goes through `open_optimization_problem.jl` instead.

### 3.7 `open_optimization_problem.jl`

**Purpose:** Compute $M$, $L$, the MPF coefficients $\vec{c}$, and all error estimates.

**`open_gram_matrix`:** Returns $(M, \text{Fs})$ — the Gram matrix using `real(inner(rho0', F, rho0))` and the list of all $\mathbb{F}_{ij}$ MPOs for reuse.

**`open_L_vector`:** Returns $(L, \text{Fex\_list})$ — the overlap vector using the same `real(inner(...))` convention.

**`reference_purity`:** Computes $\text{Tr}(\rho^2(t)) = \|S^{k_\text{ref}}(t/k_\text{ref})|\rho(0)\rangle\rangle\|^2$ by applying the forward channel MPO $k_\text{ref}$ times to `rho0` as an MPS, then taking `real(inner(rho_t, rho_t))`. This is $O(\chi^2)$ in MPS bond dimension, avoiding a full MPO contraction.

**`dynamic_mpf_coefficients`:** Unchanged from `optimization_problem.jl`. Solves the $(r+1)\times(r+1)$ block system via backslash.

**`open_dynamic_mpf_error` / `open_single_trotter_errors`:** Include the purity $\text{Tr}(\rho^2)$ term that the closed-system versions hardcode as 1. For single-Trotter errors, $\text{Tr}(\rho_{k_j}^2) = M[j,j]$ (diagonal of Gram matrix, already computed).

**`test_dynamic_mpf_open`:** Main wrapper. Computes all quantities at two accuracy levels (opt: small maxdim/k_ref for finding $\vec{c}$; ref: large maxdim/k_ref for evaluating errors with fixed $\vec{c}$), mirrors `test_dynamic_mpf_closed` from `error_estimation.jl`. The MPS-TEBD baseline from the closed-system code is **not included** here; it is not meaningful for open systems (TEBD evolves a state, not a density matrix). Replace with the `reference_purity` and reference Gram matrix as accuracy checks.

---

## 4. Verified Results

All results reported below are from the notebook runs (`example_usage.ipynb`), with $n=3$, $J_i \sim \text{Uniform}[1/4, 3/4]$ (seed 1234), $t=3$, $k=[3,8]$, first-order product formula.

### 4.1 Closed-System Sanity Check (dissipation=false)

The expectation-value matrix (Eq. 61 of main.pdf):

$$\langle\langle\rho(0)|\mathbb{F}(t)|\rho(0)\rangle\rangle = \begin{pmatrix} 1.0 & 0.938085 \\ 0.938085 & 1.0 \end{pmatrix}$$

matches the closed-system paper's result exactly (Eq. 14–16 of the paper, same parameters). The diagonal is exactly 1 (numerical noise at $\sim 10^{-15}$), confirming $\mathbb{F}_{ii} = \mathbb{1}$ when $S^\dagger S = \mathbb{1}$ (unitary $S$). Distance-to-identity diagonal is $\sim 10^{-14}$; off-diagonal distance is $\sim 2.57$ (this is large but correct — $\mathbb{F}_{12}$ with $k_1\neq k_2$ is not close to identity as a full operator, even though its matrix element in the specific initial state is close to 1).

The optimization gives $\vec{c} = [-0.1067, 1.1067]$, matching the paper's $\vec{c} = [-0.1067, 1.1067]$ exactly. Purity $\text{Tr}(\rho^2(t)) = 0.9999999999992$ (correct to 12 significant figures). DMPF error $= 0.0250$, single-Trotter errors $= [0.284, 0.0367]$ — DMPF improves over both, as expected.

### 4.2 Open System (dissipation=true, $\gamma=0.05$)

The expectation-value matrix (Eq. 62 of main.pdf):

$$\langle\langle\rho(0)|\mathbb{F}(t)|\rho(0)\rangle\rangle = \begin{pmatrix} 0.5703 & 0.5356 \\ 0.5356 & 0.5698 \end{pmatrix}$$

Diagonal entries well below 1, confirming that $S^\dagger S \neq \mathbb{1}$ for the dissipative channel — the key qualitative result demonstrating that the closed-system tractability argument breaks down.

### 4.3 Bond Dimension

For $n=3$, $k=8$ layers, the middle bond dimension of $\mathbb{F}_{ii}$ saturates at **16 immediately from the first layer** and stays there, with the true operator-Schmidt rank also 16. This is the maximum possible ($d^2 \times d^2 = 4\times4 = 16$ for $n=3$ Liouville space with the middle bond between sites 1 and 2–3), confirming that the near-identity cancellation is completely absent. The open-system $\mathbb{F}_{ii}$ has full operator-Schmidt rank immediately; it cannot be compressed below maxdim=16 for $n=3$.

For larger $n$, the bond dimension ceiling grows as $4^{n/2}$, making the brute-force direct approach intractable at scale (as expected, since this is exactly what motivates the Kraus-based comparison).

---

## 5. Known Issues, Open Problems, and Things for Future Conversations

### 5.1 Bond Dimension Blowup — The Core Open Problem

The brute-force implementation saturates at full bond dimension immediately. This was expected (documented in the project notes already, confirmed numerically). The next step is:

1. **Kraus-operator-based comparison:** `kraus_channel_gate` is already implemented in `liouville_space.jl` (but unused). Wire it into a new file (e.g., `open_product_formula_generation_kraus.jl`) by replacing `amplitude_damping_gate` / `amplitude_damping_dag_gate` with a Kraus-based construction and running the same bond-dimension study. The open question is whether the Kraus route gives meaningfully smaller bond dimension or just shifts where the growth occurs.

2. **System-size scaling study:** Run `track_Fii_bond_dimension` for $n = 3, 4, 5, \ldots$ and several values of $k$, $\gamma$, and $t$, to characterize the scaling law $\chi_\text{mid}(n, k, t, \gamma)$ for both the direct and Kraus routes.

3. **Physical intuition for reduced bond dimension:** The closed-system near-identity argument breaks down as $\mathbb{F}_{ii} \to e^{-\Gamma t} \cdot \mathbb{1} + \text{corrections}$ for some effective decay rate $\Gamma$. There may still be a regime (small $\gamma t$, or for specific Lindblad operators) where $\mathbb{F}_{ii}$ is approximately diagonal in some basis and thus low-rank. This has not been quantified.

### 5.2 Unverified ITensors API Assumptions

The following assumptions were cross-checked against documentation and source but could not be tested without a live Julia session. They should be verified on first run:

- **`apply` with dual-index MPO tensors:** `apply(gates, S)` when $S$'s tensors carry two independent site-index pairs (ket + bra) at each chain position, and the gate only touches one pair. This is the "purification/ancilla" pattern, confirmed to work by ITensors support forum, but not run directly.
- **`inner(rho0', F, rho0)` with MPS tensors carrying two site indices:** Same "ancilla" pattern. If ITensorMPS's `inner` finds index mismatches (because `rho0'` primes all indices including bra, which may or may not be expected by `F`'s input legs), a rewrite using explicit contraction may be needed. The closed-system precedent (`inner(psi', H, psi)`) works for single-index-per-site MPS; extension to two-index is documented but less battle-tested.
- **`+(F, -1*Id; cutoff, maxdim)` for MPO subtraction:** The scalar coefficient form was used (based on confirmed `+(psi, c*phi)` MPS overloads). The exact MPO signature may differ; if it fails, an explicit manual sum loop is the fallback.
- **`MPO(tensors::Vector{ITensor})`:** Used to construct the identity MPO from a manually assembled list of tensors with explicit link indices. The standard ITensorMPS constructor for `MPS` is documented; the `MPO` equivalent is analogous but less explicitly documented.

### 5.3 Order-4 Product Formula Not Tested

The `order=4` path (`get_open_step_gates_order4`, `get_open_step_gates_order4_dag`) was added to the uploaded version of `open_product_formula_generation.jl` but was not used in any notebook run. Its adjoint construction is designed to correctly handle the non-palindromic block structure: the order-4 scheme is $S_4 = B_1 B_2 B_3 B_4 B_5$ with $B_1=B_2=B_4=B_5=S_2(p_1 \Delta t)$ and $B_3=S_2(p_2\Delta t)$, and its adjoint is $B_5^\dagger B_4^\dagger B_3^\dagger B_2^\dagger B_1^\dagger$. Since $p_2 < 0$ for the standard Suzuki coefficients, the sub-step $p_2\Delta t$ is negative. This is handled correctly for unitary layers (a negative $\alpha$ in `heisenberg_bond_unitary` just gives $U^\dagger$ via the cosine/sine formulas). But the dissipator gate at negative $\Delta t$ (`exp(p_2\Delta t \cdot \mathcal{L}_\text{diss})` with $p_2 < 0$) is not a physical channel (exponentiating the Lindbladian with a negative time step is not trace-preserving). This is mathematically valid as part of a higher-order formula but should be flagged and tested carefully.

### 5.4 The MPS Baseline Is Missing for the Open System

`test_dynamic_mpf_closed` in `error_estimation.jl` includes an MPS TEBD baseline (`evolve_mps_tebd`, `mps_baseline_frobenius_error`). The open-system `test_dynamic_mpf_open` does not include an equivalent. A proper open-system baseline would evolve $\rho(0)$ using a Lindblad time-stepper (e.g., a fine-grained first-order channel applied $k_\text{mps}$ times with large $k_\text{mps}$) and use the resulting $|\rho_\text{mps}\rangle\rangle$ MPS as the reference state. This would give the analogue of $E_\text{MPS}$ for benchmarking. Not implemented yet.

### 5.5 Observable Estimation Not Implemented

The closed-system code includes `observable_estimation.jl` with functions for computing $\langle O\rangle = \sum_j c_j \langle O\rangle_{k_j}$. The open-system analogue is $\text{Tr}(O\mu) = \sum_j c_j \text{Tr}(O\rho_{k_j})$, where $\text{Tr}(O\rho_{k_j}) = \langle\langle O|\rho_{k_j}\rangle\rangle$ in Liouville space. This requires (a) vectorizing the observable $O$ as a supervector $|O\rangle\rangle$ and (b) computing inner products with the approximate state MPSs $S^{k_j}|\rho(0)\rangle\rangle$. Not implemented.

### 5.6 Sigma_minus Convention

The Pauli lowering operator is defined as `SIGMA_MINUS = [0 1; 0 0]` (maps column 2 to column 1, i.e., $|1\rangle \mapsto |0\rangle$ in column-vector convention with $|0\rangle = [1,0]^T$, $|1\rangle = [0,1]^T$). This is $|0\rangle\langle 1|$, which is the standard amplitude-damping Lindblad operator. Verify this convention matches the matrix representation used elsewhere in the project before generalizing to other Lindblad operators.

### 5.7 Norm Distance Off-Diagonal Values

The distance-to-identity for off-diagonal $\mathbb{F}_{ij}$ ($i\neq j$) is large ($\sim 2.6$) even in the closed-system case, which is mathematically correct (explained in Section 4.1 — a single matrix element being close to 1 does not imply the full operator is close to identity). However, the open-system distance values ($\sim 2.2$ diagonal, $\sim 2.27$ off-diagonal) show less spread than might be naively expected. A dense-matrix cross-check at the exact system parameters would confirm whether these values are correct or reflect a normalization/convention issue.

---

## 6. File Dependency on Closed-System Code

The open-system files were designed to be **self-contained** and do not `include` any of the original closed-system files. Specifically:

- `middle_out_contraction.jl`, `product_formula_generation.jl`, `direct_MPO_computation.jl`, `error_estimation.jl`, `observable_estimation.jl`, `optimization_problem.jl` are no longer needed as dependencies.

They may be kept as reference implementations (for comparison, cross-checking, or if the closed-system workflow is still used standalone), or removed from the project. The open-system code replicates their functionality with the required Liouville-space extensions.

---

## 7. Quick Reference: Key Equations and Their Code Locations

| Equation | Description | Code location |
|---|---|---|
| Eq. 43 | Vectorized Lindbladian $\mathcal{L}$ | `single_qubit_dissipator_generator` in `open_product_formula_generation.jl` |
| Eq. 44–45 | $e^{t\mathcal{L}}$ and $S^k$ as superoperators | `get_open_step_MPO`, `reference_purity` |
| Eq. 50–51 | $\mathbb{F}_{ij}$, $\mathbb{F}_{\text{ex},j}$ | `build_open_F`, `build_open_F_ex` in `open_middle_out_contraction.jl` |
| Eq. 52–53 | $M_{ij}$, $L_j$ via inner products | `open_gram_matrix`, `open_L_vector` in `open_optimization_problem.jl` |
| Eq. 40 | Optimization problem | `dynamic_mpf_coefficients` (unchanged) |
| Eq. 24 | Block linear system for $\vec{c}$ | `dynamic_mpf_coefficients` |
| Eq. 39 | Cost function with purity | `open_dynamic_mpf_error` |
| Eqs. 14–16 | F-matrix validation (closed system) | `F_report` in `F_diagnostics.jl` |
| Eqs. 61–62 | F-matrix validation (open system) | `F_report` in `F_diagnostics.jl` |

---

## 8. Parameter Conventions

| Parameter | Convention |
|---|---|
| Site numbering | 1-indexed in Julia code; 0-indexed in user-facing `initially_excited` strings |
| $J$ | Vector of length $n-1$, one coupling per bond |
| `gammas` | Vector of length $n$, one rate per site |
| `ks` | Vector of integers, the Trotter step counts to compare |
| `k_ref` / `k0` | Fine-grained reference step count ($\gg \max(k_j)$); used to approximate $e^{t\mathcal{L}}$ |
| `order` | Product formula order (1, 2, or 4); applied to both forward $S$ and adjoint $S^\dagger$ |
| `cutoff` / `maxdim` | SVD truncation parameters for `apply`; `cutoff=0` means no truncation, `maxdim` caps bond dimension |
| `dissipation::Bool` | Toggle: `false` exactly recovers closed-system Heisenberg evolution (zeroes all gammas) |

"""
Hardware-aware dissipative Heisenberg (XXZ) spin chain in Qiskit.

    H = -sum_i [ J_i (X_i X_{i+1} + Y_i Y_{i+1}) + 2 J_i Z_i Z_{i+1} ]

with local amplitude damping

    L_i = sqrt(gamma_i) * sigma^-_i

-----------------------------------------------------------------------------
WHAT THIS BUILDS, AND WHY IT IS PARAMETERISED THIS WAY

The circuit only ever sees two angles:

    theta   the rxx/ryy angle; rzz takes 2*theta
    lam     the cry angle for the damping, with p = sin^2(lam/2)

Everything physical enters through those two and nothing else:

    theta = 2 * J * dt              dt = the Trotter step
    p     = 1 - exp(-gamma * dt)    lam = 2*arcsin(sqrt(p))

so a circuit is fixed by (n, k, theta, lam) and the physical triple (J, gamma,
dt) is a units choice on top. `angles_from_physical` and `angles_from_circuit`
convert either way. Both are plain floats computed in Python rather than
symbolic ParameterExpressions: Qiskit's ParameterExpression does carry .exp()
and .arcsin() through symengine, but it is fragile and the transpiler has been
known to choke on the resulting expression trees. Two abstract Parameters bound
to numbers is equivalent and cannot misbehave.

OPTION B (what this file implements). The Trotter step is FIXED and time
advances by repeating the layer: step s is the time point, with

    J*t     = s * theta/2
    gamma*t = -s * ln(1-p)

so the k time points are k circuits of increasing depth, all with the same
angles. `build_pubs` returns one PUB per step, ready for EstimatorV2.

This is NOT the same as binding a sweep of t into one fixed-k circuit, which
would be Option A: there theta and p grow with t, the depth stays constant, and
every time point is a different (coarser) Trotterisation. Option A has the nice
property of t-independent device noise, but its step size grows along the
series; Option B keeps one well-defined Floquet map, which is what the
tensor-network study characterised.

MATCHING THE TENSOR-NETWORK RUNS. The ITensor code uses
    theta_of(J,t,k) = 2*J*t/k                  <- identical to theta here
    p_of(gamma,t,k) = 1 - exp(-gamma*t/k)      <- identical to p here
    circuit_bond_matrix(theta) = rxx(theta) ryy(theta) rzz(2*theta)
so the two agree gate for gate once the angles match. The one thing that does
NOT match automatically is the initial state: the ITensor runs used
EXCITED=random, a Haar-random product state, which is not what `excited`
(a list of qubits to flip) prepares. Pass the initial_state_*.csv that
MODE=evolve writes to `init_csv` and the preparation layer is reproduced
exactly; that is the only way the two curves are comparable.
"""

from __future__ import annotations

import csv
import math
from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import Layout


# =============================================================================
# Angle conversions
# =============================================================================

def angles_from_physical(J: float, gamma: float, dt: float) -> tuple[float, float]:
    """(theta, lam) from a physical coupling, damping rate and Trotter step."""
    theta = 2.0 * J * dt
    p = 1.0 - math.exp(-gamma * dt)
    return theta, 2.0 * math.asin(math.sqrt(p))


def angles_from_circuit(theta: float, p: float) -> tuple[float, float]:
    """(theta, lam) from the circuit-native pair used by the ITensor study."""
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must be in [0,1), got {p}")
    return theta, 2.0 * math.asin(math.sqrt(p))


def physical_from_angles(theta: float, lam: float, k: int, J: float = 1.0):
    """Invert: pick a J, get (gamma, dt, t) and the invariants (J*t, gamma*t).

    Only gamma/J, J*t and gamma*t are physical; J itself is a units choice, so
    this takes it as an argument rather than pretending to derive it.
    """
    p = math.sin(lam / 2.0) ** 2
    dt = theta / (2.0 * J)
    gamma = -math.log1p(-p) / dt
    t = k * dt
    return dict(J=J, gamma=gamma, dt=dt, t=t, p=p,
                gamma_over_J=gamma / J, Jt=J * t, gamma_t=gamma * t)


# =============================================================================
# Initial state
# =============================================================================

def load_initial_state(path: str) -> list[tuple[float, float]]:
    """Read initial_state_*.csv -> [(u3_theta, u3_phi)] indexed by Qiskit qubit.

    The CSV is written by MODE=evolve and carries, per site, the amplitudes of
    the Haar-random product state the tensor-network run started from, already
    converted to Euler angles:

        |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>  =  U(theta, phi, 0)|0>

    the third Euler angle being free because lambda only phases |0>. Rows carry
    a 0-indexed `qubit` column (= Julia site - 1), and this orders by it, so the
    chain order is preserved and bond (j, j+1) in Julia is (q, q+1) here.

    The bx/by/bz columns are the Bloch vector of each qubit. They are
    convention-independent, so they are the right thing to check against
    hardware: prepare this layer, measure X, Y, Z per qubit BEFORE any
    entangling gate, and compare.
    """
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["qubit"]), float(r["u3_theta"]), float(r["u3_phi"])))
    rows.sort(key=lambda x: x[0])
    if [q for q, _, _ in rows] != list(range(len(rows))):
        raise ValueError(f"{path}: qubit column is not 0..n-1 without gaps")
    return [(th, ph) for _, th, ph in rows]


def _apply_initial_state(qc, system, n, excited=None, prep=None):
    """Depth-1 state preparation. `prep` (from the CSV) wins over `excited`."""
    if prep is not None and excited:
        raise ValueError("pass either `excited` or `init_csv`/`prep`, not both")
    if prep is not None:
        if len(prep) != n:
            raise ValueError(f"prep has {len(prep)} entries, need n={n}")
        for q, (th, ph) in enumerate(prep):
            qc.u(th, ph, 0.0, system[q])
    elif excited:
        for i in excited:
            qc.x(system[int(i)])


# =============================================================================
# One Trotter step
# =============================================================================

def _append_step(qc, system, ancillas, n, theta, lam, dissipation=True):
    """Append ONE Trotter step in place.

    Layout is  q0 q1 a0 q2 q3 a1 q4 q5 a2 ...  so the odd bonds (0,1), (2,3)...
    are already adjacent and the even bonds (1,2), (3,4)... are reached by
    swapping the odd-index system qubit into the ancilla that sits between them.

    Order: odd layer, route, even layer, damping, route back. The routing swap
    is undone at the END OF EVERY STEP regardless of `dissipation`, so after any
    whole number of steps logical qubit j is back in system[j] and an observable
    written on system[j] means what it says. (In the previous version the
    swap-back sat inside `if dissipation:`, so with dissipation=False the chain
    stayed permuted after every step and the gamma=0 control runs were wrong.)
    """
    m = n // 2

    # --- odd bonds (0,1), (2,3), ... -----------------------------------------
    for i in range(0, n - 1, 2):
        qc.rxx(theta, system[i], system[i + 1])
        qc.ryy(theta, system[i], system[i + 1])
        qc.rzz(2 * theta, system[i], system[i + 1])

    # --- route: logical (i+1) into the ancilla between it and (i+2) ----------
    for c, i in enumerate(range(0, n - 1, 2)):
        qc.swap(system[i + 1], ancillas[c])

    # --- even bonds (1,2), (3,4), ... ----------------------------------------
    if n > 2:
        for c, i in enumerate(range(1, n - 1, 2)):
            qc.rxx(theta, ancillas[c], system[i + 1])
            qc.ryy(theta, ancillas[c], system[i + 1])
            qc.rzz(2 * theta, ancillas[c], system[i + 1])

    # --- amplitude damping ---------------------------------------------------
    # system[i+1] is free here (its logical state is in ancillas[c]), so it
    # serves as the damping ancilla for BOTH qubits of the pair, reset between.
    if dissipation:
        for c, i in enumerate(range(0, n - 1, 2)):
            free = system[i + 1]
            qc.cry(lam, system[i], free)       # logical i still in system[i]
            qc.cx(free, system[i])
            qc.reset(free)
            qc.cry(lam, ancillas[c], free)     # logical i+1 is in ancillas[c]
            qc.cx(free, ancillas[c])
            qc.reset(free)

    # --- route back: ALWAYS, not only when damping ---------------------------
    for c, i in enumerate(range(0, n - 1, 2)):
        qc.swap(ancillas[c], system[i + 1])

    # --- odd n: the last qubit is in no pair; damp it against a free ancilla --
    if dissipation and n % 2 == 1:
        qc.cry(lam, system[n - 1], ancillas[m - 1])
        qc.cx(ancillas[m - 1], system[n - 1])
        qc.reset(ancillas[m - 1])


# =============================================================================
# Full circuit
# =============================================================================

def ha_circuit(
    n: int,
    k: int,
    theta=None,
    lam=None,
    *,
    excited: Iterable | None = None,
    init_csv: str | None = None,
    prep: Sequence[tuple[float, float]] | None = None,
    dissipation: bool = True,
    backend: BackendV2 | None = None,
):
    """k Trotter steps, built by appending ONE fixed step k times (Option B).

    theta, lam : float, or None to leave them as circuit Parameters (the
        default, so the primitive does the binding). Pass floats to pre-bind.
    excited : qubits to flip, for a computational-basis initial state.
    init_csv / prep : the Haar-random preparation layer from the tensor-network
        run. Use this when the point is to compare against those curves.

    Returns (qc, init_layout, params) where `params` is {name: Parameter} for
    whichever of theta/lam were left symbolic.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if k < 1:
        raise ValueError("k must be >= 1")

    system = QuantumRegister(n, "q")
    ancillas = QuantumRegister(n // 2, "a")
    qc = QuantumCircuit(system, ancillas)

    params = {}
    if theta is None:
        theta = Parameter("theta"); params["theta"] = theta
    if lam is None:
        lam = Parameter("lam"); params["lam"] = lam

    if init_csv is not None:
        if prep is not None:
            raise ValueError("pass either init_csv or prep, not both")
        prep = load_initial_state(init_csv)
    _apply_initial_state(qc, system, n, excited=excited, prep=prep)

    for _ in range(k):
        _append_step(qc, system, ancillas, n, theta, lam, dissipation=dissipation)

    layout = chain_init_layout(n, backend, system, ancillas) if backend else None
    return qc, layout, params


# =============================================================================
# Observables and PUBs
# =============================================================================

def system_observables(n: int, which=("z", "zz_nn", "zz_center")) -> dict[str, SparsePauliOp]:
    """Z-basis observables on the SYSTEM qubits of the (n + n//2)-wide circuit.

    Returned keys:
        z_{j}      <Z_j> for every system qubit
        mz         (1/n) sum_j <Z_j>              mean magnetisation
        zz_{j}     <Z_j Z_{j+1}> for every bond
        zz_nn      mean over bonds                <- the primary two-body
                   observable: averaging the bonds cancels per-bond fluctuation
                   AND cuts shot noise by sqrt(n-1).
        zz_center  <Z_{n/2-1} Z_{n/2}>            <- single central bond; small
                   and fast-oscillating, so it carries the largest relative
                   Trotter error. Keep it as a secondary check, not the headline.

    These are written on the virtual circuit qubits, which is valid at the end
    of any whole number of Trotter steps because the routing swap is undone
    every step.
    """
    total = n + n // 2
    obs: dict[str, SparsePauliOp] = {}
    if "z" in which:
        for j in range(n):
            obs[f"z_{j}"] = SparsePauliOp.from_sparse_list([("Z", [j], 1.0)], num_qubits=total)
        obs["mz"] = SparsePauliOp.from_sparse_list(
            [("Z", [j], 1.0 / n) for j in range(n)], num_qubits=total)
    if "zz_nn" in which:
        for j in range(n - 1):
            obs[f"zz_{j}"] = SparsePauliOp.from_sparse_list(
                [("ZZ", [j, j + 1], 1.0)], num_qubits=total)
        obs["zz_nn"] = SparsePauliOp.from_sparse_list(
            [("ZZ", [j, j + 1], 1.0 / (n - 1)) for j in range(n - 1)], num_qubits=total)
    if "zz_center" in which and n >= 2:
        c = max(0, n // 2 - 1)
        obs["zz_center"] = SparsePauliOp.from_sparse_list(
            [("ZZ", [c, c + 1], 1.0)], num_qubits=total)
    return obs


def build_pubs(
    n: int,
    k: int,
    theta_val: float,
    lam_val: float,
    *,
    observables=None,
    init_csv: str | None = None,
    prep=None,
    excited: Iterable | None = None,
    dissipation: bool = True,
    backend: BackendV2 | None = None,
    pass_manager=None,
    include_step_zero: bool = True,
):
    """One PUB per Trotter step: [(circuit, observables, parameter_values), ...].

    Option B means the time points are DIFFERENT CIRCUITS of increasing depth,
    not one circuit at different parameter values -- so each PUB carries its own
    circuit, and they all share the same binding. Step s is the time point

        J*t = s*theta/2 ,   gamma*t = -s*ln(1-p) ,   p = sin^2(lam/2)

    `pass_manager` : if given, each circuit is run through it (transpile to the
        backend's ISA). Observables are applied with .apply_layout() to follow.
        Required for hardware; omit for a statevector/density-matrix check.

    Returns (pubs, meta) with meta[i] describing step i: its s, Jt, gamma_t and
    two-qubit gate count, so the time axis and the depth are recorded with the
    data rather than reconstructed later.
    """
    if observables is None:
        observables = system_observables(n)
    obs_list = list(observables.values())
    obs_names = list(observables.keys())

    p = math.sin(lam_val / 2.0) ** 2
    pubs, meta = [], []
    steps = range(0 if include_step_zero else 1, k + 1)
    for s in steps:
        qc, _, params = ha_circuit(
            n, max(s, 1), theta=None, lam=None, excited=excited,
            init_csv=init_csv, prep=prep, dissipation=dissipation, backend=None)
        if s == 0:  # state preparation only
            qc, _, params = ha_circuit(
                n, 1, theta=None, lam=None, excited=excited, init_csv=init_csv,
                prep=prep, dissipation=dissipation, backend=None)
            qc = qc.copy_empty_like()
            _apply_initial_state(qc, qc.qregs[0], n, excited=excited,
                                 prep=prep if prep is not None
                                 else (load_initial_state(init_csv) if init_csv else None))
            params = {}

        binding = {}
        if "theta" in params:
            binding[params["theta"]] = theta_val
        if "lam" in params:
            binding[params["lam"]] = lam_val

        ops = obs_list
        if pass_manager is not None:
            qc = pass_manager.run(qc)
            ops = [o.apply_layout(qc.layout) for o in obs_list]

        # Order the values by qc.parameters, NOT by the dict. qc.parameters is
        # sorted by name, so "lam" comes before "theta" -- passing [theta, lam]
        # positionally would silently swap them.
        values = [binding[par] for par in qc.parameters] if binding else []

        pubs.append((qc, ops, values) if values else (qc, ops))
        meta.append(dict(step=s, Jt=s * theta_val / 2.0,
                         gamma_t=-s * math.log1p(-p) if p > 0 else 0.0,
                         p=p, theta=theta_val, lam=lam_val,
                         two_qubit_gates=sum(1 for inst in qc.data
                                             if inst.operation.num_qubits == 2),
                         depth=qc.depth(), observables=obs_names))
    return pubs, meta


# =============================================================================
# Layout (unchanged)
# =============================================================================

def chain_init_layout(n: int, backend: BackendV2, system_qubits: QuantumRegister,
                      ancillas: QuantumRegister):
    """Map the virtual chain onto the best physical chain the backend offers.

    For n=4 this gives q_0 - q_1 - a_0 - q_2 - q_3 - a_1.
    """
    total_qubits_needed = n + np.floor_divide(n, 2)

    if backend.name == "aer_simulator":
        return None  # a layout only means something on a real/fake backend

    if total_qubits_needed <= 4:
        physical_qubits = backend.properties().general_qlists[0]["qubits"]
    elif total_qubits_needed > 100:
        physical_qubits = (list(range(0, 16)) + [19] + list(range(35, 20, -1)) + [36]
                           + list(range(41, 56, 1)) + [59] + list(range(75, 60, -1)) + [76]
                           + list(range(81, 96, 1)) + [99] + list(range(115, 100, -1)) + [116]
                           + list(range(121, 136, 1)) + [139] + list(range(155, 139, -1)))
    else:
        physical_qubits = backend.properties().general_qlists[total_qubits_needed - 4]["qubits"]

    init_layout = Layout()
    site = 0
    for i in range(0, n, 2):
        init_layout.add(system_qubits[i], physical_qubits[site]); site += 1
        if i + 1 <= (n - 1):
            init_layout.add(system_qubits[i + 1], physical_qubits[site]); site += 1
        j = int(i / 2)
        if j < (np.floor_divide(n, 2)):
            init_layout.add(ancillas[j], physical_qubits[site])
        site += 1
    return init_layout


# =============================================================================
if __name__ == "__main__":
    # The chosen operating point: theta = 0.30, p = 0.016067, k = 15.
    theta_val, lam_val = angles_from_circuit(0.300, 0.016067)
    print(f"theta = {theta_val:.6f}   lam = {lam_val:.6f}   rzz angle = {2*theta_val:.6f}")
    print("physical reading with J = 1:", physical_from_angles(theta_val, lam_val, 15, J=1.0))

    n, k = 8, 15
    pubs, meta = build_pubs(n, k, theta_val, lam_val, excited=[0, 2, 4, 6])
    print(f"\n{len(pubs)} PUBs (steps 0..{k})")
    print(f"{'step':>5} {'J*t':>8} {'gamma*t':>9} {'2q gates':>9} {'depth':>7}")
    for m in meta[:4] + [meta[-1]]:
        print(f"{m['step']:5d} {m['Jt']:8.4f} {m['gamma_t']:9.5f} "
              f"{m['two_qubit_gates']:9d} {m['depth']:7d}")

"""
Hardware-unaware (naive) version of the dissipative Heisenberg (XXZ) spin chain in Qiskit.
 
    H = -sum_i [ J_i (X_i X_{i+1} + Y_i Y_{i+1}) + 2 J_i Z_i Z_{i+1} ]
 
with local amplitude damping
 
    L_i = sqrt(gamma_i) * sigma_i
 
and observables given by the single-qubit populations
 
    n_i = (sigma_i)^dag sigma_i = 0.5* <( I_i - Z_i )>
"""

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2

def sigma_plus(q, num_qubits):
    """|1><0| on qubit q."""
    return SparsePauliOp.from_sparse_list(
        [("X", [q], 0.5), ("Y", [q], -0.5j)], num_qubits=num_qubits)

def sigma_minus(q, num_qubits):
    """|0><1| on qubit q."""
    return SparsePauliOp.from_sparse_list(
        [("X", [q], 0.5), ("Y", [q], 0.5j)], num_qubits=num_qubits)

def population(q, num_qubits):
    return sigma_plus(q, num_qubits)@sigma_minus(q, num_qubits)

def zz_correlator(q, num_qubits):
    """Z_q Z_{q+1} two-qubit correlator."""
    return SparsePauliOp.from_sparse_list(
        [("ZZ", [q, q + 1], 1.0)], num_qubits=num_qubits)

def solve_qc(
    n: int,
    qc: QuantumCircuit,
    tlist: Sequence[float],
    obs: str,
    backend,
    ):

    num_qubits = qc.num_qubits

    if obs == "populations":
        observables = [ population(i, num_qubits) for i in range(n) ]
    elif obs == "correlators":
        observables = [ zz_correlator(i, num_qubits) for i in range(n-1) ]

    estimator = EstimatorV2(mode = backend)

    pubs = [(qc, x, tlist) for x in observables]
    job = estimator.run(pubs)
    pubs_result = job.result()

    evs = { str(i) : None for i in range(len(observables)) }
    for i in range(len(observables)):
        evs[str(i)] = pubs_result[i].data.evs

    return evs
    
"""
Hardware-unaware (naive) version of the dissipative Heisenberg (XXZ) spin chain in Qiskit.
 
    H = -sum_i [ J_i (X_i X_{i+1} + Y_i Y_{i+1}) + 2 J_i Z_i Z_{i+1} ]
 
with local amplitude damping
 
    L_i = sqrt(gamma_i) * sigma_i
 
and observables given by the single-qubit populations
 
    n_i = (sigma_i)^dag sigma_i = 0.5* <( I_i - Z_i )>
"""

from typing import Iterable, Sequence

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2

def naive_circuit(
    J,
    gamma,
    n: int,
    excited: Iterable,
    k: int,
    dissipation: bool = True,
    ):

    system_qubits = QuantumRegister(n, "q")
    ancillas = QuantumRegister(n, "a")

    qc = QuantumCircuit(system_qubits, ancillas)
    t = Parameter("t")

    [qc.x(system_qubits[int(i)]) for i in excited]

    for _ in range(k):
        for i in range(n-1):
            alpha = 2*J[i]*(t/k)
            q0, q1 = system_qubits[i], system_qubits[i+1]
            qc.rxx(alpha, q0, q1)
            qc.ryy(alpha, q0, q1)
            qc.rzz(2*alpha, q0, q1)

        qc.barrier()

        if dissipation:
            for j in range(n):
                theta = 2*(((1 - (-gamma[j]*t/k).exp())**(1/2)).arcsin())
                q, a = system_qubits[j], ancillas[j]
                qc.cry(theta, q, a)
                qc.cx(a, q)

            qc.reset(ancillas)

            qc.barrier()

    return qc

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

def solve_qc(
    n: int,
    qc: QuantumCircuit,
    tlist: Sequence[float],
    backend,
    ):

    num_qubits = qc.num_qubits
    observables = [ population(i, num_qubits) for i in range(n) ]

    estimator = EstimatorV2(mode = backend)

    pubs = [(qc, x, tlist) for x in observables]
    job = estimator.run(pubs)
    pubs_result = job.result()

    evs = { str(i) : None for i in range(n) }
    for i in range(n):
        evs[str(i)] = pubs_result[i].data.evs

    return evs
    
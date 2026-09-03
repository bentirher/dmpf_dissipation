"""
Hardware-aware version of the dissipative Heisenberg (XXZ) spin chain in Qiskit.
 
    H = -sum_i [ J_i (X_i X_{i+1} + Y_i Y_{i+1}) + 2 J_i Z_i Z_{i+1} ]
 
with local amplitude damping
 
    L_i = sqrt(gamma_i) * sigma_i
"""

from typing import Iterable
from qiskit.transpiler import Layout
from qiskit.providers import BackendV2
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
import numpy as np

def ha_circuit(
    J,
    gamma,
    n: int,
    excited: Iterable,
    k: int,
    backend: BackendV2,
    dissipation: bool = True
    ):

    system_qubits = QuantumRegister(n, "q")
    ancillas = QuantumRegister(np.floor_divide(n, 2), "a")

    qc = QuantumCircuit(system_qubits, ancillas)
    t = Parameter("t")

    [qc.x(system_qubits[int(i)]) for i in excited]

    for _ in range(k):
        
        for i in range(0, n-1, 2):
            alpha = 2*J[i]*(t/k)
            q1, q2 = system_qubits[i], system_qubits[i+1]
            qc.rxx(alpha, q1, q2)
            qc.ryy(alpha, q1, q2)
            qc.rzz(2*alpha, q1, q2)

        qc.barrier()

        counter = 0
        for i in range(0, n-1, 2):
            q1, q2 = system_qubits[i], system_qubits[i+1]
            qc.swap(q2, ancillas[counter])
            counter =+ 1

        qc.barrier()

        if n > 2:
            for i in range(1, n-1, 2):
                alpha = 2*J[i]*(t/k)
                q1, q2 = ancillas[counter], system_qubits[i+1]
                qc.rxx(alpha, q1, q2)
                qc.ryy(alpha, q1, q2)
                qc.rzz(2*alpha, q1, q2)

        qc.barrier()

        if dissipation:
            counter = 0
            for j in range(0, n-1, 2):
                theta1, theta2 = 2*(((1 - (-gamma[j]*t/k).exp())**(1/2)).arcsin()), 2*(((1 - (-gamma[j+1]*t/k).exp())**(1/2)).arcsin())
                q1, q2, a = system_qubits[j], ancillas[counter], system_qubits[j+1]
                qc.cry(theta1, q1, a)
                qc.cx(a, q1)
                qc.reset(a)

                qc.cry(theta2, q2, a)
                qc.cx(a, q2)
                qc.reset(a)

            qc.barrier()

            counter = 0
            for j in range(0, n-1, 2):
                q1, q2, a = system_qubits[j], ancillas[counter], system_qubits[j+1]
                qc.swap(q2, a)
                counter =+ 1

            qc.barrier()

    init_layout = chain_init_layout(n, backend, system_qubits, ancillas)           
    return qc, init_layout

def chain_init_layout(n:int, backend: BackendV2, system_qubits:QuantumRegister, ancillas:QuantumRegister):

    """
    Creates an initial layout of $n$ qubits using the best
    chain of physical qubits available in the `backend`.

    Parameters
    ----------
    n : int
        Number of qubits in the system's register.
    backend : BackendV1, BackendV2
        Target backend for the circuit. Used to determine the initial layout.
    system : QuantumRegister
        Register containing the system qubits
    environment: QuantumRegister
        Register containing the ancillary qubits

    Returns
    -------
    Layout

    Description
    -----------
   
    Examples
    --------
    For a Markovian chain of $n=4$, this would return a layout that maps
    the virtual qubits to physical qubits as q_0 - q_1 - e_0 - q_2 - q_3 - e_1.

    """

    total_qubits_needed = n + np.floor_divide(n, 2)

    if backend.name == 'aer_simulator':
        return None # Layout only makes sense for real/fake backend simulations
        
    else:
        if total_qubits_needed <= 4:
            physical_qubits = backend.properties().general_qlists[0]['qubits']

        elif total_qubits_needed > 100:
            physical_qubits = list(range(0, 16)) + [19] + list(range(35, 20, -1))+ [36] + list(range(41, 56, 1)) + [59] + list(range(75, 60, -1)) + [76] + list(range(81, 96, 1)) + [99] + list(range(115, 100, -1)) + [116] + list(range(121, 136, 1)) + [139] + list(range(155, 139, -1))
        
        else:
            physical_qubits = backend.properties().general_qlists[total_qubits_needed-4]['qubits']

            
        init_layout = Layout()
        site = 0

        for i in range(0,n,2):
            init_layout.add(system_qubits[i], physical_qubits[site])
            site = site + 1

            if i+1<=(n-1):
                init_layout.add(system_qubits[i+1], physical_qubits[site])
                site = site + 1

            j = int(i/2)

            if j<(np.floor_divide(n, 2)):
                init_layout.add(ancillas[j], physical_qubits[site])

            site = site +1
    
        return init_layout
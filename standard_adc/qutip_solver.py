"""
Dissipative Heisenberg (XXZ) spin chain in QuTiP.
 
    H = -sum_i [ J_i (X_i X_{i+1} + Y_i Y_{i+1}) + 2 J_i Z_i Z_{i+1} ]
 
with optional local amplitude damping
 
    L_i = sqrt(gamma_i) * sigma^-_i
 
and observables given by the single-qubit populations
 
    n_i = (sigma^-_i)^dag sigma^-_i .
 
 
Convention note
---------------
`qutip.sigmam()` is [[0, 0], [1, 0]] = (X - iY)/2, i.e. exactly the operator you
wrote up to the factor 1/2.  It maps basis(2, 0) -> basis(2, 1), so in QuTiP's
spin convention **basis(2, 0) is the excited ("up") state** and basis(2, 1) is
the ground state.  With that identification everything is self-consistent:
sigmam().dag() * sigmam() = |0><0| is the projector onto the excited level.
 
If you prefer the quantum-information labelling (|1> = excited), set
`excited_level=1`; the lowering operator then becomes `sigmap()` and the excited
ket becomes basis(2, 1).  The Hamiltonian is invariant under relabelling
(X -> X, Y -> -Y, Z -> -Z leaves XX, YY and ZZ unchanged), so both choices give
numerically identical population curves -- only the basis labels differ.
 
The factor of 1/2 does matter, though: using L_i = sqrt(gamma_i) (X_i - i Y_i)
literally would give an effective decay rate of 4*gamma_i.  This module uses
sqrt(gamma_i) * sigmam(), so gamma_i is the physical decay rate.
"""
 
from __future__ import annotations
 
from typing import Iterable, Sequence
 
import numpy as np
from qutip import (Qobj, basis, mesolve, qeye, sigmam, sigmap, sigmax, sigmay,
                   sigmaz, tensor)
 
__all__ = ["build_hamiltonian", "build_initial_state", "lowering_op",
           "simulate_heisenberg"]
 
 
# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _embed(op: Qobj, i: int, n: int) -> Qobj:
    """n-qubit operator with `op` acting on site i and identity elsewhere."""
    ops = [qeye(2)] * n
    ops[i] = op
    return tensor(ops)

def _embed_correlator(op: Qobj, i: int, j: int, n: int) -> Qobj:
    """n-qubit operator with `op` acting on site i, j, and identity elsewhere."""
    ops = [qeye(2)] * n
    ops[i] = op
    ops[j]= op
    return tensor(ops)
 
 
def lowering_op(excited_level: int = 0) -> Qobj:
    """Single-qubit lowering operator that de-excites `basis(2, excited_level)`."""
    if excited_level == 0:
        return sigmam()          # (X - iY)/2 = |1><0| : basis(2,0) -> basis(2,1)
    if excited_level == 1:
        return sigmap()          # (X + iY)/2 = |0><1| : basis(2,1) -> basis(2,0)
    raise ValueError("excited_level must be 0 or 1.")
 
def _n_bonds(n: int, periodic: bool) -> int:
    # a ring only makes sense for n > 2; for n == 2 it would double-count the bond
    return n if (periodic and n > 2) else n - 1
 
 
def _broadcast(values, size: int, name: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    if arr.size == 1 and size != 1:
        arr = np.full(size, arr[0])
    if arr.size != size:
        raise ValueError(f"`{name}` has length {arr.size}, expected {size} (or a scalar).")
    return arr
 
 
# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_hamiltonian(J, n: int, periodic: bool = False) -> Qobj:
    """
    Heisenberg XXZ Hamiltonian on `n` qubits.
 
    Parameters
    ----------
    J : float or sequence
        Coupling strength per bond: length `n-1` for an open chain, `n` for a
        periodic ring.  A scalar is broadcast to every bond.
    n : int
        Number of qubits.
    periodic : bool
        Close the chain with a bond between qubit n-1 and qubit 0.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
 
    nb = _n_bonds(n, periodic)
    H = 0 * tensor([qeye(2)] * n)          # zero operator with the right dims
    if nb == 0:                            # single qubit: no bonds
        return H
 
    J = _broadcast(J, nb, "J")
    for b in range(nb):
        i, j = b, (b + 1) % n
        XX = _embed(sigmax(), i, n) * _embed(sigmax(), j, n)
        YY = _embed(sigmay(), i, n) * _embed(sigmay(), j, n)
        ZZ = _embed(sigmaz(), i, n) * _embed(sigmaz(), j, n)
        H += -J[b] * (XX + YY + 2.0 * ZZ)
    return H
 
 
def build_initial_state(n: int, excited: Iterable, excited_level: int = 0) -> Qobj:
    """
    Product state with the listed qubits excited.
 
    `excited` is a list of site indices given as strings or ints, e.g.
    ["0", "2"] puts qubits 0 and 2 in basis(2, excited_level) and the rest in
    the complementary (ground) level.
    """
    idx = {int(k) for k in excited}
    bad = [k for k in idx if not 0 <= k < n]
    if bad:
        raise ValueError(f"Excited indices {sorted(bad)} out of range for n = {n}.")
    ground_level = 1 - excited_level
    return tensor([basis(2, excited_level if i in idx else ground_level)
                   for i in range(n)])
 
 
# --------------------------------------------------------------------------- #
# main driver
# --------------------------------------------------------------------------- #
def simulate_heisenberg(
    J,
    gamma,
    n: int,
    excited: Iterable,
    tlist: Sequence[float],
    dissipation: bool = True,
    obs : str = "populations",
    periodic: bool = False,
    excited_level: int = 0,
    options: dict | None = None,
):
    """
    Evolve the dissipative Heisenberg chain and return the single-qubit populations.
 
    Parameters
    ----------
    J : float or sequence
        Coupling strengths (see `build_hamiltonian`).
    gamma : float or sequence of length n
        Amplitude-damping rate per qubit.  Ignored when `dissipation=False`.
    n : int
        Number of qubits.
    excited : iterable of str or int
        Indices of the initially excited qubits, e.g. ["0", "2"].
    tlist : array_like
        Times at which the solution is evaluated.
    dissipation : bool
        Toggle the collapse operators.  False -> unitary (Schrodinger) evolution.
    periodic : bool
        Periodic boundary conditions.
    excited_level : {0, 1}
        Which QuTiP basis level counts as "excited"; see the module docstring.
    options : dict, optional
        Forwarded to `qutip.mesolve`.
 
    Returns
    -------
    populations : ndarray, shape (n, len(tlist))
        populations[i] = <n_i(t)>.
    result : qutip.solver.Result
        Full solver output.
    """
    tlist = np.asarray(tlist, dtype=float)
 
    H = build_hamiltonian(J, n, periodic=periodic)
    psi0 = build_initial_state(n, excited, excited_level=excited_level)

    s = lowering_op(excited_level)
    sm = [_embed(s, i, n) for i in range(n)]      # sigma^-_i
    
    if obs == "populations":
        e_ops = [o.dag() * o for o in sm]             # n_i = sigma^+_i sigma^-_i
    elif obs == "correlators":
        s = sigmaz()
        e_ops = [_embed_correlator(s, i, i+1, n) for i in range(n-1)]    # Z_iZ_{i+1}
 
    if dissipation:
        g = _broadcast(gamma, n, "gamma")
        if np.any(g < 0):
            raise ValueError("Decay rates gamma must be non-negative.")
        c_ops = [np.sqrt(g[i]) * sm[i] for i in range(n) if g[i] > 0]
    else:
        c_ops = []
 
    result = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=e_ops,
                     options=options or {})
    return np.array(result.expect), result
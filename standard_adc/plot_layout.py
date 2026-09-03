"""Plot the physical qubits a transpiled circuit occupies on an IBM heavy-hex device.

Typical use:

    from heavy_hex_layout import plot_layout, plot_layouts

    isa_qc = pm.run(qc)
    plot_layout(isa_qc, backend)                      # one circuit
    plot_layouts([pm.run(q) for q in circuits], backend)   # a grid of them
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

__all__ = [
    "heavy_hex_coordinates",
    "used_qubits",
    "plot_layout",
    "plot_layouts",
]

USED_COLOR = "#e8492e"
IDLE_COLOR = "#f4f4f4"
LINE_COLOR = "#d6d6d6"
NODE_EDGE_COLOR = "#9a9a9a"

#: instruction names that occupy qubits without being gates
NON_GATES = frozenset({"barrier", "delay"})

_COORD_CACHE: dict[tuple, dict[int, tuple[float, float]]] = {}


# --------------------------------------------------------------------------- #
# backend plumbing
# --------------------------------------------------------------------------- #

def _coupling_map(backend):
    """Accept either a backend or a CouplingMap."""
    cm = getattr(backend, "coupling_map", backend)
    if not hasattr(cm, "get_edges"):
        raise TypeError(
            "expected a BackendV2 or a CouplingMap, got "
            f"{type(backend).__name__}"
        )
    return cm


def _num_qubits(backend, coupling_map):
    n = getattr(backend, "num_qubits", None)
    if n is None:
        n = coupling_map.size()
    return n


def _cache_key(backend, coupling_map):
    name = getattr(backend, "name", None)
    if callable(name):          # BackendV1 exposes name() as a method
        name = name()
    return (name, _num_qubits(backend, coupling_map))


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def heavy_hex_coordinates(backend, validate=True, use_cache=True):
    """Return ``{qubit: (x, y)}`` laying a heavy-hex device out as on IBM Quantum.

    The coordinates are derived from the coupling map alone, using the fact that
    IBM numbers these devices row-major: a horizontal rail is a maximal run of
    consecutive indices that are actually connected, and every remaining qubit is
    a vertical rung between two rails.

    Rails sit two units apart in ``y`` and rungs sit halfway between them, so
    every coupling-map edge has unit length and the plot comes out square.

    Parameters
    ----------
    backend : BackendV2 or CouplingMap
    validate : bool
        Check that every edge is unit length and raise if not. Cheap; leave on.
    use_cache : bool
        Reuse coordinates previously computed for the same backend name.

    Raises
    ------
    ValueError
        If the device is not laid out the way this function assumes.
    """
    coupling_map = _coupling_map(backend)
    num_qubits = _num_qubits(backend, coupling_map)

    key = _cache_key(backend, coupling_map)
    if use_cache and key[0] is not None and key in _COORD_CACHE:
        return dict(_COORD_CACHE[key])

    adj = defaultdict(set)
    for a, b in coupling_map.get_edges():
        adj[a].add(b)
        adj[b].add(a)

    # maximal runs of consecutive, connected indices -> horizontal rails
    runs, cur = [], [0]
    for q in range(1, num_qubits):
        if q in adj[q - 1]:
            cur.append(q)
        else:
            runs.append(cur)
            cur = [q]
    runs.append(cur)

    rails = [r for r in runs if len(r) > 1]
    rungs = [q for r in runs if len(r) == 1 for q in r]
    if len(rails) < 2:
        raise ValueError(
            "coupling map does not look like a row-major heavy-hex device "
            f"({len(rails)} rails found)"
        )

    grid = {q: (ri, j) for ri, rail in enumerate(rails) for j, q in enumerate(rail)}

    # each rung connects one rail above to one rail below
    rung_anchors = {}
    for c in rungs:
        anchors = [grid[nb] for nb in adj[c] if nb in grid]
        if len(anchors) != 2:
            raise ValueError(
                f"qubit {c} is not a clean rung: it touches {len(anchors)} rail "
                "qubits, expected 2"
            )
        rung_anchors[c] = tuple(sorted(anchors))

    # rails may be horizontally offset from one another; propagate via the rungs
    offset = {0: 0}
    while len(offset) < len(rails):
        progress = False
        for (r1, j1), (r2, j2) in rung_anchors.values():
            if r1 in offset and r2 not in offset:
                offset[r2] = offset[r1] + j1 - j2
                progress = True
            elif r2 in offset and r1 not in offset:
                offset[r1] = offset[r2] + j2 - j1
                progress = True
        if not progress:
            raise ValueError("coupling map is disconnected")

    coords = {q: (offset[ri] + j, -2 * ri) for q, (ri, j) in grid.items()}
    for c, ((r1, j1), (r2, j2)) in rung_anchors.items():
        coords[c] = (offset[r1] + j1, -(r1 + r2))

    if validate:
        bad = [
            (a, b)
            for a, b in coupling_map.get_edges()
            if abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) != 1
        ]
        if bad:
            raise ValueError(
                f"{len(bad)} coupling-map edges are not unit length "
                f"(e.g. {bad[:3]}); this device is probably not numbered row-major"
            )

    if key[0] is not None:
        _COORD_CACHE[key] = dict(coords)
    return coords


# --------------------------------------------------------------------------- #
# circuit inspection
# --------------------------------------------------------------------------- #

def used_qubits(isa_circuit):
    """Physical qubits and connections a transpiled circuit actually touches.

    Returns ``(touched, edges)`` where ``touched`` is a set of physical qubit
    indices and ``edges`` is a set of ``frozenset({a, b})`` pairs carrying at
    least one two-qubit gate.

    This reads the circuit rather than its ``layout``: routing can push SWAPs
    through qubits that were assigned as ancillas, so ``touched`` is often a
    superset of the initial layout.
    """
    index = {bit: i for i, bit in enumerate(isa_circuit.qubits)}

    touched, edges = set(), set()
    for inst in isa_circuit.data:
        if inst.operation.name in NON_GATES:
            continue
        qubits = [index[q] for q in inst.qubits]
        touched.update(qubits)
        if len(qubits) == 2:
            edges.add(frozenset(qubits))
    return touched, edges


def two_qubit_depth(circuit):
    """Circuit depth counting only genuine multi-qubit gates (barriers excluded)."""
    return circuit.depth(
        lambda instr: instr.operation.num_qubits > 1
        and instr.operation.name not in NON_GATES
    )


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def _draw(ax, coords, coupling_map, touched, edges, node_size, line_width):
    for a, b in coupling_map.get_edges():
        hot = frozenset((a, b)) in edges
        ax.add_line(
            Line2D(
                *zip(coords[a], coords[b]),
                color=USED_COLOR if hot else LINE_COLOR,
                lw=line_width * (2.5 if hot else 1.0),
                zorder=2 if hot else 1,
                solid_capstyle="round",
            )
        )

    qubits = sorted(coords)
    xs = [coords[q][0] for q in qubits]
    ys = [coords[q][1] for q in qubits]
    colors = [USED_COLOR if q in touched else IDLE_COLOR for q in qubits]
    ax.scatter(
        xs, ys, s=node_size, c=colors,
        edgecolors=NODE_EDGE_COLOR, linewidths=0.7, zorder=3,
    )

    ax.set_aspect("equal")
    ax.margins(0.04)
    ax.axis("off")
    return ax


def plot_layout(
    isa_circuit,
    backend,
    ax=None,
    coordinates=None,
    node_size=110,
    line_width=1.2,
    scale=0.42,
):
    """Plot the backend coupling map with the circuit's physical qubits highlighted.

    Parameters
    ----------
    isa_circuit : QuantumCircuit
        A transpiled circuit defined on the backend's physical qubits.
    backend : BackendV2 or CouplingMap
    ax : matplotlib Axes, optional
        Draw here instead of creating a new figure.
    coordinates : dict, optional
        Precomputed ``heavy_hex_coordinates`` output, to skip recomputation.
    scale : float
        Inches per lattice unit when creating the figure.

    Returns
    -------
    matplotlib Axes
    """
    coupling_map = _coupling_map(backend)
    coords = coordinates or heavy_hex_coordinates(backend)
    touched, edges = used_qubits(isa_circuit)

    if ax is None:
        xs = [x for x, _ in coords.values()]
        ys = [y for _, y in coords.values()]
        width = max(3.0, (max(xs) - min(xs) + 2) * scale)
        height = max(3.0, (max(ys) - min(ys) + 2) * scale)
        _, ax = plt.subplots(figsize=(width, height))

    return _draw(ax, coords, coupling_map, touched, edges, node_size, line_width)


def plot_layouts(
    isa_circuits,
    backend,
    ncols=4,
    titles=None,
    node_size=90,
    line_width=1.2,
    scale=0.34,
    **subplot_kwargs,
):
    """Plot several transpiled circuits on a grid of coupling maps.

    ``titles`` may be a list of strings (one per circuit) or ``None`` for none.
    Returns ``(fig, axes)`` with ``axes`` flattened and unused panels hidden.
    """
    isa_circuits = list(isa_circuits)
    coords = heavy_hex_coordinates(backend)

    n = len(isa_circuits)
    ncols = min(ncols, n) or 1
    nrows = -(-n // ncols)

    xs = [x for x, _ in coords.values()]
    ys = [y for _, y in coords.values()]
    panel_w = (max(xs) - min(xs) + 2) * scale
    panel_h = (max(ys) - min(ys) + 2) * scale

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        squeeze=False,
        **subplot_kwargs,
    )
    axes = axes.ravel()

    for ax, circuit in zip(axes, isa_circuits):
        plot_layout(
            circuit, backend, ax=ax, coordinates=coords,
            node_size=node_size, line_width=line_width,
        )
    if titles is not None:
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=11)
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes
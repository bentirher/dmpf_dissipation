"""Plot the initial layout a transpiler chose, coloured by quantum register.

    from plot_initial_layout import plot_layout, plot_layouts

    isa_qc = pm.run(qc)
    plot_layout(isa_qc, backend)

System qubits (register "q") and ancillas (register "a") get different colours;
qubits the circuit does not use stay pale, and every coupling-map connection is
drawn in grey.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

__all__ = ["heavy_hex_coordinates", "register_layout", "plot_layout", "plot_layouts"]

SYSTEM_COLOR = "#6c169f"
ANCILLA_COLOR = "#f4bca6"

#: register name -> colour; extend or override via the `colors` argument
DEFAULT_COLORS = {"q": SYSTEM_COLOR, "a": ANCILLA_COLOR}

#: colours for any further registers, used in order of first appearance
FALLBACK_COLORS = ["#2a9d8f", "#e76f51", "#457b9d", "#b5179e"]

IDLE_COLOR = "#f4f4f4"
LINE_COLOR = "#d6d6d6"
NODE_EDGE_COLOR = "#9a9a9a"

#: registers added by the transpiler to pad the circuit out to the device size
FILLER_REGISTERS = ("ancilla",)


# --------------------------------------------------------------------------- #
# device geometry
# --------------------------------------------------------------------------- #

def heavy_hex_coordinates(backend):
    """``{physical qubit: (x, y)}`` laying an IBM heavy-hex device out as on the platform.

    Derived from the coupling map alone: IBM numbers these devices row-major, so a
    horizontal rail is a maximal run of consecutive connected indices and every
    other qubit is a vertical rung between two rails. Rails sit two units apart
    and rungs halfway between, so every connection has unit length.
    """
    coupling_map = getattr(backend, "coupling_map", backend)
    num_qubits = getattr(backend, "num_qubits", None) or coupling_map.size()

    adj = defaultdict(set)
    for a, b in coupling_map.get_edges():
        adj[a].add(b)
        adj[b].add(a)

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
        raise ValueError("coupling map does not look like a row-major heavy-hex device")

    grid = {q: (ri, j) for ri, rail in enumerate(rails) for j, q in enumerate(rail)}

    anchors = {}
    for c in rungs:
        touching = [grid[nb] for nb in adj[c] if nb in grid]
        if len(touching) != 2:
            raise ValueError(f"qubit {c} touches {len(touching)} rail qubits, expected 2")
        anchors[c] = tuple(sorted(touching))

    offset = {0: 0}
    while len(offset) < len(rails):
        progress = False
        for (r1, j1), (r2, j2) in anchors.values():
            if r1 in offset and r2 not in offset:
                offset[r2] = offset[r1] + j1 - j2
                progress = True
            elif r2 in offset and r1 not in offset:
                offset[r1] = offset[r2] + j2 - j1
                progress = True
        if not progress:
            raise ValueError("coupling map is disconnected")

    coords = {q: (offset[ri] + j, -2 * ri) for q, (ri, j) in grid.items()}
    for c, ((r1, j1), (r2, j2)) in anchors.items():
        coords[c] = (offset[r1] + j1, -(r1 + r2))

    bad = [
        (a, b)
        for a, b in coupling_map.get_edges()
        if abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) != 1
    ]
    if bad:
        raise ValueError(
            f"{len(bad)} connections are not unit length (e.g. {bad[:3]}); "
            "this device is probably not numbered row-major"
        )
    return coords


# --------------------------------------------------------------------------- #
# reading the layout
# --------------------------------------------------------------------------- #

def _register_name(bit):
    """Name of the QuantumRegister a Qubit belongs to, or None."""
    reg = getattr(bit, "_register", None)
    if reg is None:
        try:
            reg = bit.register
        except (AttributeError, DeprecationWarning):
            return None
    return getattr(reg, "name", None)


def register_layout(isa_circuit, ignore=FILLER_REGISTERS):
    """``{physical qubit: register name}`` from ``isa_circuit.layout.initial_layout``.

    Only the qubits of the circuit as you wrote it are returned; the padding
    register the transpiler adds (named in ``ignore``) is dropped.

    This is the assignment chosen by the layout pass, before routing. Use
    ``isa_circuit.layout.routing_permutation()`` if you need where things end up.
    """
    layout = getattr(isa_circuit, "layout", None)
    if layout is None:
        raise ValueError(
            "circuit has no layout; pass a circuit returned by the pass manager"
        )

    assignment = {}
    for bit, physical in layout.initial_layout.get_virtual_bits().items():
        name = _register_name(bit)
        if name is None or name in ignore:
            continue
        assignment[physical] = name
    return assignment


def _color_map(register_names, colors):
    """Resolve a colour for every register name, in order of first appearance."""
    palette = dict(DEFAULT_COLORS)
    palette.update(colors or {})

    spare = iter(FALLBACK_COLORS)
    resolved = {}
    for name in register_names:
        resolved[name] = palette.get(name) or next(spare, "#777777")
    return resolved


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def plot_layout(
    isa_circuit,
    backend,
    ax=None,
    coordinates=None,
    colors=None,
    legend=True,
    node_size=110,
    line_width=1.2,
    scale=0.42,
):
    """Draw the coupling map with the initial layout coloured by register.

    Parameters
    ----------
    isa_circuit : QuantumCircuit
        A transpiled circuit carrying a layout.
    backend : BackendV2 or CouplingMap
    colors : dict, optional
        Register name to colour, e.g. ``{"q": "#6c169f", "a": "#f4bca6"}``.
        Overrides the defaults; unknown registers get a fallback colour.
    legend : bool
        Label the registers in the corner.
    coordinates : dict, optional
        Reuse a previous ``heavy_hex_coordinates`` result.

    Returns
    -------
    matplotlib Axes
    """
    coupling_map = getattr(backend, "coupling_map", backend)
    coords = coordinates if coordinates is not None else heavy_hex_coordinates(backend)
    assignment = register_layout(isa_circuit)

    # registers in order of first appearance, so colours stay stable across sizes
    order = []
    for physical in sorted(assignment):
        name = assignment[physical]
        if name not in order:
            order.append(name)
    palette = _color_map(order, colors)

    if ax is None:
        xs = [x for x, _ in coords.values()]
        ys = [y for _, y in coords.values()]
        width = max(3.0, (max(xs) - min(xs) + 2) * scale)
        height = max(3.0, (max(ys) - min(ys) + 2) * scale)
        _, ax = plt.subplots(figsize=(width, height))

    for a, b in coupling_map.get_edges():
        ax.add_line(
            Line2D(
                *zip(coords[a], coords[b]),
                color=LINE_COLOR, lw=line_width, zorder=1, solid_capstyle="round",
            )
        )

    qubits = sorted(coords)
    ax.scatter(
        [coords[q][0] for q in qubits],
        [coords[q][1] for q in qubits],
        s=node_size,
        c=[palette.get(assignment.get(q), IDLE_COLOR) for q in qubits],
        edgecolors=NODE_EDGE_COLOR,
        linewidths=0.7,
        zorder=2,
    )

    if legend and order:
        ax.legend(
            handles=[
                Line2D([], [], marker="o", linestyle="none", markersize=8,
                       markerfacecolor=palette[name], markeredgecolor=NODE_EDGE_COLOR,
                       label=name)
                for name in order
            ],
            loc="upper right", frameon=False, fontsize=9, handletextpad=0.3,
        )

    ax.set_aspect("equal")
    ax.margins(0.04)
    ax.axis("off")
    return ax


def plot_layouts(
    isa_circuits,
    backend,
    ncols=4,
    titles=None,
    colors=None,
    legend=True,
    node_size=90,
    line_width=1.2,
    scale=0.34,
):
    """Draw several transpiled circuits on a grid, one coupling map each.

    ``legend`` puts a single legend on the figure rather than one per panel.
    Returns ``(fig, axes)`` with ``axes`` flattened.
    """
    isa_circuits = list(isa_circuits)
    coords = heavy_hex_coordinates(backend)

    n = len(isa_circuits)
    ncols = min(ncols, n) or 1
    nrows = -(-n // ncols)

    xs = [x for x, _ in coords.values()]
    ys = [y for _, y in coords.values()]
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=((max(xs) - min(xs) + 2) * scale * ncols,
                 (max(ys) - min(ys) + 2) * scale * nrows),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, circuit in zip(axes, isa_circuits):
        plot_layout(
            circuit, backend, ax=ax, coordinates=coords, colors=colors,
            legend=False, node_size=node_size, line_width=line_width,
        )
    if titles is not None:
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=11)
    for ax in axes[n:]:
        ax.axis("off")

    if legend and isa_circuits:
        names, seen = [], register_layout(isa_circuits[-1])
        for physical in sorted(seen):
            if seen[physical] not in names:
                names.append(seen[physical])
        palette = _color_map(names, colors)
        fig.legend(
            handles=[
                Line2D([], [], marker="o", linestyle="none", markersize=8,
                       markerfacecolor=palette[name], markeredgecolor=NODE_EDGE_COLOR,
                       label=name)
                for name in names
            ],
            loc="lower center", ncol=len(names), frameon=False, fontsize=10,
        )

    fig.tight_layout()
    return fig, axes

"""
Lattice & Structured Graph Construction Algorithms.

This subpackage contains 9 graph builders that produce deterministic,
regular graph topologies:

1.  Grid / Lattice (2D)
2.  Ring / Cycle
3.  Star
4.  Complete Bipartite
5.  Hypercube
6.  Torus
7.  Hexagonal Lattice
8.  Triangular Lattice
9.  Petersen Graph

Unlike other categories, these builders generate their own canonical
node positions since the graph topology defines the geometry.  The
``layout.n_points`` is used only as a rough target for the number of
nodes.  Each builder stores the computed positions in
``G.graph["positions"]`` as an (n, 2) array.

These graphs use the :class:`PointLayout` from ``layout`` only for
node count guidance.  Visualization should use the positions stored
on the graph rather than the layout positions.
"""

from graphgallery.lattice.grid import GridGraph
from graphgallery.lattice.ring import RingGraph
from graphgallery.lattice.star import StarGraph
from graphgallery.lattice.complete_bipartite import CompleteBipartiteGraph
from graphgallery.lattice.hypercube import HypercubeGraph
from graphgallery.lattice.torus import TorusGraph
from graphgallery.lattice.hexagonal import HexagonalLatticeGraph
from graphgallery.lattice.triangular_lattice import TriangularLatticeGraph
from graphgallery.lattice.petersen import PetersenGraph

__all__ = [
    "GridGraph",
    "RingGraph",
    "StarGraph",
    "CompleteBipartiteGraph",
    "HypercubeGraph",
    "TorusGraph",
    "HexagonalLatticeGraph",
    "TriangularLatticeGraph",
    "PetersenGraph",
]


def all_lattice_builders():
    """Return instances of every lattice/structured builder with defaults."""
    return [
        GridGraph(),
        RingGraph(),
        StarGraph(),
        CompleteBipartiteGraph(),
        HypercubeGraph(),
        TorusGraph(),
        HexagonalLatticeGraph(),
        TriangularLatticeGraph(),
        PetersenGraph(),
    ]

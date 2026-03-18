"""
Miscellaneous Graph Construction Algorithms.

This subpackage contains 7 graph builders that span diverse domains
and don't fit cleanly into the other categories:

1.  Power Diagram Graph
2.  KD-Tree Neighbor Graph
3.  Ball Tree Neighbor Graph
4.  Disk Graph
5.  Intersection Graph
6.  De Bruijn Graph
7.  Cayley Graph

These range from spatial data structures (KD-tree, Ball tree) and
computational geometry (Power diagram, Disk graph) to combinatorics
(De Bruijn) and abstract algebra (Cayley).

Some builders (De Bruijn, Cayley) generate their own topology
independent of point coordinates, similar to the lattice subpackage.
Others (KD-tree, Ball tree, Disk, Power diagram) operate directly
on the spatial layout.
"""

from graphgallery.misc.power_diagram import PowerDiagramGraph
from graphgallery.misc.kdtree_neighbor import KDTreeNeighborGraph
from graphgallery.misc.balltree_neighbor import BallTreeNeighborGraph
from graphgallery.misc.disk import DiskGraph
from graphgallery.misc.intersection import IntersectionGraph
from graphgallery.misc.debruijn import DeBruijnGraph
from graphgallery.misc.cayley import CayleyGraph

__all__ = [
    "PowerDiagramGraph",
    "KDTreeNeighborGraph",
    "BallTreeNeighborGraph",
    "DiskGraph",
    "IntersectionGraph",
    "DeBruijnGraph",
    "CayleyGraph",
]


def all_misc_builders():
    """Return instances of every miscellaneous builder with defaults."""
    return [
        PowerDiagramGraph(),
        KDTreeNeighborGraph(),
        BallTreeNeighborGraph(),
        DiskGraph(),
        IntersectionGraph(),
        DeBruijnGraph(),
        CayleyGraph(),
    ]

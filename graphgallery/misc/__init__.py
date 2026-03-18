"""
Miscellaneous Graph Construction Algorithms.

This subpackage contains graph builders that span diverse domains
and don't fit cleanly into the other categories:

1.  KD-Tree Neighbor Graph
2.  Ball Tree Neighbor Graph
3.  Disk Graph
4.  Intersection Graph

These range from spatial data structures (KD-tree, Ball tree) and
computational geometry (Power diagram, Disk graph) to combinatorics
(De Bruijn) and abstract algebra (Cayley).

Some builders (De Bruijn, Cayley) generate their own topology
independent of point coordinates, similar to the lattice subpackage.
Others (KD-tree, Ball tree, Disk, Power diagram) operate directly
on the spatial layout.
"""

from graphgallery.misc.kdtree_neighbor import KDTreeNeighborGraph
from graphgallery.misc.balltree_neighbor import BallTreeNeighborGraph
from graphgallery.misc.disk import DiskGraph
from graphgallery.misc.intersection import IntersectionGraph

__all__ = [
    "KDTreeNeighborGraph",
    "BallTreeNeighborGraph",
    "DiskGraph",
    "IntersectionGraph",
]


def all_misc_builders():
    """Return instances of every miscellaneous builder with defaults."""
    return [
        KDTreeNeighborGraph(),
        BallTreeNeighborGraph(),
        DiskGraph(),
        IntersectionGraph(),
    ]

"""
Spanning Tree-Based Graph Construction Algorithms.

This subpackage contains 6 graph builders that construct trees or
tree-like structures connecting all (or most) nodes:

1.  Minimum Spanning Tree — Prim's algorithm
2.  Minimum Spanning Tree — Kruskal's algorithm
3.  Minimum Spanning Tree — Borůvka's algorithm
4.  Euclidean Minimum Spanning Tree (via Delaunay)
5.  Random Spanning Tree (Wilson's algorithm)
6.  k-MST Overlay (union of multiple diverse spanning trees)

A spanning tree of a graph G is a subgraph that is a tree which
includes all vertices of G, using exactly n-1 edges.

All three MST algorithms produce identical output (the MST is unique
when edge weights are distinct); they differ in strategy and
performance characteristics.
"""

from graphgallery.spanning.mst_prim import MSTPrimGraph
from graphgallery.spanning.mst_kruskal import MSTKruskalGraph
from graphgallery.spanning.mst_boruvka import MSTBoruvkaGraph
from graphgallery.spanning.emst import EuclideanMSTGraph
from graphgallery.spanning.random_spanning import RandomSpanningTreeGraph
from graphgallery.spanning.k_mst_overlay import KMSTOverlayGraph

__all__ = [
    "MSTPrimGraph",
    "MSTKruskalGraph",
    "MSTBoruvkaGraph",
    "EuclideanMSTGraph",
    "RandomSpanningTreeGraph",
    "KMSTOverlayGraph",
]


def all_spanning_builders():
    """Return instances of every spanning-tree builder with default params."""
    return [
        MSTPrimGraph(),
        MSTKruskalGraph(),
        MSTBoruvkaGraph(),
        EuclideanMSTGraph(),
        RandomSpanningTreeGraph(),
        KMSTOverlayGraph(),
    ]

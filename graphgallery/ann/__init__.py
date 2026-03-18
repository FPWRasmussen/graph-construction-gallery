"""
Approximate Nearest Neighbor (ANN) Graph Construction Algorithms.

This subpackage contains 6 graph builders designed for efficient
nearest neighbor search via graph traversal:

1.  Navigable Small World (NSW)
2.  Hierarchical Navigable Small World (HNSW)
3.  Vamana (DiskANN)
4.  NN-Descent
5.  Random Projection Forest Graph
6.  LSH-Based Graph (Locality-Sensitive Hashing)

Unlike exact proximity graphs, ANN graphs are constructed with
*search efficiency* as a primary goal.  They trade perfect accuracy
for dramatically faster query times, typically achieving high recall
(>95%) with sublinear search complexity.

Graph-based ANN methods are among the fastest known approaches for
high-recall nearest neighbor search, forming the backbone of modern
vector search engines and recommendation systems.

All builders accept a ``seed`` parameter for reproducibility since
they involve randomized construction.
"""

from graphgallery.ann.nsw import NSWGraph
from graphgallery.ann.hnsw import HNSWGraph
from graphgallery.ann.vamana import VamanaGraph
from graphgallery.ann.nn_descent import NNDescentGraph
from graphgallery.ann.rp_forest import RPForestGraph
from graphgallery.ann.lsh import LSHGraph

__all__ = [
    "NSWGraph",
    "HNSWGraph",
    "VamanaGraph",
    "NNDescentGraph",
    "RPForestGraph",
    "LSHGraph",
]


def all_ann_builders():
    """Return instances of every ANN graph builder with default params."""
    return [
        NSWGraph(),
        HNSWGraph(),
        VamanaGraph(),
        NNDescentGraph(),
        RPForestGraph(),
        LSHGraph(),
    ]

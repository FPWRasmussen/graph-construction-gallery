"""
KD-Tree Neighbor Graph.

A KD-tree (k-dimensional tree) is a space-partitioning data structure
for organizing points in k-dimensional space.  It recursively splits
the space along coordinate axes, enabling efficient nearest neighbor
queries in O(log n) average time (compared to O(n) brute force).

Construction:
    1. Build a KD-tree from the input points.
    2. For each point, query the tree for its k nearest neighbors.
    3. Create edges to these neighbors.

The resulting graph is identical to the brute-force k-NN graph
but is constructed much more efficiently — O(n log n) instead of
O(n²) for building the tree plus O(n · k · log n) for all queries.

KD-trees are most effective in low dimensions (d ≤ 20).  In higher
dimensions, the "curse of dimensionality" degrades performance
toward brute force.

This builder demonstrates how spatial data structures accelerate
graph construction — the graph topology is the same as k-NN, but
the algorithm is fundamentally different.

Reference:
    Bentley, J.L. (1975). "Multidimensional binary search trees used
    for associative searching." Communications of the ACM, 18(9).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class KDTreeNeighborGraph(GraphBuilder):
    """k-NN graph constructed efficiently via KD-tree queries.

    Produces the same graph as brute-force k-NN but uses O(n log n)
    construction and O(k log n) per query.

    Parameters:
        k: Number of nearest neighbors per node.
        leaf_size: KD-tree leaf size (affects build/query balance).
    """

    slug = "kdtree_neighbor"
    category = "misc"

    def __init__(self, k: int = 5, leaf_size: int = 10):
        self.k = k
        self.leaf_size = leaf_size

    @property
    def name(self) -> str:
        return "KD-Tree Neighbor Graph"

    @property
    def description(self) -> str:
        return (
            f"k-NN (k={self.k}) via KD-tree. Same result as brute-force "
            f"but O(n log n) construction."
        )

    @property
    def complexity(self) -> str:
        return "O(n log n) build + O(nk log n) queries"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("k", "Number of nearest neighbors", "int", 5, "1 ≤ k < n"),
            ParamInfo("leaf_size", "KD-tree leaf size", "int", 10, "≥ 1"),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if self.k >= layout.n_points:
            raise ValueError(
                f"k={self.k} must be < n_points={layout.n_points}"
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points

        # Build KD-tree
        tree = KDTree(points, leafsize=self.leaf_size)

        # Query k+1 neighbors (includes self)
        distances, indices = tree.query(points, k=self.k + 1)

        # Build graph (skip self-neighbor at index 0)
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j_pos in range(1, self.k + 1):
                j = int(indices[i, j_pos])
                d = float(distances[i, j_pos])
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=d)

        G.graph["algorithm"] = "kdtree_neighbor"
        G.graph["k"] = self.k
        G.graph["leaf_size"] = self.leaf_size

        return G

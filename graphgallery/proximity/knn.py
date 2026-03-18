"""
k-Nearest Neighbors (Directed) Graph.

Each node has exactly k outgoing edges pointing to its k closest
neighbors. The result is a directed graph (DiGraph) since the
relationship is asymmetric: i may be a neighbor of j without
j being a neighbor of i.

See also:
    - SymmetricKNNGraph: undirected union (edge if either direction)
    - MutualKNNGraph:    undirected intersection (edge if both directions)
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances, k_nearest_indices


class KNNGraph(GraphBuilder):
    """Directed graph where each node connects to its k nearest neighbors.

    Every node has out-degree exactly k. The in-degree varies: popular
    hub nodes may receive many incoming edges while peripheral nodes
    receive few.

    Parameters:
        k: Number of nearest neighbors per node.
    """

    slug = "knn"
    category = "proximity"

    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return "k-Nearest Neighbors"

    @property
    def description(self) -> str:
        return (
            "Directed graph — each node has k outgoing edges "
            "to its closest neighbors."
        )

    @property
    def is_directed(self) -> bool:
        return True

    @property
    def complexity(self) -> str:
        return "O(n²d) brute-force, O(nd log n) with KD-tree"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("k", "Number of nearest neighbors", "int", 5, "1 ≤ k < n"),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if self.k >= layout.n_points:
            raise ValueError(
                f"k={self.k} must be less than n_points={layout.n_points}"
            )

    def build(self, layout: PointLayout) -> nx.DiGraph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)
        neighbors = k_nearest_indices(dist, self.k)

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in neighbors[i]:
                G.add_edge(i, int(j), weight=float(dist[i, j]))

        return G

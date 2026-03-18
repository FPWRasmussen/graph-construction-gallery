"""
Symmetric k-Nearest Neighbors Graph (Union).

An undirected edge (i, j) exists if *either* i is among j's k nearest
neighbors *or* j is among i's k nearest neighbors. This is the union
of the forward and reverse directed k-NN graphs.

This is the most common interpretation of "k-NN graph" in machine
learning (e.g. spectral clustering, UMAP).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances, k_nearest_indices


class SymmetricKNNGraph(GraphBuilder):
    """Undirected k-NN graph via union of directed neighborhoods.

    Produces more edges than the mutual variant since only one
    direction of the neighbor relationship is required.

    Parameters:
        k: Number of nearest neighbors per node.
    """

    slug = "symmetric_knn"
    category = "proximity"

    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return "Symmetric k-NN"

    @property
    def description(self) -> str:
        return "Undirected edge if either node is in the other's k-NN (union)."

    @property
    def complexity(self) -> str:
        return "O(n²d)"

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

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)
        neighbors = k_nearest_indices(dist, self.k)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Union: add edge if i→j OR j→i exists in directed k-NN.
        # Iterating over all directed edges and adding to an undirected
        # graph automatically takes the union.
        for i in range(n):
            for j in neighbors[i]:
                j = int(j)
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=float(dist[i, j]))

        return G

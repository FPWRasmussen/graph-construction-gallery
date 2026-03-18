"""
Mutual k-Nearest Neighbors Graph (Intersection).

An undirected edge (i, j) exists only if *both* i is among j's
k nearest neighbors *and* j is among i's k nearest neighbors.
This is the intersection of the forward and reverse directed k-NN.

Mutual k-NN graphs are sparser and tend to better preserve cluster
structure, since inter-cluster edges are less likely to be reciprocal.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances, k_nearest_indices


class MutualKNNGraph(GraphBuilder):
    """Undirected k-NN graph via intersection of directed neighborhoods.

    Produces fewer edges than the symmetric variant, requiring the
    neighbor relationship to hold in both directions.

    Parameters:
        k: Number of nearest neighbors per node.
    """

    slug = "mutual_knn"
    category = "proximity"

    def __init__(self, k: int = 5):
        self.k = k

    @property
    def name(self) -> str:
        return "Mutual k-NN"

    @property
    def description(self) -> str:
        return "Undirected edge only if both nodes are in each other's k-NN."

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

        # Build neighbor sets for O(1) membership lookup
        neighbor_sets = [set(neighbors[i].tolist()) for i in range(n)]

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in neighbor_sets[i]:
                # Only add if mutual AND avoid duplicate (i < j)
                if j > i and i in neighbor_sets[j]:
                    G.add_edge(i, j, weight=float(dist[i, j]))

        return G

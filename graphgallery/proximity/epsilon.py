"""
ε-Neighborhood (Radius) Graph.

Connect all pairs of points whose Euclidean distance is at most ε.
Also known as a "radius graph" or "ball graph". The choice of ε
critically affects connectivity: too small yields many components,
too large approaches the complete graph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class EpsilonNeighborhoodGraph(GraphBuilder):
    """Connect all pairs of points within distance ε.

    Parameters:
        epsilon: Maximum distance threshold for edge creation.
    """

    slug = "epsilon"
    category = "proximity"

    def __init__(self, epsilon: float = 1.2):
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        return "ε-Neighborhood"

    @property
    def description(self) -> str:
        return f"Connect all pairs within Euclidean distance ε={self.epsilon}."

    @property
    def complexity(self) -> str:
        return "O(n²d)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "epsilon",
                "Distance threshold for edge creation",
                "float",
                1.2,
                "ε > 0",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= self.epsilon:
                    G.add_edge(i, j, weight=float(dist[i, j]))

        return G

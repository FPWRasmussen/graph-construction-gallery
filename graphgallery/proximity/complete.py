"""
Complete (Fully Connected) Graph.

Every node is connected to every other node, producing n(n-1)/2
undirected edges weighted by Euclidean distance.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class CompleteGraph(GraphBuilder):
    """Connect every pair of nodes, weighted by Euclidean distance.

    The complete graph K_n has n(n-1)/2 edges and is the densest
    possible simple undirected graph. It serves as the upper bound
    that all other proximity graphs are subgraphs of.

    Parameters:
        weighted: If True, edges carry a ``weight`` attribute equal
            to the Euclidean distance. If False, edges are unweighted.
    """

    slug = "complete"
    category = "proximity"

    def __init__(self, weighted: bool = True):
        self.weighted = weighted

    @property
    def name(self) -> str:
        return "Complete Graph"

    @property
    def description(self) -> str:
        return "Every node connected to every other node. O(n²) edges."

    @property
    def complexity(self) -> str:
        return "O(n²)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "weighted",
                "Attach Euclidean distance as edge weight",
                "bool",
                True,
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
                if self.weighted:
                    G.add_edge(i, j, weight=float(dist[i, j]))
                else:
                    G.add_edge(i, j)

        return G

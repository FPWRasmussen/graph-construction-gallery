"""
t-Spanner via Edge Filtering.

A simple, naïve approach to constructing a t-spanner: sort all
pairwise edges by weight and add each edge only if the current
shortest path between its endpoints exceeds t times the Euclidean
distance.

This is conceptually identical to the greedy spanner but implemented
with a different structure — here we iterate over ALL possible edges
(from shortest to longest) rather than using a more efficient
algorithmic framework.

For a point set in R^d, a t-spanner guarantees that for every pair
of points p, q, the shortest path distance in the spanner is at
most t·|pq|.

The stretch factor t ≥ 1 controls the trade-off: lower t gives
better approximation but more edges.

Reference:
    Althöfer, I., Das, G., Dobkin, D., Joseph, D., & Soares, J.
    (1993). "On sparse spanners of weighted graphs." Discrete &
    Computational Geometry, 9(1), 81–100.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.spanners._spanner_utils import (
    compute_stretch_factor,
    dijkstra_single_source,
)


class TSpannerGraph(GraphBuilder):
    """Sparse t-spanner via greedy edge filtering on all pairs.

    Adds edges shortest-first, skipping any edge whose endpoints
    are already within stretch t in the current graph.

    Parameters:
        t: Stretch factor (dilation bound).
    """

    slug = "t_spanner"
    category = "spanners"

    def __init__(self, t: float = 2.0):
        if t < 1.0:
            raise ValueError(f"Stretch factor t must be ≥ 1, got {t}")
        self.t = t

    @property
    def name(self) -> str:
        return "t-Spanner"

    @property
    def description(self) -> str:
        return (
            f"Sparse subgraph with stretch factor t={self.t}. "
            f"All-pairs greedy edge filtering."
        )

    @property
    def complexity(self) -> str:
        return "O(n³) due to shortest-path recomputation"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("t", "Stretch factor", "float", 2.0, "t ≥ 1"),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        # Sort all edges by weight (shortest first)
        edges: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((float(dist[i, j]), i, j))
        edges.sort(key=lambda e: e[0])

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Adjacency list for efficient Dijkstra
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}

        for weight, u, v in edges:
            # Check current shortest path between u and v
            sp_dist = dijkstra_single_source(adj, u, n)

            if sp_dist[v] > self.t * weight:
                # Add the edge — current path is too long
                G.add_edge(u, v, weight=weight)
                adj[u].append((v, weight))
                adj[v].append((u, weight))

        G.graph["algorithm"] = "t_spanner"
        G.graph["target_stretch"] = self.t
        G.graph["actual_stretch"] = compute_stretch_factor(G, dist)

        return G

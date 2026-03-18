"""
Greedy Geometric Spanner.

The greedy spanner is built by considering all pairwise edges in
order of increasing length and adding an edge only if the current
graph-distance between its endpoints exceeds t times the Euclidean
distance.

This is the same approach as the t-spanner module, but the "greedy
spanner" name specifically refers to this classical algorithm in the
computational geometry literature.  It is known to produce high-quality
spanners — often the sparsest possible for a given stretch factor.

The greedy algorithm produces a graph that is a subgraph of every
other t-spanner with the same stretch factor, making it optimal in
a strong structural sense.

Key properties:
    - O(n / t^d) edges for d-dimensional points (near-optimal)
    - Weight: O(wt(MST) · log n) in the worst case
    - Maximum degree: O(1) for fixed t > 1 and d = 2
    - Contains the MST as a subgraph
    - Produces the sparsest t-spanner among all "greedy-like" methods

Reference:
    Althöfer, I., Das, G., Dobkin, D., Joseph, D., & Soares, J.
    (1993). "On sparse spanners of weighted graphs." Discrete &
    Computational Geometry, 9(1), 81–100.

    Bose, P., Gudmundsson, J., & Smid, M. (2005). "Constructing
    plane spanners of bounded degree and low weight." Algorithmica,
    42, 249–264.
"""

from __future__ import annotations

import heapq

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.spanners._spanner_utils import (
    compute_stretch_factor,
    dijkstra_single_source,
)


class GreedySpannerGraph(GraphBuilder):
    """Greedy geometric spanner — edges added shortest-first.

    The canonical greedy algorithm for geometric spanners.  Produces
    near-optimal sparsity for any given stretch factor.

    Parameters:
        t: Stretch factor (dilation bound).  Must be > 1.
    """

    slug = "greedy_spanner"
    category = "spanners"

    def __init__(self, t: float = 2.0):
        if t < 1.0:
            raise ValueError(f"Stretch factor t must be ≥ 1, got {t}")
        self.t = t

    @property
    def name(self) -> str:
        return "Greedy Spanner"

    @property
    def description(self) -> str:
        return (
            f"Greedy geometric spanner (t={self.t}). "
            f"Near-optimal sparsity among all t-spanners."
        )

    @property
    def complexity(self) -> str:
        return "O(n² log n) for sorting + O(n²) Dijkstra checks"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("t", "Stretch factor", "float", 2.0, "t > 1"),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        # Collect and sort all edges
        edges: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((float(dist[i, j]), i, j))
        edges.sort(key=lambda e: e[0])

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Adjacency list for fast Dijkstra
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}

        n_considered = 0
        n_added = 0

        for weight, u, v in edges:
            n_considered += 1

            # Check if current shortest path ≤ t · Euclidean distance
            sp_dist = dijkstra_single_source(adj, u, n)

            if sp_dist[v] > self.t * weight:
                G.add_edge(u, v, weight=weight)
                adj[u].append((v, weight))
                adj[v].append((u, weight))
                n_added += 1

        G.graph["algorithm"] = "greedy_spanner"
        G.graph["target_stretch"] = self.t
        G.graph["edges_considered"] = n_considered
        G.graph["edges_added"] = n_added
        G.graph["actual_stretch"] = compute_stretch_factor(G, dist)

        return G

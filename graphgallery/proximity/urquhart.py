"""
Urquhart Graph.

Constructed by removing the longest edge from every triangle in the
Delaunay triangulation. An edge is deleted if it is the longest edge
in *at least one* of its incident triangles.

The result is a good approximation of the Relative Neighborhood Graph
and is much cheaper to compute (O(n log n) via Delaunay).

Reference:
    Urquhart, R.B. (1980). "Algorithms for Computation of Relative
    Neighbourhood Graph." Electronics Letters, 16(14).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class UrquhartGraph(GraphBuilder):
    """Delaunay triangulation with the longest edge of each triangle removed.

    The Urquhart graph is a fast approximation of the Relative
    Neighborhood Graph. It always contains the MST.
    """

    slug = "urquhart"
    category = "proximity"

    @property
    def name(self) -> str:
        return "Urquhart Graph"

    @property
    def description(self) -> str:
        return (
            "Delaunay triangulation minus the longest edge of each triangle. "
            "Approximates the RNG."
        )

    @property
    def complexity(self) -> str:
        return "O(n log n)"

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if layout.n_points < 3:
            raise ValueError(
                "Urquhart graph requires at least 3 non-collinear points."
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        # Step 1: Compute the Delaunay triangulation
        tri = Delaunay(layout.points)

        # Step 2: Collect all Delaunay edges
        all_edges: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    u, v = int(simplex[a]), int(simplex[b])
                    all_edges.add((min(u, v), max(u, v)))

        # Step 3: Find the longest edge in each triangle and flag it
        longest_edges: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
            edges = [
                (min(i, j), max(i, j)),
                (min(i, k), max(i, k)),
                (min(j, k), max(j, k)),
            ]
            lengths = [dist[e[0], e[1]] for e in edges]
            longest_idx = int(np.argmax(lengths))
            longest_edges.add(edges[longest_idx])

        # Step 4: Build the graph excluding longest edges
        surviving_edges = all_edges - longest_edges

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v in surviving_edges:
            G.add_edge(u, v, weight=float(dist[u, v]))

        return G

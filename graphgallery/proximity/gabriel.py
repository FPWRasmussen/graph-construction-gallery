"""
Gabriel Graph.

Two points p_i and p_j are connected by an edge if and only if the
closed disk with diameter segment p_i–p_j contains no other point.
Equivalently, no point p_k satisfies:

    ‖p_k − midpoint(p_i, p_j)‖ < ‖p_i − p_j‖ / 2

The Gabriel graph is a subgraph of the Delaunay triangulation and
a supergraph of the Relative Neighborhood Graph. It is also the
β-skeleton with β = 1.

Reference:
    Gabriel, K.R. & Sokal, R.R. (1969). "A New Statistical Approach
    to Geographic Variation Analysis." Systematic Zoology, 18(3).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class GabrielGraph(GraphBuilder):
    """Connect points whose diametral disk contains no other point.

    The Gabriel graph is equivalent to the β-skeleton with β = 1.
    It sits between the Delaunay triangulation (superset) and the
    Relative Neighborhood Graph (subset) in the proximity graph
    hierarchy.
    """

    slug = "gabriel"
    category = "proximity"

    @property
    def name(self) -> str:
        return "Gabriel Graph"

    @property
    def description(self) -> str:
        return (
            "Connect if no other point lies inside the diametral disk. "
            "Equivalent to β-skeleton with β=1."
        )

    @property
    def complexity(self) -> str:
        return "O(n³) naïve, O(n log n) via Delaunay filtering"

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                d_ij = dist[i, j]
                midpoint = (points[i] + points[j]) / 2.0
                radius = d_ij / 2.0

                # Check if any other point lies strictly inside the
                # diametral disk (open disk test)
                is_gabriel = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    dist_to_mid = np.linalg.norm(points[k] - midpoint)
                    if dist_to_mid < radius:
                        is_gabriel = False
                        break

                if is_gabriel:
                    G.add_edge(i, j, weight=float(d_ij))

        return G

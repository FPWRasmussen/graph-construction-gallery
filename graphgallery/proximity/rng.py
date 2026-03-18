"""
Relative Neighborhood Graph (RNG).

Two points p_i and p_j are connected if no third point p_k is closer
to *both* of them than they are to each other. Formally, the edge
(i, j) exists iff for all k ≠ i, j:

    max(d(i, k), d(j, k)) ≥ d(i, j)

The "lune" of p_i and p_j is the intersection of the open balls
B(p_i, d(i,j)) and B(p_j, d(i,j)). The edge exists iff the lune
is empty of other points.

The RNG is a subgraph of the Gabriel graph and a supergraph of
the Minimum Spanning Tree. It is also the β-skeleton with β = 2.

Reference:
    Toussaint, G.T. (1980). "The Relative Neighbourhood Graph of a
    Finite Planar Set." Pattern Recognition, 12(4).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class RelativeNeighborhoodGraph(GraphBuilder):
    """Connect points whose lune contains no other point.

    The RNG is equivalent to the β-skeleton with β = 2. It is sparser
    than the Gabriel graph but always contains the Minimum Spanning Tree.
    """

    slug = "rng"
    category = "proximity"

    @property
    def name(self) -> str:
        return "Relative Neighborhood Graph"

    @property
    def description(self) -> str:
        return (
            "Connect if no third point is closer to both endpoints. "
            "Equivalent to β-skeleton with β=2."
        )

    @property
    def complexity(self) -> str:
        return "O(n³)"

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                d_ij = dist[i, j]

                # Check the lune: intersection of B(i, d_ij) ∩ B(j, d_ij).
                # Point k is in the lune iff max(d(i,k), d(j,k)) < d_ij.
                lune_empty = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if max(dist[i, k], dist[j, k]) < d_ij:
                        lune_empty = False
                        break

                if lune_empty:
                    G.add_edge(i, j, weight=float(d_ij))

        return G

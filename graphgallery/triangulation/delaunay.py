"""
Delaunay Triangulation Graph.

The Delaunay triangulation of a point set P in 2D is the triangulation
DT(P) such that no point in P lies inside the circumcircle of any
triangle.  Equivalently, it maximizes the minimum angle across all
possible triangulations, avoiding "sliver" triangles.

The Delaunay triangulation is the dual of the Voronoi diagram: two
points share a Delaunay edge iff their Voronoi cells are adjacent.

Key properties:
    - Contains the Euclidean Minimum Spanning Tree as a subgraph
    - Contains the Gabriel graph as a subgraph
    - Contains the Relative Neighborhood Graph as a subgraph
    - Has at most 3n − 6 edges (for n ≥ 3)
    - Can be computed in O(n log n)

Reference:
    Delaunay, B. (1934). "Sur la sphère vide." Bulletin de l'Académie
    des Sciences de l'URSS, Classe des Sciences Mathématiques et
    Naturelles, 6, 793–800.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


def delaunay_edges(points: np.ndarray) -> set[tuple[int, int]]:
    """Extract unique undirected edges from a Delaunay triangulation.

    Args:
        points: (n, 2) array of point coordinates.

    Returns:
        Set of (i, j) tuples with i < j.
    """
    tri = Delaunay(points)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                u, v = int(simplex[a]), int(simplex[b])
                edges.add((min(u, v), max(u, v)))
    return edges


class DelaunayGraph(GraphBuilder):
    """Delaunay triangulation of a 2D point set.

    Maximizes the minimum angle over all triangulations.  The resulting
    graph is planar and has O(n) edges.
    """

    slug = "delaunay"
    category = "triangulation"

    @property
    def name(self) -> str:
        return "Delaunay Triangulation"

    @property
    def description(self) -> str:
        return (
            "Triangulation maximizing the minimum angle. "
            "Dual of the Voronoi diagram."
        )

    @property
    def complexity(self) -> str:
        return "O(n log n)"

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if layout.n_points < 3:
            raise ValueError(
                "Delaunay triangulation requires at least 3 "
                "non-collinear points."
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)
        edges = delaunay_edges(layout.points)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v in edges:
            G.add_edge(u, v, weight=float(dist[u, v]))

        return G

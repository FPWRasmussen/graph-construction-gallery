"""
Voronoi Dual Graph.

The Voronoi diagram partitions the plane into cells, one per input
point, where each cell V(p_i) contains all locations closer to p_i
than to any other point.  The **Voronoi dual graph** (also called the
"Voronoi adjacency graph") connects two input points iff their Voronoi
cells share an edge.

This graph is equivalent to the Delaunay triangulation for points in
general position (no four co-circular points).  However, this
implementation constructs it *from the Voronoi diagram* rather than
the Delaunay triangulation, and can optionally include Voronoi vertex
and ridge metadata for visualization.

Reference:
    Voronoi, G. (1908). "Nouvelles applications des paramètres continus
    à la théorie des formes quadratiques." Journal für die Reine und
    Angewandte Mathematik, 134, 198–287.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import Voronoi

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class VoronoiDualGraph(GraphBuilder):
    """Connect points whose Voronoi cells are adjacent.

    The resulting graph is the dual of the Voronoi diagram, which
    (for points in general position) is equivalent to the Delaunay
    triangulation.

    Parameters:
        store_voronoi: If True, store the full ``scipy.spatial.Voronoi``
            object in ``G.graph["voronoi"]`` for downstream visualization
            of cell boundaries.
        finite_only: If True, only include edges whose shared Voronoi
            ridge is finite (both vertices are finite).  This excludes
            edges on the convex hull boundary.
    """

    slug = "voronoi_dual"
    category = "triangulation"

    def __init__(
        self,
        store_voronoi: bool = True,
        finite_only: bool = False,
    ):
        self.store_voronoi = store_voronoi
        self.finite_only = finite_only

    @property
    def name(self) -> str:
        return "Voronoi Dual Graph"

    @property
    def description(self) -> str:
        return "Connect points whose Voronoi cells share a boundary edge."

    @property
    def complexity(self) -> str:
        return "O(n log n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "store_voronoi",
                "Store the scipy Voronoi object on the graph",
                "bool",
                True,
            ),
            ParamInfo(
                "finite_only",
                "Only include edges with finite Voronoi ridges",
                "bool",
                False,
            ),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if layout.n_points < 3:
            raise ValueError(
                "Voronoi dual graph requires at least 3 points."
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        vor = Voronoi(layout.points)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # ``vor.ridge_points`` is (m, 2): pairs of input points that
        # share a Voronoi ridge.
        # ``vor.ridge_vertices`` is (m, 2): pairs of Voronoi vertex
        # indices for each ridge (-1 indicates a point at infinity).
        for ridge_idx, (u, v) in enumerate(vor.ridge_points):
            u, v = int(u), int(v)

            if self.finite_only:
                ridge_verts = vor.ridge_vertices[ridge_idx]
                if -1 in ridge_verts:
                    continue

            # Store the Voronoi ridge vertices on the edge
            ridge_v = tuple(int(x) for x in vor.ridge_vertices[ridge_idx])

            G.add_edge(
                u, v,
                weight=float(dist[u, v]),
                ridge_vertices=ridge_v,
            )

        # Store Voronoi diagram for visualization
        if self.store_voronoi:
            G.graph["voronoi"] = vor

        return G

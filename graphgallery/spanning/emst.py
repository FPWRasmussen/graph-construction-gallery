"""
Euclidean Minimum Spanning Tree (EMST).

The Euclidean MST of a set of points in R^d is the MST of the
complete graph with Euclidean distance weights.  Computing it via
the complete graph takes O(n² log n).

A classical result in computational geometry states that the EMST
is always a subgraph of the Delaunay triangulation.  This means we
can compute the EMST efficiently:

    1. Compute the Delaunay triangulation in O(n log n).
    2. Run any MST algorithm on the O(n) Delaunay edges.
    3. Total: O(n log n) — much better than O(n²) on the complete graph.

This is the recommended approach for spatial point sets in 2D.

Reference:
    Preparata, F.P. & Shamos, M.I. (1985). "Computational Geometry:
    An Introduction." Springer-Verlag. Chapter 6.
"""

from __future__ import annotations

import heapq

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class EuclideanMSTGraph(GraphBuilder):
    """Euclidean MST computed efficiently via Delaunay triangulation.

    Instead of considering all O(n²) pairwise edges, this builder
    first computes the Delaunay triangulation (O(n log n)) and then
    finds the MST of the O(n) Delaunay edges using Prim's algorithm.

    This exploits the fundamental property that the EMST is always
    a subgraph of the Delaunay triangulation.
    """

    slug = "emst"
    category = "spanning"

    @property
    def name(self) -> str:
        return "Euclidean MST"

    @property
    def description(self) -> str:
        return (
            "Minimum spanning tree via Delaunay triangulation. "
            "O(n log n) instead of O(n²)."
        )

    @property
    def complexity(self) -> str:
        return "O(n log n)"

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if layout.n_points < 3:
            raise ValueError("EMST via Delaunay requires at least 3 points.")

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points

        # Step 1: Delaunay triangulation
        tri = Delaunay(points)

        # Step 2: Build adjacency from Delaunay edges
        #   adj[i] = list of (weight, neighbor_index)
        adj: list[list[tuple[float, int]]] = [[] for _ in range(n)]
        seen_edges: set[tuple[int, int]] = set()

        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    u, v = int(simplex[a]), int(simplex[b])
                    edge_key = (min(u, v), max(u, v))
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                    w = float(np.linalg.norm(points[u] - points[v]))
                    adj[u].append((w, v))
                    adj[v].append((w, u))

        # Step 3: Prim's MST on the sparse Delaunay adjacency
        in_tree = np.zeros(n, dtype=bool)
        mst_edges: list[tuple[int, int, float]] = []

        # Start from vertex 0
        in_tree[0] = True
        heap: list[tuple[float, int, int]] = []
        for w, v in adj[0]:
            heapq.heappush(heap, (w, v, 0))

        while heap and len(mst_edges) < n - 1:
            weight, v, u = heapq.heappop(heap)
            if in_tree[v]:
                continue

            in_tree[v] = True
            mst_edges.append((u, v, weight))

            for w_next, nbr in adj[v]:
                if not in_tree[nbr]:
                    heapq.heappush(heap, (w_next, nbr, v))

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, w in mst_edges:
            G.add_edge(u, v, weight=w)

        G.graph["algorithm"] = "emst_via_delaunay"
        G.graph["total_weight"] = sum(w for _, _, w in mst_edges)
        G.graph["n_delaunay_edges"] = len(seen_edges)

        return G

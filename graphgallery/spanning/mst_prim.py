"""
Minimum Spanning Tree — Prim's Algorithm.

Prim's algorithm grows the MST from a single starting vertex by
repeatedly adding the cheapest edge that connects a vertex already
in the tree to one that is not.

Strategy:  vertex-centric, greedy, uses a priority queue.

    1. Start with an arbitrary vertex s in the tree set T.
    2. While T does not span all vertices:
       a. Find the minimum-weight edge (u, v) with u ∈ T, v ∉ T.
       b. Add v to T and edge (u, v) to the MST.

Time complexity:
    - O(n²) with an adjacency matrix (used here for simplicity)
    - O((n + m) log n) with a binary heap and adjacency list
    - O(m + n log n) with a Fibonacci heap

Prim's is well-suited for dense graphs where m ≈ n², since the
adjacency-matrix version avoids heap overhead.

Reference:
    Prim, R.C. (1957). "Shortest connection networks and some
    generalizations." Bell System Technical Journal, 36(6).
"""

from __future__ import annotations

import heapq

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class MSTPrimGraph(GraphBuilder):
    """Minimum Spanning Tree via Prim's vertex-growing algorithm.

    Grows the tree from a start vertex, always attaching the nearest
    non-tree vertex.  Uses a binary min-heap for efficiency.

    Parameters:
        start_vertex: Index of the starting vertex (default 0).
    """

    slug = "mst_prim"
    category = "spanning"

    def __init__(self, start_vertex: int = 0):
        self.start_vertex = start_vertex

    @property
    def name(self) -> str:
        return "MST (Prim's)"

    @property
    def description(self) -> str:
        return (
            "Minimum spanning tree grown from a start vertex. "
            "Greedy vertex-centric approach with a priority queue."
        )

    @property
    def complexity(self) -> str:
        return "O(n²) with adjacency matrix, O((n+m) log n) with heap"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "start_vertex",
                "Index of the starting vertex",
                "int",
                0,
                "0 ≤ start < n",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        start = self.start_vertex % n  # Wrap around safely

        # --- Prim's with binary heap ---
        in_tree = np.zeros(n, dtype=bool)
        mst_edges: list[tuple[int, int, float]] = []

        # Heap entries: (weight, target_vertex, source_vertex)
        heap: list[tuple[float, int, int]] = []
        in_tree[start] = True

        # Push all edges from the start vertex
        for j in range(n):
            if j != start:
                heapq.heappush(heap, (float(dist[start, j]), j, start))

        while heap and len(mst_edges) < n - 1:
            weight, v, u = heapq.heappop(heap)

            if in_tree[v]:
                continue  # Already absorbed

            # Add vertex v and edge (u, v) to the tree
            in_tree[v] = True
            mst_edges.append((u, v, weight))

            # Push edges from v to all non-tree vertices
            for w in range(n):
                if not in_tree[w]:
                    heapq.heappush(heap, (float(dist[v, w]), w, v))

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, w in mst_edges:
            G.add_edge(u, v, weight=w)

        G.graph["algorithm"] = "prim"
        G.graph["total_weight"] = sum(w for _, _, w in mst_edges)

        return G

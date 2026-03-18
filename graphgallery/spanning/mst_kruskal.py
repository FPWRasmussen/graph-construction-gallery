"""
Minimum Spanning Tree — Kruskal's Algorithm.

Kruskal's algorithm builds the MST by sorting all edges by weight
and greedily adding the cheapest edge that does not create a cycle,
using a Union-Find (disjoint set) data structure.

Strategy:  edge-centric, greedy, uses Union-Find.

    1. Sort all edges by weight.
    2. For each edge (u, v) in sorted order:
       a. If u and v are in different components, add the edge
          and merge the components.
       b. Stop when n-1 edges have been added.

Time complexity:
    - O(m log m) for sorting edges  (m = n(n-1)/2 for complete graph)
    - O(m α(n)) for union-find operations (nearly linear)
    - Total: O(n² log n) for the complete Euclidean graph

Kruskal's is well-suited for sparse graphs.  For the complete graph
used here, the O(n² log n) sort dominates.

Reference:
    Kruskal, J.B. (1956). "On the shortest spanning subtree of a
    graph and the traveling salesman problem." Proceedings of the
    American Mathematical Society, 7(1).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class _UnionFind:
    """Weighted Union-Find with path compression.

    Supports near-constant-time ``find`` and ``union`` operations
    via union-by-rank and path compression.
    """

    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the root representative with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path halving
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Merge the sets containing x and y.

        Returns True if they were in different sets (merge happened),
        False if they were already in the same set.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same component."""
        return self.find(x) == self.find(y)


class MSTKruskalGraph(GraphBuilder):
    """Minimum Spanning Tree via Kruskal's edge-sorting algorithm.

    Sorts all pairwise edges by Euclidean distance and greedily adds
    the shortest edge that connects two different components.
    """

    slug = "mst_kruskal"
    category = "spanning"

    @property
    def name(self) -> str:
        return "MST (Kruskal's)"

    @property
    def description(self) -> str:
        return (
            "Minimum spanning tree via sorted edge insertion with "
            "Union-Find cycle detection."
        )

    @property
    def complexity(self) -> str:
        return "O(n² log n) for complete graph, O(m log m) in general"

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        # Step 1: Collect all edges from the upper triangle
        edges: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((float(dist[i, j]), i, j))

        # Step 2: Sort by weight
        edges.sort(key=lambda e: e[0])

        # Step 3: Greedily add edges using Union-Find
        uf = _UnionFind(n)
        mst_edges: list[tuple[int, int, float]] = []

        for weight, u, v in edges:
            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                if len(mst_edges) == n - 1:
                    break

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, w in mst_edges:
            G.add_edge(u, v, weight=w)

        G.graph["algorithm"] = "kruskal"
        G.graph["total_weight"] = sum(w for _, _, w in mst_edges)

        return G

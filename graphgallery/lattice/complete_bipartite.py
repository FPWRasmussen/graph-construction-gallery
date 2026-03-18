"""
Complete Bipartite Graph K_{a,b}.

A complete bipartite graph has two disjoint sets A and B where
every node in A is connected to every node in B, and there are
no edges within A or within B.

Properties:
    - Nodes: |A| + |B| = a + b
    - Edges: a × b
    - Regular iff a = b
    - Planar iff min(a, b) ≤ 2
    - Diameter: 2 (if both sets non-empty)
    - K_{3,3} is the smallest non-planar complete bipartite graph

For the gallery, we use the layout's cluster sizes (10 and 20)
as the two partitions by default.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.lattice._layout_utils import make_structural_layout


class CompleteBipartiteGraph(GraphBuilder):
    """Every node in set A connected to every node in set B.

    Parameters:
        a: Size of partition A (None = first cluster size).
        b: Size of partition B (None = second cluster size).
    """

    slug = "complete_bipartite"
    category = "lattice"

    def __init__(self, a: int | None = None, b: int | None = None):
        self.a = a
        self.b = b

    @property
    def name(self) -> str:
        return "Complete Bipartite"

    @property
    def description(self) -> str:
        return "Two sets: every node in A connects to every node in B."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(a × b)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("a", "Size of partition A", "int | None", None, "≥ 1"),
            ParamInfo("b", "Size of partition B", "int | None", None, "≥ 1"),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        a, b = self._resolve_sizes(layout)
        n = a + b

        # Layout: two horizontal rows
        positions = np.zeros((n, 2), dtype=np.float64)

        # Partition A: top row
        a_spacing = max(1.0, (b - 1) * 0.5 / max(a - 1, 1))
        for i in range(a):
            positions[i] = [
                (i - (a - 1) / 2.0) * a_spacing,
                1.5,
            ]

        # Partition B: bottom row
        b_spacing = 0.5
        for i in range(b):
            positions[a + i] = [
                (i - (b - 1) / 2.0) * b_spacing,
                -1.5,
            ]

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(a):
            for j in range(a, n):
                G.add_edge(i, j)

        # Node metadata
        for i in range(a):
            G.nodes[i]["partition"] = "A"
        for i in range(a, n):
            G.nodes[i]["partition"] = "B"

        G.graph["positions"] = positions
        G.graph["a"] = a
        G.graph["b"] = b
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

    def _resolve_sizes(self, layout: PointLayout) -> tuple[int, int]:
        """Determine partition sizes."""
        if self.a is not None and self.b is not None:
            return self.a, self.b

        if layout.n_clusters >= 2:
            sizes = [
                int((layout.labels == c).sum())
                for c in range(layout.n_clusters)
            ]
            sizes.sort()
            return sizes[0], sizes[1]

        # Default split
        n = layout.n_points
        return n // 3, n - n // 3

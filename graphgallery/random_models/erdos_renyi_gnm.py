"""
Erdős–Rényi G(n, m) Random Graph.

The G(n, m) model selects exactly m edges uniformly at random from
the n(n-1)/2 possible undirected edges.  Unlike G(n, p), the number
of edges is deterministic.

The two Erdős–Rényi models are asymptotically equivalent when
m ≈ p · n(n-1)/2, but they differ in finite-size behavior.

Reference:
    Erdős, P. & Rényi, A. (1959). "On random graphs I."
    Publicationes Mathematicae, 6, 290–297.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class ErdosRenyiGnmGraph(GraphBuilder):
    """Exactly m edges chosen uniformly at random.

    Parameters:
        m: Exact number of edges.
        seed: Random seed.
    """

    slug = "erdos_renyi_gnm"
    category = "random_models"

    def __init__(self, m: int = 60, seed: int | None = None):
        self.m = m
        self.seed = seed

    @property
    def name(self) -> str:
        return "Erdős–Rényi G(n, m)"

    @property
    def description(self) -> str:
        return f"Exactly {self.m} edges chosen uniformly at random."

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(m) with rejection sampling"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("m", "Exact number of edges", "int", 60, "0 ≤ m ≤ n(n-1)/2"),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        max_edges = n * (n - 1) // 2
        m = min(self.m, max_edges)

        seed = self.seed if self.seed is not None else (layout.seed + 101)
        rng = np.random.default_rng(seed)

        # Enumerate all possible edges and sample m of them
        all_edges: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                all_edges.append((i, j))

        chosen_indices = rng.choice(len(all_edges), size=m, replace=False)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for idx in chosen_indices:
            u, v = all_edges[idx]
            G.add_edge(u, v)

        G.graph["model"] = "G(n,m)"
        G.graph["m"] = m

        return G

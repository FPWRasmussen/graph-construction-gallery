"""
Erdős–Rényi G(n, p) Random Graph.

The G(n, p) model includes each of the n(n-1)/2 possible undirected
edges independently with probability p.  This is the simplest and
most studied random graph model.

Key properties:
    - Expected number of edges: p · n(n-1)/2
    - Expected degree: p · (n-1)
    - Degree distribution: Binomial(n-1, p) → Poisson for large n
    - Phase transition at p = 1/n: giant component emerges
    - Connectivity threshold: p ≈ ln(n)/n

Reference:
    Erdős, P. & Rényi, A. (1959). "On random graphs I."
    Publicationes Mathematicae, 6, 290–297.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class ErdosRenyiGnpGraph(GraphBuilder):
    """Each edge included independently with probability p.

    Parameters:
        p: Edge inclusion probability.
        seed: Random seed.
    """

    slug = "erdos_renyi_gnp"
    category = "random_models"

    def __init__(self, p: float = 0.15, seed: int | None = None):
        self.p = p
        self.seed = seed

    @property
    def name(self) -> str:
        return "Erdős–Rényi G(n, p)"

    @property
    def description(self) -> str:
        return f"Each edge included independently with probability p={self.p}."

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n²)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("p", "Edge probability", "float", 0.15, "0 ≤ p ≤ 1"),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        seed = self.seed if self.seed is not None else (layout.seed + 100)
        rng = np.random.default_rng(seed)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < self.p:
                    G.add_edge(i, j)

        G.graph["model"] = "G(n,p)"
        G.graph["p"] = self.p

        return G

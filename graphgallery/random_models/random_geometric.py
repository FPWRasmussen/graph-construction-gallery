"""
Random Geometric Graph (RGG).

A random geometric graph connects nodes that are within a fixed
radius r in a metric space.  Unlike the ε-neighborhood graph
(which operates on a fixed point set), the classic RGG also
randomizes the point positions uniformly in a domain.

Here we use the canonical layout positions directly (they are
already random), making this equivalent to the ε-neighborhood
graph but categorized under random models to emphasize the
generative interpretation.

Key properties:
    - Expected degree ≈ ρ · π · r² where ρ = n / area
    - Exhibits spatial clustering (high clustering coefficient)
    - Phase transition for connectivity at r = √(ln(n) / (π n))
    - Often used to model wireless / sensor networks

Reference:
    Penrose, M.D. (2003). "Random Geometric Graphs."
    Oxford University Press.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class RandomGeometricGraph(GraphBuilder):
    """Connect nodes within radius r using layout positions.

    Parameters:
        r: Connection radius.
        seed: Random seed (used if generating fresh positions).
    """

    slug = "random_geometric"
    category = "random_models"

    def __init__(self, r: float = 1.0, seed: int | None = None):
        self.r = r
        self.seed = seed

    @property
    def name(self) -> str:
        return "Random Geometric"

    @property
    def description(self) -> str:
        return f"Connect nodes within Euclidean distance r={self.r}."

    @property
    def is_deterministic(self) -> bool:
        return True  # Deterministic given fixed positions

    @property
    def is_spatial(self) -> bool:
        return True

    @property
    def complexity(self) -> str:
        return "O(n²)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("r", "Connection radius", "float", 1.0, "r > 0"),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= self.r:
                    G.add_edge(i, j, weight=float(dist[i, j]))

        G.graph["model"] = "random_geometric"
        G.graph["r"] = self.r

        return G

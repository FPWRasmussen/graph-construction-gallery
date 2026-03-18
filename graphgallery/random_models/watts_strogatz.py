"""
Watts–Strogatz Small-World Graph.

The Watts–Strogatz model generates graphs with high clustering and
short average path length ("small-world" property):

    1. Start with a ring lattice of n nodes, each connected to its
       k nearest neighbors on the ring (k/2 on each side).
    2. For each edge, with probability p, rewire it to a uniformly
       random target (avoiding self-loops and duplicates).

    p = 0:  Perfect ring lattice (high clustering, long paths)
    p = 1:  Fully random graph (low clustering, short paths)
    p ≈ 0.1–0.3: Small-world regime (high clustering AND short paths)

Reference:
    Watts, D.J. & Strogatz, S.H. (1998). "Collective dynamics of
    'small-world' networks." Nature, 393(6684), 440–442.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class WattsStrogatzGraph(GraphBuilder):
    """Ring lattice with random rewiring → small-world properties.

    Parameters:
        k: Each node connects to k nearest ring neighbors (must be even).
        p: Rewiring probability for each edge.
        seed: Random seed.
    """

    slug = "watts_strogatz"
    category = "random_models"

    def __init__(self, k: int = 4, p: float = 0.3, seed: int | None = None):
        if k % 2 != 0:
            raise ValueError(f"k must be even, got {k}")
        self.k = k
        self.p = p
        self.seed = seed

    @property
    def name(self) -> str:
        return "Watts–Strogatz"

    @property
    def description(self) -> str:
        return (
            f"Small-world model: ring lattice (k={self.k}) with "
            f"p={self.p} rewiring probability."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(nk)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("k", "Ring neighbors (must be even)", "int", 4, "2 ≤ k < n, even"),
            ParamInfo("p", "Rewiring probability", "float", 0.3, "0 ≤ p ≤ 1"),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        k = min(self.k, n - 1)
        if k % 2 != 0:
            k -= 1
        half_k = k // 2

        seed = self.seed if self.seed is not None else (layout.seed + 102)
        rng = np.random.default_rng(seed)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Step 1: Build ring lattice
        for i in range(n):
            for offset in range(1, half_k + 1):
                j = (i + offset) % n
                G.add_edge(i, j)

        # Step 2: Rewire edges
        # For each node i, consider each edge to i's "right" neighbors
        # and rewire with probability p
        for i in range(n):
            for offset in range(1, half_k + 1):
                j = (i + offset) % n

                if rng.random() < self.p:
                    # Choose a random target (not self, not existing neighbor)
                    G.remove_edge(i, j)

                    candidates = [
                        v for v in range(n)
                        if v != i and not G.has_edge(i, v)
                    ]
                    if candidates:
                        new_target = int(rng.choice(candidates))
                        G.add_edge(i, new_target)
                    else:
                        # Restore if no valid target
                        G.add_edge(i, j)

        G.graph["model"] = "watts_strogatz"
        G.graph["k"] = k
        G.graph["p"] = self.p

        return G

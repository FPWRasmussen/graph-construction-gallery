"""
Barabási–Albert Preferential Attachment Graph.

The BA model generates scale-free networks with power-law degree
distributions by growing the graph one node at a time:

    1. Start with a small seed graph of m₀ connected nodes.
    2. At each step, add one new node with m ≤ m₀ edges.
    3. Each new edge connects to an existing node v with probability
       proportional to v's current degree: P(v) = deg(v) / Σ deg(u).

This "rich get richer" mechanism produces networks where a few hub
nodes accumulate many connections while most nodes have few —
yielding a power-law degree distribution P(k) ~ k^(-3).

Reference:
    Barabási, A.-L. & Albert, R. (1999). "Emergence of scaling in
    random networks." Science, 286(5439), 509–512.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class BarabasiAlbertGraph(GraphBuilder):
    """Scale-free graph via preferential attachment.

    Parameters:
        m: Number of edges each new node brings.
        seed: Random seed.
    """

    slug = "barabasi_albert"
    category = "random_models"

    def __init__(self, m: int = 2, seed: int | None = None):
        self.m = m
        self.seed = seed

    @property
    def name(self) -> str:
        return "Barabási–Albert"

    @property
    def description(self) -> str:
        return (
            f"Preferential attachment: each new node adds m={self.m} edges. "
            f"Produces power-law degree distribution."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(nm)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("m", "Edges per new node", "int", 2, "1 ≤ m < n"),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        m = min(self.m, n - 1)

        seed = self.seed if self.seed is not None else (layout.seed + 103)
        rng = np.random.default_rng(seed)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Seed graph: complete graph on first m+1 nodes
        m0 = m + 1
        for i in range(m0):
            for j in range(i + 1, m0):
                G.add_edge(i, j)

        # Repeated targets list for O(1) preferential attachment sampling.
        # Each node appears once per edge endpoint it has.
        targets: list[int] = []
        for i in range(m0):
            for j in range(i + 1, m0):
                targets.extend([i, j])

        # Grow the graph
        for new_node in range(m0, n):
            # Select m distinct targets with probability ∝ degree
            chosen: set[int] = set()
            attempts = 0
            max_attempts = m * 20  # Safety bound

            while len(chosen) < m and attempts < max_attempts:
                target = targets[int(rng.integers(len(targets)))]
                if target != new_node:
                    chosen.add(target)
                attempts += 1

            # Add edges
            for target in chosen:
                G.add_edge(new_node, target)
                targets.append(new_node)
                targets.append(target)

        G.graph["model"] = "barabasi_albert"
        G.graph["m"] = m

        return G

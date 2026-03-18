"""
Forest Fire Model.

The forest fire model generates graphs with heavy-tailed degree
distributions, densification, and community structure via a
"burning" metaphor:

    1. A new node v arrives and connects to a random existing
       "ambassador" node w.
    2. v "burns" through w's neighbors: each neighbor is followed
       (edge created to v) independently with forward probability p.
    3. The fire spreads recursively from each burned neighbor,
       with geometrically decaying reach.
    4. A backward burning probability (optional) also follows
       in-edges in directed graphs.

The model is one of the few that naturally produces densification
(edges growing superlinearly in n) and shrinking diameter.

Reference:
    Leskovec, J., Kleinberg, J., & Faloutsos, C. (2007). "Graph
    evolution: Densification and shrinking diameters." ACM Trans.
    Knowledge Discovery from Data, 1(1), Article 2.
"""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class ForestFireGraph(GraphBuilder):
    """Graph grown by recursive "burning" through neighbors.

    Parameters:
        p_forward: Probability of burning each forward neighbor.
        p_backward: Probability of burning each backward neighbor
            (only relevant for directed; set to 0 for undirected feel).
        seed: Random seed.
    """

    slug = "forest_fire"
    category = "random_models"

    def __init__(
        self,
        p_forward: float = 0.35,
        p_backward: float = 0.20,
        seed: int | None = None,
    ):
        self.p_forward = p_forward
        self.p_backward = p_backward
        self.seed = seed

    @property
    def name(self) -> str:
        return "Forest Fire"

    @property
    def description(self) -> str:
        return (
            f"Nodes burn through neighbors (p_fwd={self.p_forward}, "
            f"p_bwd={self.p_backward}). Produces densification."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n · E[fire_size])"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "p_forward", "Forward burning probability",
                "float", 0.35, "0 < p < 1",
            ),
            ParamInfo(
                "p_backward", "Backward burning probability",
                "float", 0.20, "0 ≤ p < 1",
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points

        seed = self.seed if self.seed is not None else (layout.seed + 109)
        rng = np.random.default_rng(seed)

        G = nx.Graph()
        G.add_node(0)

        # Grow the graph one node at a time
        for new_node in range(1, n):
            # Pick a random ambassador from existing nodes
            ambassador = int(rng.integers(new_node))

            # BFS-like burning process
            burned: set[int] = set()
            burned.add(new_node)  # Don't burn back to self

            queue: deque[int] = deque([ambassador])
            burned.add(ambassador)
            G.add_node(new_node)
            G.add_edge(new_node, ambassador)

            while queue:
                current = queue.popleft()

                # Get unburned neighbors
                neighbors = [
                    nbr for nbr in G.neighbors(current)
                    if nbr not in burned
                ]

                if not neighbors:
                    continue

                # Number to burn: Geometric(1 - p) draws for fwd/bwd
                n_forward = self._geometric_sample(
                    self.p_forward, len(neighbors), rng
                )
                n_backward = self._geometric_sample(
                    self.p_backward, len(neighbors), rng
                )
                n_burn = min(n_forward + n_backward, len(neighbors))

                if n_burn <= 0:
                    continue

                # Select which neighbors to burn
                burn_indices = rng.choice(
                    len(neighbors), size=n_burn, replace=False
                )

                for idx in burn_indices:
                    nbr = neighbors[idx]
                    burned.add(nbr)
                    G.add_edge(new_node, nbr)
                    queue.append(nbr)

        G.graph["model"] = "forest_fire"
        G.graph["p_forward"] = self.p_forward
        G.graph["p_backward"] = self.p_backward

        return G

    @staticmethod
    def _geometric_sample(
        p: float, max_val: int, rng: np.random.Generator
    ) -> int:
        """Sample from a geometric distribution, capped at max_val.

        Returns the number of "successes" before the first failure,
        where each trial succeeds with probability p.
        """
        if p <= 0 or max_val <= 0:
            return 0

        count = 0
        while count < max_val and rng.random() < p:
            count += 1
        return count

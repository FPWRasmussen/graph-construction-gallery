"""
Holme–Kim Model.

The Holme–Kim model extends Barabási–Albert preferential attachment
with an additional "triad formation" step that increases clustering:

    For each new node:
    1. Attach to one existing node v via preferential attachment
       (same as BA).
    2. For each of the remaining m-1 edges:
       a. With probability p, perform *triad formation*: connect to
          a random neighbor of the most recently attached node (if
          one exists that isn't already connected).
       b. With probability 1-p, perform standard preferential
          attachment.

This produces networks with:
    - Power-law degree distribution (like BA)
    - Tunable clustering coefficient (unlike BA, which has C → 0)

Reference:
    Holme, P. & Kim, B.J. (2002). "Growing scale-free networks with
    tunable clustering." Physical Review E, 65(2), 026107.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class HolmeKimGraph(GraphBuilder):
    """Preferential attachment + triad formation → tunable clustering.

    Parameters:
        m: Edges per new node.
        p: Triad formation probability.
        seed: Random seed.
    """

    slug = "holme_kim"
    category = "random_models"

    def __init__(
        self,
        m: int = 2,
        p: float = 0.5,
        seed: int | None = None,
    ):
        self.m = m
        self.p = p
        self.seed = seed

    @property
    def name(self) -> str:
        return "Holme–Kim"

    @property
    def description(self) -> str:
        return (
            f"BA + triad formation: m={self.m}, p_triad={self.p}. "
            f"Power-law degrees with tunable clustering."
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
            ParamInfo(
                "p", "Triad formation probability",
                "float", 0.5, "0 ≤ p ≤ 1",
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        m = min(self.m, n - 1)

        seed = self.seed if self.seed is not None else (layout.seed + 111)
        rng = np.random.default_rng(seed)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Seed graph: complete graph on first m+1 nodes
        m0 = m + 1
        for i in range(m0):
            for j in range(i + 1, m0):
                G.add_edge(i, j)

        # Repeated targets for preferential attachment
        targets: list[int] = []
        for i in range(m0):
            targets.extend([i] * (m0 - 1))  # Each has degree m0-1

        for new_node in range(m0, n):
            added: set[int] = set()

            # First edge: always preferential attachment
            first_target = self._preferential_choice(
                targets, new_node, added, rng
            )
            if first_target is not None:
                added.add(first_target)
                last_target = first_target

            # Remaining m-1 edges
            for _ in range(m - 1):
                if len(added) >= min(m, new_node):
                    break

                if rng.random() < self.p:
                    # Triad formation: connect to a neighbor of last_target
                    triad_target = self._triad_step(
                        G, last_target, new_node, added, rng
                    )
                    if triad_target is not None:
                        added.add(triad_target)
                        last_target = triad_target
                        continue

                # Preferential attachment (fallback or chosen)
                pa_target = self._preferential_choice(
                    targets, new_node, added, rng
                )
                if pa_target is not None:
                    added.add(pa_target)
                    last_target = pa_target

            # Add edges and update target list
            for target in added:
                G.add_edge(new_node, target)
                targets.append(new_node)
                targets.append(target)

        G.graph["model"] = "holme_kim"
        G.graph["m"] = m
        G.graph["p_triad"] = self.p

        return G

    @staticmethod
    def _preferential_choice(
        targets: list[int],
        new_node: int,
        exclude: set[int],
        rng: np.random.Generator,
    ) -> int | None:
        """Pick a node via preferential attachment, excluding already chosen."""
        max_attempts = len(targets)
        for _ in range(max_attempts):
            candidate = targets[int(rng.integers(len(targets)))]
            if candidate != new_node and candidate not in exclude:
                return candidate
        return None

    @staticmethod
    def _triad_step(
        G: nx.Graph,
        last_target: int,
        new_node: int,
        exclude: set[int],
        rng: np.random.Generator,
    ) -> int | None:
        """Try to connect to a neighbor of last_target (triad formation)."""
        neighbors = [
            nbr for nbr in G.neighbors(last_target)
            if nbr != new_node and nbr not in exclude
        ]
        if not neighbors:
            return None
        return int(rng.choice(neighbors))

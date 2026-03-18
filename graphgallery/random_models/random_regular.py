"""
Random Regular Graph.

A random d-regular graph is a graph chosen uniformly at random from
the set of all simple d-regular graphs on n vertices (where every
node has exactly degree d).

We use the pairing model (configuration model with fixed degree d for
all nodes) with rejection of self-loops and multi-edges.  For small
d and large n, this succeeds quickly.

Key properties:
    - Every node has exactly degree d
    - Good expander properties (spectral gap)
    - High connectivity for d ≥ 3
    - Used as null model for network analysis

Constraints:
    - n · d must be even (total stub count must be even)
    - d < n

Reference:
    Bollobás, B. (1980). "A probabilistic proof of an asymptotic
    formula for the number of labelled regular graphs." European
    Journal of Combinatorics, 1(4), 311–316.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class RandomRegularGraph(GraphBuilder):
    """Uniform random graph where all nodes have degree d.

    Parameters:
        d: Target degree for every node.
        seed: Random seed.
        max_retries: Maximum pairing attempts before giving up.
    """

    slug = "random_regular"
    category = "random_models"

    def __init__(
        self,
        d: int = 3,
        seed: int | None = None,
        max_retries: int = 100,
    ):
        self.d = d
        self.seed = seed
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        return "Random Regular"

    @property
    def description(self) -> str:
        return f"Every node has exactly degree d={self.d}."

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(nd) expected per attempt"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("d", "Degree for every node", "int", 3, "1 ≤ d < n, n·d even"),
            ParamInfo("seed", "Random seed", "int | None", None),
            ParamInfo("max_retries", "Max pairing attempts", "int", 100),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        n = layout.n_points
        if self.d >= n:
            raise ValueError(f"d={self.d} must be < n={n}")
        if (n * self.d) % 2 != 0:
            raise ValueError(
                f"n·d = {n}·{self.d} = {n * self.d} must be even."
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        d = self.d

        seed = self.seed if self.seed is not None else (layout.seed + 112)
        rng = np.random.default_rng(seed)

        for attempt in range(self.max_retries):
            result = self._try_pairing(n, d, rng)
            if result is not None:
                G = result
                G.graph["model"] = "random_regular"
                G.graph["d"] = d
                G.graph["attempts"] = attempt + 1
                return G

        # Fallback: use NetworkX's implementation
        import warnings
        warnings.warn(
            f"Pairing model failed after {self.max_retries} attempts. "
            f"Falling back to NetworkX random_regular_graph.",
            stacklevel=2,
        )
        G = nx.random_regular_graph(d, n, seed=int(rng.integers(2**31)))
        G = nx.relabel_nodes(G, {old: new for new, old in enumerate(G.nodes())})
        G.graph["model"] = "random_regular"
        G.graph["d"] = d
        G.graph["attempts"] = self.max_retries
        return G

    @staticmethod
    def _try_pairing(
        n: int, d: int, rng: np.random.Generator
    ) -> nx.Graph | None:
        """Attempt to generate a simple d-regular graph via stub pairing.

        Returns None if the pairing produces self-loops or multi-edges
        (caller should retry).
        """
        # Create d stubs per node
        stubs = np.repeat(np.arange(n), d)
        rng.shuffle(stubs)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        seen_edges: set[tuple[int, int]] = set()

        for idx in range(0, len(stubs), 2):
            u, v = int(stubs[idx]), int(stubs[idx + 1])

            # Reject self-loops
            if u == v:
                return None

            edge = (min(u, v), max(u, v))

            # Reject multi-edges
            if edge in seen_edges:
                return None

            seen_edges.add(edge)
            G.add_edge(u, v)

        return G

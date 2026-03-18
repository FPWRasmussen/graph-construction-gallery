"""
Configuration Model.

The configuration model generates a random graph with a prescribed
degree sequence.  Each node i is given d_i "stubs" (half-edges), and
stubs are paired uniformly at random to form edges.

The result may contain self-loops and multi-edges.  These can
optionally be removed to produce a simple graph (at the cost of
slightly altering the degree sequence).

The configuration model is central to network science as a null
model: "given this degree sequence, what does a maximally random
graph look like?"

Reference:
    Molloy, M. & Reed, B. (1995). "A critical point for random
    graphs with a given degree sequence." Random Structures &
    Algorithms, 6(2–3), 161–180.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class ConfigurationModelGraph(GraphBuilder):
    """Random graph with a prescribed degree sequence via stub matching.

    Parameters:
        degree_sequence: Optional explicit degree sequence (length n).
            If None, a power-law-ish sequence is auto-generated.
        remove_self_loops: Remove self-loops after construction.
        remove_multi_edges: Collapse multi-edges to simple edges.
        seed: Random seed.
    """

    slug = "configuration"
    category = "random_models"

    def __init__(
        self,
        degree_sequence: list[int] | np.ndarray | None = None,
        remove_self_loops: bool = True,
        remove_multi_edges: bool = True,
        seed: int | None = None,
    ):
        self.degree_sequence = degree_sequence
        self.remove_self_loops = remove_self_loops
        self.remove_multi_edges = remove_multi_edges
        self.seed = seed

    @property
    def name(self) -> str:
        return "Configuration Model"

    @property
    def description(self) -> str:
        return "Random graph with a prescribed degree sequence via stub matching."

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(Σd_i)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "degree_sequence",
                "Target degree per node or None for auto",
                "list[int] | None", None,
            ),
            ParamInfo("remove_self_loops", "Remove self-loops", "bool", True),
            ParamInfo("remove_multi_edges", "Collapse multi-edges", "bool", True),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points

        seed = self.seed if self.seed is not None else (layout.seed + 106)
        rng = np.random.default_rng(seed)

        # Resolve degree sequence
        degrees = self._resolve_degrees(n, rng)

        # Build stub list: node i appears degrees[i] times
        stubs: list[int] = []
        for node, deg in enumerate(degrees):
            stubs.extend([node] * deg)

        # Shuffle and pair stubs
        rng.shuffle(stubs)

        G = nx.MultiGraph() if not self.remove_multi_edges else nx.Graph()
        G.add_nodes_from(range(n))

        for idx in range(0, len(stubs) - 1, 2):
            u, v = stubs[idx], stubs[idx + 1]

            if self.remove_self_loops and u == v:
                continue

            G.add_edge(u, v)

        # Convert MultiGraph to Graph if needed
        if isinstance(G, nx.MultiGraph) and self.remove_multi_edges:
            G = nx.Graph(G)

        G.graph["model"] = "configuration"
        G.graph["target_degrees"] = degrees

        return G

    def _resolve_degrees(
        self, n: int, rng: np.random.Generator
    ) -> list[int]:
        """Get or generate a valid degree sequence."""
        if self.degree_sequence is not None:
            degrees = list(self.degree_sequence)
            if len(degrees) != n:
                raise ValueError(
                    f"Degree sequence length {len(degrees)} != n={n}"
                )
        else:
            # Auto-generate: power-law-ish with minimum degree 1
            # Use a truncated Zipf-like distribution
            raw = rng.zipf(2.5, size=n)
            degrees = [max(1, min(int(d), n - 1)) for d in raw]

        # Ensure the sum is even (stubs must pair up)
        total = sum(degrees)
        if total % 2 != 0:
            # Increment the degree of a random node
            idx = int(rng.integers(n))
            degrees[idx] += 1

        return degrees

"""
Chung–Lu Model.

The Chung–Lu model generates a random graph where the *expected*
degree of each node matches a given weight sequence.  The probability
of an edge (i, j) is:

    P(i, j) = w_i · w_j / S

where w_i is the weight (expected degree) of node i and S = Σ w_k.
This assumes max(w_i)² ≤ S (otherwise probabilities exceed 1).

Unlike the configuration model, the Chung–Lu model produces simple
graphs directly (no multi-edges or self-loops) and the actual degrees
are random variables that concentrate around their expectations.

Reference:
    Chung, F. & Lu, L. (2002). "Connected components in random
    graphs with given expected degree sequences." Annals of
    Combinatorics, 6(2), 125–145.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class ChungLuGraph(GraphBuilder):
    """Random graph with specified expected degree sequence.

    Parameters:
        expected_degrees: Optional (n,) weight vector.  If None,
            auto-generated from a log-normal distribution.
        seed: Random seed.
    """

    slug = "chung_lu"
    category = "random_models"

    def __init__(
        self,
        expected_degrees: list[float] | np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.expected_degrees = expected_degrees
        self.seed = seed

    @property
    def name(self) -> str:
        return "Chung–Lu"

    @property
    def description(self) -> str:
        return "Random graph with specified expected degree sequence."

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
            ParamInfo(
                "expected_degrees",
                "Expected degree per node or None for auto",
                "list[float] | None", None,
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points

        seed = self.seed if self.seed is not None else (layout.seed + 107)
        rng = np.random.default_rng(seed)

        # Resolve weights
        w = self._resolve_weights(n, rng)
        S = w.sum()

        # Validate: max(w)² ≤ S
        if w.max() ** 2 > S:
            # Scale down to satisfy the condition
            w = w * np.sqrt(S) / w.max()
            S = w.sum()

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                p_ij = min(w[i] * w[j] / S, 1.0)
                if rng.random() < p_ij:
                    G.add_edge(i, j)

        G.graph["model"] = "chung_lu"
        G.graph["expected_degrees"] = w.tolist()

        return G

    def _resolve_weights(
        self, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Get or generate the expected degree weights."""
        if self.expected_degrees is not None:
            w = np.asarray(self.expected_degrees, dtype=np.float64)
            if w.shape != (n,):
                raise ValueError(
                    f"Expected degrees shape {w.shape} != ({n},)"
                )
            return w

        # Auto: log-normal distribution → moderate heterogeneity
        w = rng.lognormal(mean=1.0, sigma=0.8, size=n)
        # Scale so mean expected degree ≈ 4
        w = w * (4.0 * n) / (w.sum())
        return w

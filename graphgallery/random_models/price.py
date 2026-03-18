"""
Price's Model (Directed Preferential Attachment).

Price's model (1976) is the directed precursor of the Barabási–Albert
model.  It was originally designed to explain the power-law distribution
of citation counts in academic papers:

    1. Nodes arrive sequentially (papers published over time).
    2. Each new node creates m directed edges (citations) pointing
       to existing nodes.
    3. The probability of citing node v is proportional to its current
       in-degree plus a constant a (initial attractiveness):

           P(v) = (deg_in(v) + a) / Σ_u (deg_in(u) + a)

The constant a > 0 ensures new nodes with zero in-degree still have
a nonzero probability of being cited.

The resulting in-degree distribution follows a power law with
exponent γ = 2 + a/m.

Reference:
    Price, D. de S. (1976). "A general theory of bibliometric and
    other cumulative advantage processes." JASIS, 27(5), 292–306.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class PriceGraph(GraphBuilder):
    """Directed citation network via Price's cumulative advantage.

    Parameters:
        m: Number of citations (out-edges) per new node.
        a: Initial attractiveness (prevents zero-probability nodes).
        seed: Random seed.
    """

    slug = "price"
    category = "random_models"

    def __init__(
        self,
        m: int = 3,
        a: float = 1.0,
        seed: int | None = None,
    ):
        self.m = m
        self.a = a
        self.seed = seed

    @property
    def name(self) -> str:
        return "Price's Model"

    @property
    def description(self) -> str:
        return (
            f"Directed preferential attachment (citations). "
            f"m={self.m} out-edges, attractiveness a={self.a}."
        )

    @property
    def is_directed(self) -> bool:
        return True

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
            ParamInfo("m", "Citations per new node", "int", 3, "m ≥ 1"),
            ParamInfo(
                "a", "Initial attractiveness",
                "float", 1.0, "a > 0",
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.DiGraph:
        self.validate_layout(layout)
        n = layout.n_points
        m = min(self.m, n - 1)

        seed = self.seed if self.seed is not None else (layout.seed + 110)
        rng = np.random.default_rng(seed)

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        # Seed: node 0 exists, node 1 cites node 0
        if n >= 2:
            G.add_edge(1, 0)

        # In-degree tracker (faster than querying G.in_degree each time)
        in_deg = np.zeros(n, dtype=np.float64)
        if n >= 2:
            in_deg[0] = 1.0

        # Grow: each new node cites m existing nodes
        for new_node in range(2, n):
            # Compute citation probabilities for nodes 0..new_node-1
            existing = new_node
            weights = in_deg[:existing] + self.a
            total = weights.sum()

            if total <= 0:
                # Uniform fallback
                probs = np.ones(existing) / existing
            else:
                probs = weights / total

            # Select m distinct targets
            m_actual = min(m, existing)
            targets = rng.choice(
                existing, size=m_actual, replace=False, p=probs
            )

            for target in targets:
                G.add_edge(new_node, int(target))
                in_deg[int(target)] += 1.0

        G.graph["model"] = "price"
        G.graph["m"] = m
        G.graph["a"] = self.a

        return G

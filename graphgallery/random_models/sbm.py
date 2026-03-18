"""
Stochastic Block Model (SBM).

The SBM generates graphs with planted community structure.  Nodes
are partitioned into groups, and the probability of an edge depends
only on the group memberships of the two endpoints:

    P(edge between i, j) = B[g(i), g(j)]

where B is a k×k matrix of inter/intra-group edge probabilities
and g(i) is the group assignment of node i.

When the diagonal of B is larger than the off-diagonal, the model
produces assortative (community) structure.  When the off-diagonal
dominates, it produces disassortative (bipartite-like) structure.

The SBM is the foundation of community detection theory and
is widely used as a benchmark for clustering algorithms.

Reference:
    Holland, P.W., Laskey, K.B., & Leinhardt, S. (1983). "Stochastic
    blockmodels: First steps." Social Networks, 5(2), 109–137.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class StochasticBlockModelGraph(GraphBuilder):
    """Community-structured random graph with planted partitions.

    Uses the cluster labels from the layout as group assignments.

    Parameters:
        p_within: Edge probability within the same group.
        p_between: Edge probability between different groups.
        seed: Random seed.
    """

    slug = "sbm"
    category = "random_models"

    def __init__(
        self,
        p_within: float = 0.4,
        p_between: float = 0.05,
        seed: int | None = None,
    ):
        self.p_within = p_within
        self.p_between = p_between
        self.seed = seed

    @property
    def name(self) -> str:
        return "Stochastic Block Model"

    @property
    def description(self) -> str:
        return (
            f"Community structure: p_within={self.p_within}, "
            f"p_between={self.p_between}."
        )

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
                "p_within", "Intra-community edge probability",
                "float", 0.4, "0 ≤ p ≤ 1",
            ),
            ParamInfo(
                "p_between", "Inter-community edge probability",
                "float", 0.05, "0 ≤ p ≤ 1",
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        labels = layout.labels

        seed = self.seed if self.seed is not None else (layout.seed + 105)
        rng = np.random.default_rng(seed)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    p = self.p_within
                else:
                    p = self.p_between

                if rng.random() < p:
                    G.add_edge(i, j)

        G.graph["model"] = "sbm"
        G.graph["p_within"] = self.p_within
        G.graph["p_between"] = self.p_between

        return G

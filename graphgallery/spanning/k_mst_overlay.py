"""
k-MST Overlay Graph.

The k-MST overlay constructs multiple diverse spanning trees and
takes their union (edge superposition). This produces a graph that
is richer than any single spanning tree but much sparser than the
complete graph.

Strategy:
    1. Build the 1st MST normally.
    2. For each subsequent tree, perturb the edge weights (by adding
       random noise or increasing the weights of previously-used edges)
       and compute a new MST on the perturbed graph.
    3. Take the union of all k trees.

The result has between n-1 (if all trees are identical) and
k(n-1) edges (if all trees are completely disjoint).  In practice
it produces a graph with good connectivity, moderate density, and
edges that span diverse geometric paths through the point set.

Use cases:
    - Graph-based semi-supervised learning (more robust than single MST)
    - Approximate nearest neighbor graph initialization
    - Network design with redundancy
"""

from __future__ import annotations

import heapq
from typing import Optional

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class KMSTOverlayGraph(GraphBuilder):
    """Union of k diverse minimum spanning trees.

    After the first MST, each subsequent MST is computed on perturbed
    weights that discourage reuse of previously selected edges.

    Parameters:
        k: Number of spanning trees to overlay.
        penalty: Multiplicative penalty applied to already-used edges
            before computing the next MST. Higher values produce more
            diverse trees.
        noise_scale: Scale of additive Gaussian noise on edge weights
            for each subsequent tree (0 = no noise, just penalty).
        seed: Random seed for noise generation.
    """

    slug = "k_mst_overlay"
    category = "spanning"

    def __init__(
        self,
        k: int = 3,
        penalty: float = 2.0,
        noise_scale: float = 0.1,
        seed: int | None = None,
    ):
        self.k = k
        self.penalty = penalty
        self.noise_scale = noise_scale
        self.seed = seed

    @property
    def name(self) -> str:
        return "k-MST Overlay"

    @property
    def description(self) -> str:
        return (
            f"Union of {self.k} diverse spanning trees. "
            f"Denser than a single MST but much sparser than complete."
        )

    @property
    def is_deterministic(self) -> bool:
        return self.noise_scale == 0.0

    @property
    def complexity(self) -> str:
        return f"O(k · n² log n) for k trees on complete graph"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("k", "Number of spanning trees", "int", 3, "k ≥ 1"),
            ParamInfo(
                "penalty",
                "Weight multiplier for reused edges",
                "float",
                2.0,
                "> 1.0 for diversity",
            ),
            ParamInfo(
                "noise_scale",
                "Gaussian noise added to weights each round",
                "float",
                0.1,
                "≥ 0",
            ),
            ParamInfo(
                "seed",
                "Random seed for noise",
                "int | None",
                None,
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        base_dist = pairwise_distances(layout.points)

        seed = self.seed if self.seed is not None else (layout.seed + 13)
        rng = np.random.default_rng(seed)

        # Track which edges have been used across all trees
        edge_usage = np.zeros((n, n), dtype=np.float64)

        all_edges: dict[tuple[int, int], float] = {}
        tree_count = 0

        for t in range(self.k):
            # Perturbed weights = base + penalty for reused + noise
            weights = base_dist.copy()

            if t > 0:
                # Penalty: multiply distance by (1 + penalty * usage_count)
                weights *= (1.0 + self.penalty * edge_usage)

                # Additive noise
                if self.noise_scale > 0:
                    noise = rng.normal(
                        0, self.noise_scale * base_dist.mean(), size=(n, n)
                    )
                    noise = (noise + noise.T) / 2  # Symmetrize
                    np.fill_diagonal(noise, 0.0)
                    weights = np.maximum(weights + noise, 1e-12)

            # Compute MST via Kruskal's on the perturbed weights
            tree_edges = self._kruskal(n, weights)

            for u, v, _ in tree_edges:
                edge_key = (min(u, v), max(u, v))
                # Store original distance (not perturbed)
                if edge_key not in all_edges:
                    all_edges[edge_key] = float(base_dist[u, v])
                edge_usage[u, v] += 1.0
                edge_usage[v, u] += 1.0

            tree_count += 1

        # Build graph from the union of all tree edges
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for (u, v), w in all_edges.items():
            G.add_edge(u, v, weight=w)

        G.graph["algorithm"] = "k_mst_overlay"
        G.graph["n_trees"] = tree_count
        G.graph["total_unique_edges"] = len(all_edges)
        G.graph["total_weight"] = sum(all_edges.values())

        return G

    @staticmethod
    def _kruskal(n: int, weights: np.ndarray) -> list[tuple[int, int, float]]:
        """Kruskal's MST on an (n, n) weight matrix.

        Returns:
            List of (u, v, weight) edges in the MST.
        """
        # Collect upper-triangle edges
        edges: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((float(weights[i, j]), i, j))

        edges.sort(key=lambda e: e[0])

        # Simple union-find
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        mst: list[tuple[int, int, float]] = []
        for w, u, v in edges:
            if union(u, v):
                mst.append((u, v, w))
                if len(mst) == n - 1:
                    break

        return mst

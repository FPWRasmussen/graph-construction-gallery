"""
Expansion Graph.

The expansion graph connects nodes based on a spatial expansion
criterion: each node expands a sphere of influence whose radius
grows incrementally, and edges are created when two expanding
spheres meet.  Unlike fixed-radius graphs, the expansion rate
can adapt to local density.

This builder implements a **distance-rank** expansion variant:
    1. For each node i, rank all other nodes by distance.
    2. Connect node i to all nodes within its expansion threshold,
       defined as a percentile of the distance distribution.

It also implements a **diffusion-based** variant:
    1. Construct an initial k-NN graph.
    2. Simulate random walks / diffusion on the graph.
    3. Connect pairs with high diffusion affinity (probability of
       reaching each other via short random walks).

The diffusion variant discovers multi-scale structure and can
reveal cluster boundaries that are invisible to simple distance
thresholds.

Reference:
    Coifman, R.R. & Lafon, S. (2006). "Diffusion maps." Applied
    and Computational Harmonic Analysis, 21(1), 5–30.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import (
    PointLayout,
    pairwise_distances,
    k_nearest_indices,
)


class ExpansionGraph(GraphBuilder):
    """Connect nodes within an adaptive spatial expansion threshold.

    Parameters:
        method: Expansion method:
            - ``"percentile"``: Connect within a distance percentile.
            - ``"diffusion"``: Diffusion affinity on an initial k-NN graph.
        percentile: Distance percentile for the percentile method (0–100).
        k_initial: Initial k-NN for the diffusion method.
        diffusion_steps: Number of diffusion steps (random walk length).
        diffusion_threshold: Minimum diffusion affinity for edges.
    """

    slug = "expansion"
    category = "data_driven"

    def __init__(
        self,
        method: Literal["percentile", "diffusion"] = "diffusion",
        percentile: float = 20.0,
        k_initial: int = 5,
        diffusion_steps: int = 3,
        diffusion_threshold: float = 0.01,
    ):
        self.method = method
        self.percentile = percentile
        self.k_initial = k_initial
        self.diffusion_steps = diffusion_steps
        self.diffusion_threshold = diffusion_threshold

    @property
    def name(self) -> str:
        return "Expansion Graph"

    @property
    def description(self) -> str:
        if self.method == "percentile":
            return (
                f"Connect within {self.percentile}th percentile of "
                f"per-node distance distribution."
            )
        return (
            f"Diffusion affinity graph: {self.diffusion_steps}-step "
            f"random walk on initial k-NN."
        )

    @property
    def complexity(self) -> str:
        if self.method == "percentile":
            return "O(n² log n)"
        return "O(n² · steps) for matrix power"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "method", "Expansion method",
                "str", "diffusion", "percentile | diffusion",
            ),
            ParamInfo(
                "percentile", "Distance percentile (percentile method)",
                "float", 20.0, "0 < p ≤ 100",
            ),
            ParamInfo(
                "k_initial", "Initial k-NN (diffusion method)",
                "int", 5, "k ≥ 1",
            ),
            ParamInfo(
                "diffusion_steps", "Random walk steps",
                "int", 3, "≥ 1",
            ),
            ParamInfo(
                "diffusion_threshold", "Min diffusion affinity",
                "float", 0.01, "≥ 0",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)

        if self.method == "percentile":
            return self._build_percentile(layout)
        elif self.method == "diffusion":
            return self._build_diffusion(layout)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _build_percentile(self, layout: PointLayout) -> nx.Graph:
        """Percentile-based expansion: per-node adaptive radius."""
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            # Get distances from node i to all others
            dists_i = dist[i].copy()
            dists_i[i] = np.inf  # Exclude self

            # Compute threshold as percentile of distances
            finite_dists = dists_i[dists_i < np.inf]
            threshold_i = np.percentile(finite_dists, self.percentile)

            # Connect to all nodes within threshold
            for j in range(i + 1, n):
                if dist[i, j] <= threshold_i or dist[j, i] <= np.percentile(
                    dist[j][dist[j] < np.inf], self.percentile
                ):
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=float(dist[i, j]))

        G.graph["algorithm"] = "expansion_percentile"
        G.graph["percentile"] = self.percentile

        return G

    def _build_diffusion(self, layout: PointLayout) -> nx.Graph:
        """Diffusion-based expansion: random walk affinity."""
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        # Step 1: Build initial k-NN graph as adjacency matrix
        knn_idx = k_nearest_indices(dist, self.k_initial)

        # Gaussian kernel on k-NN edges
        sigma = np.median(dist[dist > 0])
        W = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in knn_idx[i]:
                j = int(j)
                w = np.exp(-dist[i, j] ** 2 / (2 * sigma ** 2))
                W[i, j] = w
                W[j, i] = w  # Symmetrize

        # Step 2: Normalize to get transition matrix
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        P = W / row_sums[:, np.newaxis]

        # Step 3: Diffuse: P^t gives t-step transition probabilities
        P_t = np.linalg.matrix_power(P, self.diffusion_steps)

        # Step 4: Compute diffusion affinity (symmetrize)
        affinity = (P_t + P_t.T) / 2.0
        np.fill_diagonal(affinity, 0.0)

        # Step 5: Build graph from thresholded affinity
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                a = affinity[i, j]
                if a >= self.diffusion_threshold:
                    G.add_edge(
                        i, j,
                        weight=float(a),
                        diffusion_affinity=float(a),
                    )

        G.graph["algorithm"] = "expansion_diffusion"
        G.graph["k_initial"] = self.k_initial
        G.graph["diffusion_steps"] = self.diffusion_steps
        G.graph["diffusion_threshold"] = self.diffusion_threshold
        G.graph["diffusion_matrix"] = P_t

        return G

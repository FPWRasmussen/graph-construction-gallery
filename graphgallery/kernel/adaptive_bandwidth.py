"""
Adaptive Bandwidth Kernel Graph (Self-Tuning Spectral Clustering).

In the standard Gaussian kernel graph, a single global bandwidth σ
is used for all pairs.  This works poorly when the data has clusters
of varying density: a σ suitable for a dense cluster will over-connect
sparse regions, and vice versa.

The **adaptive bandwidth** (self-tuning) approach assigns each point
a local bandwidth σ_i based on the distance to its k-th nearest
neighbor:

    σ_i = d(x_i, x_i^{(k)})

The kernel between points i and j then uses the geometric mean of
their bandwidths:

    K(x_i, x_j) = exp(−‖x_i − x_j‖² / (σ_i · σ_j))

This naturally adapts to local density:
    - Dense regions → small σ → tight, local connections
    - Sparse regions → large σ → broader, long-range connections

The result is dramatically better spectral clustering performance
on multi-scale data (like our two-cluster layout with different σ).

Reference:
    Zelnik-Manor, L. & Perona, P. (2005). "Self-Tuning Spectral
    Clustering." NeurIPS 17, pp. 1601–1608.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import (
    PointLayout,
    pairwise_distances,
    k_nearest_indices,
)
from graphgallery.kernel._kernel_utils import (
    gaussian_kernel_matrix,
    threshold_sparsify,
    knn_sparsify,
    similarity_matrix_stats,
)


class AdaptiveBandwidthGraph(GraphBuilder):
    """Gaussian kernel with per-point bandwidth from k-th neighbor distance.

    Each point's σ adapts to local density, enabling multi-scale
    connectivity.  Ideal for data with clusters of varying density.

    Parameters:
        k_bandwidth: Which nearest neighbor distance to use as σ_i.
        threshold: Minimum kernel value for edge creation.
        sparsify_knn: If > 0, additionally keep only this many
            strongest connections per node (kNN sparsification).
    """

    slug = "adaptive_bandwidth"
    category = "kernel"

    def __init__(
        self,
        k_bandwidth: int = 7,
        threshold: float = 0.05,
        sparsify_knn: int = 0,
    ):
        self.k_bandwidth = k_bandwidth
        self.threshold = threshold
        self.sparsify_knn = sparsify_knn

    @property
    def name(self) -> str:
        return "Adaptive Bandwidth Kernel"

    @property
    def description(self) -> str:
        return (
            f"Gaussian kernel with per-point σ from {self.k_bandwidth}-th "
            f"neighbor distance. Self-tuning for multi-scale data."
        )

    @property
    def complexity(self) -> str:
        return "O(n² d)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "k_bandwidth", "k-th neighbor for bandwidth estimation",
                "int", 7, "1 ≤ k < n",
            ),
            ParamInfo(
                "threshold", "Minimum kernel value for edges",
                "float", 0.05, "0 ≤ threshold < 1",
            ),
            ParamInfo(
                "sparsify_knn", "Additional kNN sparsification (0=off)",
                "int", 0, "≥ 0",
            ),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if self.k_bandwidth >= layout.n_points:
            raise ValueError(
                f"k_bandwidth={self.k_bandwidth} must be < "
                f"n_points={layout.n_points}"
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points

        dist = pairwise_distances(points)

        # Compute per-point bandwidth: distance to k-th nearest neighbor
        knn_idx = k_nearest_indices(dist, self.k_bandwidth)
        sigma = np.array([
            dist[i, knn_idx[i, -1]] for i in range(n)
        ], dtype=np.float64)

        # Ensure no zero bandwidths
        sigma = np.maximum(sigma, 1e-10)

        # Compute adaptive kernel matrix
        K = gaussian_kernel_matrix(points, sigma)

        # Sparsify by threshold
        K_sparse = threshold_sparsify(K, self.threshold)

        # Optional additional kNN sparsification
        if self.sparsify_knn > 0:
            K_sparse = knn_sparsify(K_sparse, self.sparsify_knn)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                w = K_sparse[i, j]
                if w > 0:
                    G.add_edge(i, j, weight=float(w))

        stats = similarity_matrix_stats(K)
        G.graph["algorithm"] = "adaptive_bandwidth"
        G.graph["k_bandwidth"] = self.k_bandwidth
        G.graph["threshold"] = self.threshold
        G.graph["per_point_sigma"] = sigma.tolist()
        G.graph["sigma_range"] = (float(sigma.min()), float(sigma.max()))
        G.graph["kernel_stats"] = stats

        return G

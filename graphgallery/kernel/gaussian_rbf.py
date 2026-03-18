"""
Gaussian (RBF) Kernel Graph.

The Gaussian kernel (also called the Radial Basis Function or RBF
kernel) maps Euclidean distances to similarity scores via:

    K(x_i, x_j) = exp(−‖x_i − x_j‖² / (2σ²))

where σ is the bandwidth parameter controlling the decay rate.

The kernel is continuous, symmetric, and bounded in [0, 1], making
it ideal for graph edge weights [[5]].  In the feature space
induced by the Gaussian kernel, all points lie on a unit hypersphere:
‖Φ(x)‖ = K(x, x) = 1 [[3]].

The bandwidth σ critically affects the graph structure:
    - Small σ → only very close points have nonzero weight (sparse)
    - Large σ → all pairs have similar weight (approaches complete)
    - σ ≈ median pairwise distance → reasonable default

Edges with kernel value below a threshold are pruned for sparsity.

The Gaussian kernel graph is the foundation of spectral clustering
(Ng, Jordan & Weiss, 2002) and diffusion maps.

Reference:
    Ng, A.Y., Jordan, M.I., & Weiss, Y. (2002). "On Spectral
    Clustering: Analysis and an Algorithm." NeurIPS 14.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.kernel._kernel_utils import (
    gaussian_kernel_matrix,
    threshold_sparsify,
    similarity_matrix_stats,
)


class GaussianRBFGraph(GraphBuilder):
    """Fully weighted graph from the Gaussian (RBF) kernel.

    Edge weights are K(i,j) = exp(-‖x_i - x_j‖² / 2σ²).
    Edges below a minimum weight threshold are pruned.

    Parameters:
        sigma: Bandwidth parameter.  If None, uses the median
            pairwise distance (a common heuristic).
        threshold: Minimum kernel value to create an edge.
            Set to 0 for a fully connected weighted graph.
    """

    slug = "gaussian_rbf"
    category = "kernel"

    def __init__(
        self,
        sigma: float | None = None,
        threshold: float = 0.1,
    ):
        self.sigma = sigma
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Gaussian (RBF) Kernel"

    @property
    def description(self) -> str:
        sigma_str = f"σ={self.sigma}" if self.sigma else "σ=median"
        return (
            f"Edge weights from exp(-‖x-y‖²/2σ²). "
            f"{sigma_str}, threshold={self.threshold}."
        )

    @property
    def complexity(self) -> str:
        return "O(n²d)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "sigma", "Kernel bandwidth (None=median heuristic)",
                "float | None", None, "σ > 0",
            ),
            ParamInfo(
                "threshold", "Minimum kernel value for edge creation",
                "float", 0.1, "0 ≤ threshold ≤ 1",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points

        # Resolve sigma
        sigma = self._resolve_sigma(points)

        # Compute kernel matrix
        K = gaussian_kernel_matrix(points, sigma)

        # Sparsify
        K_sparse = threshold_sparsify(K, self.threshold)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                w = K_sparse[i, j]
                if w > 0:
                    G.add_edge(i, j, weight=float(w))

        stats = similarity_matrix_stats(K)
        G.graph["algorithm"] = "gaussian_rbf"
        G.graph["sigma"] = sigma
        G.graph["threshold"] = self.threshold
        G.graph["kernel_stats"] = stats

        return G

    def _resolve_sigma(self, points: np.ndarray) -> float:
        """Determine the bandwidth σ."""
        if self.sigma is not None:
            return self.sigma

        # Median heuristic: σ = median of pairwise distances
        dist = pairwise_distances(points)
        upper_tri = dist[np.triu_indices(dist.shape[0], k=1)]
        return float(np.median(upper_tri))

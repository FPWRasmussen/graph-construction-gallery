"""
Thresholded Similarity Graph.

A generic framework for building graphs from any pairwise similarity
or kernel function by applying a threshold.  This builder supports
multiple built-in similarity measures and accepts custom functions.

Supported built-in measures:
    - ``"gaussian"``: Gaussian / RBF kernel
    - ``"cosine"``: Cosine similarity
    - ``"polynomial"``: Polynomial kernel (⟨x,y⟩ + c)^p
    - ``"laplacian"``: Laplacian kernel exp(-‖x-y‖ / σ)
    - ``"sigmoid"``: Sigmoid kernel tanh(α⟨x,y⟩ + c)
    - ``"inverse_distance"``: 1 / (1 + ‖x-y‖)

The sklearn.metrics.pairwise module provides similar functionality
for computing pairwise distances and kernel values [[1]].

An edge (i, j) is created iff similarity(i, j) ≥ threshold.

This builder serves as the "Swiss army knife" of similarity graphs,
demonstrating that the specific kernel matters less than the
combination of kernel + sparsification strategy.
"""

from __future__ import annotations

from typing import Callable, Literal

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.kernel._kernel_utils import (
    gaussian_kernel_matrix,
    cosine_similarity_matrix,
    threshold_sparsify,
    similarity_matrix_stats,
)


# ---------------------------------------------------------------------------
# Built-in similarity functions
# ---------------------------------------------------------------------------

SimilarityMeasure = Literal[
    "gaussian",
    "cosine",
    "polynomial",
    "laplacian",
    "sigmoid",
    "inverse_distance",
]


def _compute_similarity(
    points: np.ndarray,
    measure: SimilarityMeasure,
    **kwargs,
) -> np.ndarray:
    """Compute a pairwise similarity matrix for the given measure.

    Args:
        points: (n, d) array.
        measure: Name of the similarity measure.
        **kwargs: Measure-specific parameters.

    Returns:
        (n, n) symmetric similarity matrix.
    """
    if measure == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        return gaussian_kernel_matrix(points, sigma)

    elif measure == "cosine":
        return cosine_similarity_matrix(points)

    elif measure == "polynomial":
        degree = kwargs.get("degree", 3)
        coef0 = kwargs.get("coef0", 1.0)
        gram = points @ points.T
        K = (gram + coef0) ** degree
        # Normalize to [0, 1] range
        diag = np.sqrt(np.diag(K))
        diag = np.maximum(diag, 1e-12)
        K_norm = K / np.outer(diag, diag)
        return K_norm

    elif measure == "laplacian":
        sigma = kwargs.get("sigma", 1.0)
        dist = pairwise_distances(points)
        return np.exp(-dist / sigma)

    elif measure == "sigmoid":
        alpha = kwargs.get("alpha", 0.1)
        coef0 = kwargs.get("coef0", 0.0)
        gram = points @ points.T
        K = np.tanh(alpha * gram + coef0)
        # Shift to [0, 1] range: (K + 1) / 2
        K = (K + 1.0) / 2.0
        return K

    elif measure == "inverse_distance":
        dist = pairwise_distances(points)
        return 1.0 / (1.0 + dist)

    else:
        raise ValueError(f"Unknown similarity measure: {measure}")


class ThresholdedSimilarityGraph(GraphBuilder):
    """Generic thresholded similarity graph supporting multiple kernels.

    Computes a pairwise similarity matrix using the chosen measure,
    then creates edges where similarity ≥ threshold.

    Parameters:
        measure: Similarity measure name.
        threshold: Minimum similarity for edge creation.
        measure_params: Dict of measure-specific parameters.
    """

    slug = "thresholded"
    category = "kernel"

    def __init__(
        self,
        measure: SimilarityMeasure = "laplacian",
        threshold: float = 0.3,
        measure_params: dict | None = None,
    ):
        self.measure = measure
        self.threshold = threshold
        self.measure_params = measure_params or {}

    @property
    def name(self) -> str:
        return "Thresholded Similarity"

    @property
    def description(self) -> str:
        return (
            f"Binarize {self.measure} similarity at "
            f"threshold={self.threshold}."
        )

    @property
    def complexity(self) -> str:
        return "O(n²d)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "measure", "Similarity measure name",
                "str", "laplacian",
                "gaussian|cosine|polynomial|laplacian|sigmoid|inverse_distance",
            ),
            ParamInfo(
                "threshold", "Minimum similarity for edges",
                "float", 0.3, "0 ≤ threshold ≤ 1",
            ),
            ParamInfo(
                "measure_params", "Measure-specific parameters",
                "dict", {},
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points

        # Compute similarity matrix
        S = _compute_similarity(points, self.measure, **self.measure_params)

        # Sparsify
        S_sparse = threshold_sparsify(S, self.threshold)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                w = S_sparse[i, j]
                if w > 0:
                    G.add_edge(i, j, weight=float(w))

        stats = similarity_matrix_stats(S)
        G.graph["algorithm"] = "thresholded_similarity"
        G.graph["measure"] = self.measure
        G.graph["threshold"] = self.threshold
        G.graph["similarity_stats"] = stats

        return G

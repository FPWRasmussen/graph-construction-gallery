"""
Jaccard Similarity Graph.

The Jaccard index measures the overlap between two sets:

    J(A, B) = |A ∩ B| / |A ∪ B|

It ranges from 0 (disjoint) to 1 (identical) and is widely used
for binary / categorical data: document similarity, species
co-occurrence, user activity overlap, and genomic comparisons.

For our continuous 2D point coordinates, we create binary features
via multiple strategies:
    1. **Spatial binning**: Divide the bounding box into a grid and
       create binary features for which bins each point falls near.
    2. **Neighbor fingerprinting**: Use approximate k-NN membership
       as binary features (points sharing many neighbors are similar).
    3. **Multi-threshold binarization**: Binarize coordinates at
       multiple thresholds and average the Jaccard across them.

This demonstrates how a set-based similarity metric can be applied
to spatial data through feature engineering.
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
from graphgallery.kernel._kernel_utils import (
    jaccard_multithreshold_similarity,
    similarity_matrix_stats,
)


def _spatial_bin_features(
    points: np.ndarray,
    n_bins: int = 8,
    radius: float = 1.0,
) -> np.ndarray:
    """Create binary features by spatial binning with soft assignment.

    Divides the bounding box into an n_bins × n_bins grid.  Each point
    gets a 1 for every grid cell within ``radius`` of its position.

    Args:
        points: (n, 2) array.
        n_bins: Grid resolution per axis.
        radius: Soft assignment radius in grid cell units.

    Returns:
        (n, n_bins²) binary feature array.
    """
    n = points.shape[0]
    n_features = n_bins * n_bins

    # Normalize points to [0, n_bins - 1] range
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    p_range = p_max - p_min
    p_range[p_range < 1e-10] = 1.0

    normalized = (points - p_min) / p_range * (n_bins - 1)

    # Grid cell centers
    features = np.zeros((n, n_features), dtype=np.float64)

    for i in range(n):
        px, py = normalized[i]
        for bx in range(n_bins):
            for by in range(n_bins):
                dist = np.sqrt((px - bx) ** 2 + (py - by) ** 2)
                if dist <= radius:
                    features[i, bx * n_bins + by] = 1.0

    return features


def _neighbor_fingerprint_features(
    points: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Create binary features from k-NN membership.

    For each point, a binary vector indicates which other points are
    among its k nearest neighbors.  Points with overlapping neighbor
    sets will have high Jaccard similarity.

    Args:
        points: (n, 2) array.
        k: Number of neighbors.

    Returns:
        (n, n) binary feature array where feature[i][j] = 1 iff
        j is a k-NN of i.
    """
    dist = pairwise_distances(points)
    knn = k_nearest_indices(dist, k)

    n = points.shape[0]
    features = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in knn[i]:
            features[i, int(j)] = 1.0

    return features


class JaccardSimilarityGraph(GraphBuilder):
    """Connect points based on Jaccard similarity of derived binary features.

    Parameters:
        method: Feature generation method.
            - ``"spatial_bin"``: Grid-based soft binning.
            - ``"neighbor_fingerprint"``: k-NN membership overlap.
            - ``"multi_threshold"``: Multi-threshold binarization.
        threshold: Minimum Jaccard similarity for edge creation.
        method_params: Dict of method-specific parameters.
    """

    slug = "jaccard"
    category = "kernel"

    def __init__(
        self,
        method: Literal[
            "spatial_bin", "neighbor_fingerprint", "multi_threshold"
        ] = "spatial_bin",
        threshold: float = 0.3,
        method_params: dict | None = None,
    ):
        self.method = method
        self.threshold = threshold
        self.method_params = method_params or {}

    @property
    def name(self) -> str:
        return "Jaccard Similarity"

    @property
    def description(self) -> str:
        return (
            f"Set-overlap similarity via {self.method} features. "
            f"J(A,B) = |A∩B|/|A∪B| ≥ {self.threshold}."
        )

    @property
    def complexity(self) -> str:
        return "O(n² · n_features)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "method", "Feature generation method",
                "str", "spatial_bin",
                "spatial_bin | neighbor_fingerprint | multi_threshold",
            ),
            ParamInfo(
                "threshold", "Minimum Jaccard for edges",
                "float", 0.3, "0 ≤ threshold ≤ 1",
            ),
            ParamInfo(
                "method_params", "Method-specific parameters",
                "dict", {},
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points

        # Generate binary features
        features = self._generate_features(points, layout)

        # Compute Jaccard similarity matrix
        S = self._jaccard_matrix(features)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = S[i, j]
                if sim >= self.threshold:
                    G.add_edge(i, j, weight=float(sim))

        stats = similarity_matrix_stats(S)
        G.graph["algorithm"] = "jaccard"
        G.graph["method"] = self.method
        G.graph["threshold"] = self.threshold
        G.graph["n_features"] = features.shape[1]
        G.graph["similarity_stats"] = stats

        return G

    def _generate_features(
        self,
        points: np.ndarray,
        layout: PointLayout,
    ) -> np.ndarray:
        """Generate binary features based on the chosen method."""
        if self.method == "spatial_bin":
            n_bins = self.method_params.get("n_bins", 8)
            radius = self.method_params.get("radius", 1.5)
            return _spatial_bin_features(points, n_bins=n_bins, radius=radius)

        elif self.method == "neighbor_fingerprint":
            k = self.method_params.get("k", 5)
            return _neighbor_fingerprint_features(points, k=k)

        elif self.method == "multi_threshold":
            n_thresholds = self.method_params.get("n_thresholds", 10)
            # Use the multi-threshold utility
            # (returns similarity directly, not features)
            return points  # Will compute Jaccard in a special path

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _jaccard_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute pairwise Jaccard similarity."""
        if self.method == "multi_threshold":
            n_thresholds = self.method_params.get("n_thresholds", 10)
            seed = self.method_params.get("seed", 42)
            return jaccard_multithreshold_similarity(
                features, n_thresholds=n_thresholds, seed=seed
            )

        # Standard binary Jaccard
        n = features.shape[0]
        S = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i, n):
                intersection = np.sum(np.minimum(features[i], features[j]))
                union = np.sum(np.maximum(features[i], features[j]))

                if union > 0:
                    sim = intersection / union
                else:
                    sim = 0.0

                S[i, j] = sim
                S[j, i] = sim

        return S

"""
Canonical point layout generator for the Graph Construction Gallery.

Provides the standard 30-point, two-cluster layout used across all
graph construction examples, ensuring visual consistency and fair
comparison between algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ClusterSpec:
    """Specification for a single Gaussian cluster.

    Attributes:
        n_points: Number of points in the cluster.
        center: (x, y) coordinates of the cluster center.
        std: Standard deviation of the Gaussian distribution.
        label: Human-readable name for the cluster.
    """

    n_points: int
    center: tuple[float, float]
    std: float
    label: str = ""


@dataclass(frozen=True)
class PointLayout:
    """Container for a generated point layout with metadata.

    Attributes:
        points: (n, 2) array of point coordinates.
        labels: (n,) integer array indicating cluster membership.
        cluster_specs: List of ClusterSpec objects used to generate the layout.
        seed: Random seed used for reproducibility.
    """

    points: np.ndarray
    labels: np.ndarray
    cluster_specs: tuple[ClusterSpec, ...]
    seed: int

    @property
    def n_points(self) -> int:
        """Total number of points."""
        return self.points.shape[0]

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return len(self.cluster_specs)

    @property
    def x(self) -> np.ndarray:
        """X-coordinates of all points."""
        return self.points[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Y-coordinates of all points."""
        return self.points[:, 1]

    def cluster_points(self, cluster_id: int) -> np.ndarray:
        """Return points belonging to a specific cluster.

        Args:
            cluster_id: Zero-based cluster index.

        Returns:
            (m, 2) array of points in the requested cluster.
        """
        return self.points[self.labels == cluster_id]

    def bounding_box(self, padding: float = 0.5) -> tuple[float, float, float, float]:
        """Compute the bounding box of the layout with optional padding.

        Args:
            padding: Extra space around the bounding box on each side.

        Returns:
            (xmin, xmax, ymin, ymax) tuple.
        """
        return (
            float(self.x.min() - padding),
            float(self.x.max() + padding),
            float(self.y.min() - padding),
            float(self.y.max() + padding),
        )


# ---------------------------------------------------------------------------
# Default canonical layout specification
# ---------------------------------------------------------------------------

DEFAULT_CLUSTERS = (
    ClusterSpec(n_points=10, center=(-2.0, 0.0), std=0.5, label="Small cluster"),
    ClusterSpec(n_points=20, center=(2.0, 0.0), std=0.8, label="Large cluster"),
)

DEFAULT_SEED = 42


def make_layout(
    clusters: tuple[ClusterSpec, ...] = DEFAULT_CLUSTERS,
    seed: int = DEFAULT_SEED,
) -> PointLayout:
    """Generate a point layout from a sequence of cluster specifications.

    Points are drawn from isotropic 2-D Gaussian distributions defined by
    each :class:`ClusterSpec`. The resulting arrays are concatenated in
    cluster order.

    Args:
        clusters: One or more ClusterSpec definitions.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`PointLayout` containing the generated points and metadata.

    Example:
        >>> layout = make_layout()
        >>> layout.n_points
        30
        >>> layout.n_clusters
        2
    """
    rng = np.random.default_rng(seed)

    all_points: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for cluster_id, spec in enumerate(clusters):
        pts = rng.normal(
            loc=spec.center,
            scale=spec.std,
            size=(spec.n_points, 2),
        )
        all_points.append(pts)
        all_labels.append(np.full(spec.n_points, cluster_id, dtype=np.intp))

    return PointLayout(
        points=np.concatenate(all_points, axis=0),
        labels=np.concatenate(all_labels, axis=0),
        cluster_specs=clusters,
        seed=seed,
    )


def make_two_cluster_layout(seed: int = DEFAULT_SEED) -> PointLayout:
    """Convenience function: generate the canonical two-cluster layout.

    - **Cluster A (small):** 10 points, center (-2, 0), σ = 0.5
    - **Cluster B (large):** 20 points, center (2, 0), σ = 0.8

    Args:
        seed: Random seed (default 42).

    Returns:
        A :class:`PointLayout` with 30 points in 2 clusters.
    """
    return make_layout(clusters=DEFAULT_CLUSTERS, seed=seed)


def make_single_cluster_layout(
    n_points: int = 30,
    center: tuple[float, float] = (0.0, 0.0),
    std: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> PointLayout:
    """Generate a single isotropic Gaussian cluster.

    Useful for algorithms where cluster separation is irrelevant.

    Args:
        n_points: Number of points.
        center: Cluster center.
        std: Standard deviation.
        seed: Random seed.

    Returns:
        A :class:`PointLayout` with one cluster.
    """
    spec = ClusterSpec(n_points=n_points, center=center, std=std, label="Single cluster")
    return make_layout(clusters=(spec,), seed=seed)


def make_uniform_layout(
    n_points: int = 30,
    xlim: tuple[float, float] = (-4.0, 4.0),
    ylim: tuple[float, float] = (-3.0, 3.0),
    seed: int = DEFAULT_SEED,
) -> PointLayout:
    """Generate points uniformly distributed in a rectangle.

    Useful for testing algorithms on non-clustered data.

    Args:
        n_points: Number of points.
        xlim: (min, max) range for x-coordinates.
        ylim: (min, max) range for y-coordinates.
        seed: Random seed.

    Returns:
        A :class:`PointLayout` with one "cluster" of uniformly distributed points.
    """
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        rng.uniform(xlim[0], xlim[1], size=n_points),
        rng.uniform(ylim[0], ylim[1], size=n_points),
    ])
    spec = ClusterSpec(
        n_points=n_points,
        center=((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2),
        std=0.0,
        label="Uniform",
    )
    return PointLayout(
        points=pts,
        labels=np.zeros(n_points, dtype=np.intp),
        cluster_specs=(spec,),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Pairwise distance utilities (handy for many graph builders)
# ---------------------------------------------------------------------------

def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Compute the full pairwise Euclidean distance matrix.

    Args:
        points: (n, d) array of point coordinates.

    Returns:
        (n, n) symmetric distance matrix.
    """
    # Efficient computation avoiding scipy dependency in the core module
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def k_nearest_indices(
    dist_matrix: np.ndarray,
    k: int,
    exclude_self: bool = True,
) -> np.ndarray:
    """Find the k nearest neighbor indices for each point.

    Args:
        dist_matrix: (n, n) pairwise distance matrix.
        k: Number of neighbors.
        exclude_self: If True, a point is not its own neighbor.

    Returns:
        (n, k) integer array of neighbor indices.
    """
    n = dist_matrix.shape[0]
    if exclude_self:
        # Set diagonal to infinity so a point doesn't pick itself
        dm = dist_matrix.copy()
        np.fill_diagonal(dm, np.inf)
    else:
        dm = dist_matrix
    # argpartition is O(n) per row vs O(n log n) for full sort
    indices = np.argpartition(dm, k, axis=1)[:, :k]
    # Sort the k neighbors by distance for consistent ordering
    row_idx = np.arange(n)[:, np.newaxis]
    sorted_order = np.argsort(dm[row_idx, indices], axis=1)
    return indices[row_idx, sorted_order]

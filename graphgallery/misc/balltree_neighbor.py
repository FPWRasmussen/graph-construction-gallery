"""
Ball Tree Neighbor Graph.

A ball tree is a binary tree where each node represents a
hyper-sphere ("ball") containing a subset of the data points.
Child nodes partition the parent's points into two sub-balls.

Compared to KD-trees:
    - Ball trees handle higher dimensions better (no axis-aligned splits)
    - Ball trees work with arbitrary distance metrics
    - KD-trees are faster in low dimensions (d ≤ 20)
    - Ball trees degrade more gracefully in high dimensions

Construction:
    1. Build a ball tree from the input points.
    2. For each point, query the tree for its k nearest neighbors.
    3. Create edges to these neighbors.

Like the KD-tree builder, this produces the same graph as brute-force
k-NN but with better scaling.  The builder also supports radius
queries (all neighbors within distance r).

Reference:
    Omohundro, S.M. (1989). "Five Balltree Construction Algorithms."
    Technical Report, ICSI Berkeley.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class BallTreeNeighborGraph(GraphBuilder):
    """k-NN or radius graph via ball tree spatial indexing.

    Supports both k-NN queries and radius (range) queries.

    Parameters:
        k: Number of nearest neighbors (for knn mode).
        radius: Query radius (for radius mode).
        mode: ``"knn"`` or ``"radius"``.
        leaf_size: Ball tree leaf size.
    """

    slug = "balltree_neighbor"
    category = "misc"

    def __init__(
        self,
        k: int = 5,
        radius: float = 1.0,
        mode: Literal["knn", "radius"] = "knn",
        leaf_size: int = 10,
    ):
        self.k = k
        self.radius = radius
        self.mode = mode
        self.leaf_size = leaf_size

    @property
    def name(self) -> str:
        return "Ball Tree Neighbor Graph"

    @property
    def description(self) -> str:
        if self.mode == "knn":
            return (
                f"k-NN (k={self.k}) via ball tree. Handles arbitrary "
                f"metrics and higher dimensions."
            )
        return (
            f"Radius graph (r={self.radius}) via ball tree range queries."
        )

    @property
    def complexity(self) -> str:
        return "O(n log n) build + O(n log n) queries"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("k", "Nearest neighbors (knn mode)", "int", 5, "k ≥ 1"),
            ParamInfo("radius", "Query radius (radius mode)", "float", 1.0, "> 0"),
            ParamInfo("mode", "Query mode", "str", "knn", "knn | radius"),
            ParamInfo("leaf_size", "Ball tree leaf size", "int", 10, "≥ 1"),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if self.mode == "knn" and self.k >= layout.n_points:
            raise ValueError(
                f"k={self.k} must be < n_points={layout.n_points}"
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points

        try:
            from sklearn.neighbors import BallTree
            return self._build_sklearn(points, n)
        except ImportError:
            return self._build_fallback(points, n, layout)

    def _build_sklearn(self, points: np.ndarray, n: int) -> nx.Graph:
        """Use sklearn's BallTree implementation."""
        from sklearn.neighbors import BallTree

        tree = BallTree(points, leaf_size=self.leaf_size)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        if self.mode == "knn":
            distances, indices = tree.query(points, k=self.k + 1)
            for i in range(n):
                for j_pos in range(1, self.k + 1):
                    j = int(indices[i, j_pos])
                    d = float(distances[i, j_pos])
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=d)

        elif self.mode == "radius":
            ind, dist = tree.query_radius(
                points, r=self.radius, return_distance=True
            )
            for i in range(n):
                for j, d in zip(ind[i], dist[i]):
                    j = int(j)
                    if j != i and not G.has_edge(i, j):
                        G.add_edge(i, j, weight=float(d))

        G.graph["algorithm"] = "balltree_neighbor"
        G.graph["mode"] = self.mode
        G.graph["implementation"] = "sklearn"

        return G

    def _build_fallback(
        self, points: np.ndarray, n: int, layout: PointLayout
    ) -> nx.Graph:
        """Fallback: use scipy KDTree (similar performance for 2D)."""
        import warnings
        from scipy.spatial import KDTree

        warnings.warn(
            "sklearn not available; falling back to scipy KDTree.",
            stacklevel=3,
        )

        tree = KDTree(points, leafsize=self.leaf_size)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        if self.mode == "knn":
            distances, indices = tree.query(points, k=self.k + 1)
            for i in range(n):
                for j_pos in range(1, self.k + 1):
                    j = int(indices[i, j_pos])
                    d = float(distances[i, j_pos])
                    if not G.has_edge(i, j):
                        G.add_edge(i, j, weight=d)

        elif self.mode == "radius":
            results = tree.query_ball_point(points, r=self.radius)
            dist = pairwise_distances(points)
            for i in range(n):
                for j in results[i]:
                    j = int(j)
                    if j != i and not G.has_edge(i, j):
                        G.add_edge(i, j, weight=float(dist[i, j]))

        G.graph["algorithm"] = "balltree_neighbor"
        G.graph["mode"] = self.mode
        G.graph["implementation"] = "scipy_kdtree_fallback"

        return G

"""
Weighted (Regular) Triangulation.

A regular triangulation generalizes the Delaunay triangulation to
*weighted* point sets.  Each point p_i carries a weight w_i, and the
distance metric is replaced by the "power distance":

    pow(p_i, p_j) = ‖p_i − p_j‖² − w_i − w_j

An edge (i, j) appears in the regular triangulation iff there exists
an empty "power sphere" (weighted circumsphere) passing through p_i
and p_j.

The standard algorithm lifts the problem to (d+1)-dimensional space:
    - Map each 2D point (x, y) with weight w to the 3D point
      (x, y, x² + y² − w) on a shifted paraboloid.
    - Compute the 3D convex hull of the lifted points.
    - Project the *lower* facets (outward normal pointing downward)
      back to 2D.

When all weights are equal, this reduces to the standard Delaunay
triangulation.

Reference:
    Edelsbrunner, H. (1992). "Weighted Voronoi Diagrams and
    Regular Triangulations." Technical Report, University of Illinois.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class WeightedTriangulationGraph(GraphBuilder):
    """Regular triangulation via the paraboloid lifting trick.

    Each point is assigned a weight.  The resulting triangulation uses
    "power distance" instead of Euclidean distance.  When all weights
    are equal this is identical to the Delaunay triangulation.

    Parameters:
        weights: Optional (n,) array of per-point weights.  If None,
            weights are generated based on cluster membership
            (higher weight for the larger cluster).
        weight_scale: Multiplier for auto-generated weights.
    """

    slug = "weighted_triangulation"
    category = "triangulation"

    def __init__(
        self,
        weights: np.ndarray | None = None,
        weight_scale: float = 0.3,
    ):
        self.weights = weights
        self.weight_scale = weight_scale

    @property
    def name(self) -> str:
        return "Weighted (Regular) Triangulation"

    @property
    def description(self) -> str:
        return (
            "Delaunay generalization with per-point weights via the "
            "paraboloid lifting method."
        )

    @property
    def complexity(self) -> str:
        return "O(n log n) via 3D convex hull"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "weights",
                "Per-point weights (n,) or None for auto",
                "ndarray | None",
                None,
            ),
            ParamInfo(
                "weight_scale",
                "Scale factor for auto-generated weights",
                "float",
                0.3,
                "> 0",
            ),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if layout.n_points < 3:
            raise ValueError(
                "Weighted triangulation requires at least 3 points."
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        # Resolve weights
        weights = self._resolve_weights(layout)

        # --- Paraboloid lifting ---
        # Lift (x, y) with weight w to (x, y, x² + y² − w)
        x, y = points[:, 0], points[:, 1]
        z_lifted = x**2 + y**2 - weights
        lifted_points = np.column_stack([x, y, z_lifted])

        # Compute 3D convex hull
        try:
            hull = ConvexHull(lifted_points)
        except Exception as e:
            raise ValueError(
                f"Convex hull computation failed (points may be "
                f"degenerate): {e}"
            ) from e

        # Extract lower facets: outward normal has negative z-component
        # scipy ConvexHull equations: each row is [a, b, c, d] where
        # ax + by + cz + d ≤ 0 for all hull points (inward normals).
        # A "lower" facet has c > 0 (inward normal z-component > 0,
        # meaning the outward normal points downward).
        edges: set[tuple[int, int]] = set()
        for i_facet, simplex in enumerate(hull.simplices):
            normal_z = hull.equations[i_facet, 2]
            if normal_z > 0:  # Lower facet
                for a in range(3):
                    for b in range(a + 1, 3):
                        u, v = int(simplex[a]), int(simplex[b])
                        edges.add((min(u, v), max(u, v)))

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for u, v in edges:
            G.add_edge(u, v, weight=float(dist[u, v]))

        # Store weights and power distances as graph/edge metadata
        G.graph["point_weights"] = weights.copy()
        for u, v in G.edges():
            power_dist = float(dist[u, v]) ** 2 - weights[u] - weights[v]
            G[u][v]["power_distance"] = power_dist

        return G

    def _resolve_weights(self, layout: PointLayout) -> np.ndarray:
        """Determine per-point weights.

        If explicit weights are provided, validate and return them.
        Otherwise, generate weights based on cluster membership:
        points in larger clusters get higher weights (simulating
        larger "radius of influence").
        """
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float64)
            if w.shape != (layout.n_points,):
                raise ValueError(
                    f"Expected weights shape ({layout.n_points},), "
                    f"got {w.shape}"
                )
            return w

        # Auto-generate: weight proportional to cluster size
        rng = np.random.default_rng(layout.seed + 1)
        weights = np.zeros(layout.n_points, dtype=np.float64)

        for c_id in range(layout.n_clusters):
            mask = layout.labels == c_id
            cluster_size = mask.sum()
            base_weight = (cluster_size / layout.n_points) * self.weight_scale
            # Add small per-point variation
            weights[mask] = base_weight + rng.uniform(
                -0.05 * self.weight_scale,
                0.05 * self.weight_scale,
                size=cluster_size,
            )

        return weights

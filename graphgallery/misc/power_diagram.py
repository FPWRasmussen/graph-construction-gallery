"""
Power Diagram Graph.

A power diagram (also called a Laguerre–Voronoi diagram or weighted
Voronoi diagram) generalizes the Voronoi diagram to weighted point
sets.  Each site p_i carries a weight w_i, and the "power distance"
from a point x to site p_i is:

    pow(x, p_i) = ‖x − p_i‖² − w_i

The power cell of p_i is the set of all points x where p_i has the
smallest power distance.  The dual graph of the power diagram
connects two sites whose power cells share a boundary.

When all weights are equal, the power diagram reduces to the
standard Voronoi diagram.  Unequal weights shift cell boundaries
toward lighter-weight sites, effectively giving heavier sites
larger cells.

The dual of the power diagram is the **regular (weighted) Delaunay
triangulation** — the same as our weighted triangulation builder,
approached from the Voronoi side.

Applications:
    - Optimal transport and Wasserstein distances
    - Centroidal Voronoi tessellations with variable density
    - Wireless network coverage with variable power levels
    - Crystal growth simulation (Laguerre geometry)

Reference:
    Aurenhammer, F. (1987). "Power diagrams: properties, algorithms
    and applications." SIAM Journal on Computing, 16(1), 78–96.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class PowerDiagramGraph(GraphBuilder):
    """Dual graph of the power (weighted Voronoi) diagram.

    Connects sites whose power cells share a boundary.  Computed
    via the paraboloid lifting method (same as weighted triangulation).

    Parameters:
        weights: (n,) per-site weights.  None → auto-generate
            based on cluster membership.
        weight_scale: Multiplier for auto-generated weights.
    """

    slug = "power_diagram"
    category = "misc"

    def __init__(
        self,
        weights: np.ndarray | None = None,
        weight_scale: float = 0.5,
    ):
        self.weights = weights
        self.weight_scale = weight_scale

    @property
    def name(self) -> str:
        return "Power Diagram Graph"

    @property
    def description(self) -> str:
        return (
            "Dual of the weighted Voronoi diagram (Laguerre tessellation). "
            "Generalizes Voronoi dual with per-site weights."
        )

    @property
    def complexity(self) -> str:
        return "O(n log n) via 3D convex hull"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "weights", "Per-site weights or None for auto",
                "ndarray | None", None,
            ),
            ParamInfo(
                "weight_scale", "Scale for auto-generated weights",
                "float", 0.5, "> 0",
            ),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if layout.n_points < 3:
            raise ValueError("Power diagram requires at least 3 points.")

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        # Resolve weights
        weights = self._resolve_weights(layout)

        # Paraboloid lifting: (x, y) with weight w → (x, y, x²+y²-w)
        x, y = points[:, 0], points[:, 1]
        z_lifted = x ** 2 + y ** 2 - weights
        lifted = np.column_stack([x, y, z_lifted])

        # 3D convex hull
        try:
            hull = ConvexHull(lifted)
        except Exception as e:
            raise ValueError(
                f"Convex hull failed (degenerate points?): {e}"
            ) from e

        # Extract lower facets (outward normal z < 0 → inward normal z > 0)
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
            # Store both Euclidean and power distance
            eucl = float(dist[u, v])
            power_dist = eucl ** 2 - weights[u] - weights[v]
            G.add_edge(u, v, weight=eucl, power_distance=float(power_dist))

        G.graph["algorithm"] = "power_diagram"
        G.graph["weights"] = weights.tolist()

        return G

    def _resolve_weights(self, layout: PointLayout) -> np.ndarray:
        """Determine per-site weights."""
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float64)
            if w.shape != (layout.n_points,):
                raise ValueError(
                    f"Weights shape {w.shape} != ({layout.n_points},)"
                )
            return w

        # Auto-generate: larger weight for denser cluster (more influence)
        rng = np.random.default_rng(layout.seed + 400)
        weights = np.zeros(layout.n_points, dtype=np.float64)

        for c_id in range(layout.n_clusters):
            mask = layout.labels == c_id
            cluster_size = mask.sum()
            # Larger clusters get larger weights → larger cells
            base = (cluster_size / layout.n_points) * self.weight_scale
            weights[mask] = base + rng.uniform(
                -0.05 * self.weight_scale,
                0.05 * self.weight_scale,
                size=cluster_size,
            )

        return weights

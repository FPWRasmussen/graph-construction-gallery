"""
Theta (Θ) Graph.

The Theta graph is similar to the Yao graph but uses **projections**
onto cone bisectors instead of raw Euclidean distance to select the
nearest neighbor in each cone.

For each point p_i:
    1. Divide the plane into k equal cones of angle 2π/k.
    2. For each cone c with bisector direction b_c:
       a. Project every other point p_j in cone c onto b_c.
       b. Select the point with the smallest projection distance.
       c. Add edge (p_i, p_j) with weight = Euclidean distance.

Using projections instead of distances ensures that the selected
neighbor lies "in the direction" of the cone bisector, which gives
tighter theoretical bounds on the stretch factor.

Key properties:
    - At most k·n edges
    - For k ≥ 7 (odd) or k ≥ 8 (even), it is a t-spanner with
      t = 1 / (cos(π/k) − sin(π/k))
    - The half-Theta graph (Θ_{k/2}) has exactly the same stretch
      factor as the Yao graph but admits simpler analysis
    - Can be computed in O(n log n) using range tree methods

Reference:
    Clarkson, K.L. (1987). "Approximation algorithms for shortest
    path motion planning." Proc. 19th ACM STOC, pp. 56–65.

    Ruppert, J. & Seidel, R. (1991). "Approximating the d-dimensional
    complete Euclidean graph." Proc. 3rd Canadian Conference on
    Computational Geometry, pp. 207–210.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.spanners._spanner_utils import (
    cone_partition_angles,
    angle_to_point,
    assign_cone,
    compute_stretch_factor,
)


class ThetaGraph(GraphBuilder):
    """Cone-based spanner using projection distance instead of Euclidean.

    Parameters:
        k: Number of angular cones. Higher k = better stretch bound.
    """

    slug = "theta"
    category = "spanners"

    def __init__(self, k: int = 6):
        if k < 2:
            raise ValueError(f"Number of cones k must be ≥ 2, got {k}")
        self.k = k

    @property
    def name(self) -> str:
        return "Theta (Θ) Graph"

    @property
    def description(self) -> str:
        half_angle = np.pi / self.k
        cos_a, sin_a = np.cos(half_angle), np.sin(half_angle)
        denom = cos_a - sin_a
        if denom > 0:
            t = 1.0 / denom
            return (
                f"Projection-based spanner with {self.k} cones. "
                f"Stretch ≤ {t:.2f}."
            )
        return f"Projection-based spanner with {self.k} cones."

    @property
    def complexity(self) -> str:
        return "O(kn²) naïve, O(n log n) with range trees"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "k", "Number of angular cones",
                "int", 6, "k ≥ 7 guarantees spanner property",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        cone_bounds = cone_partition_angles(self.k)

        # Precompute cone bisector unit vectors
        bisectors = np.zeros((self.k, 2), dtype=np.float64)
        for c in range(self.k):
            bisector_angle = (cone_bounds[c] + cone_bounds[c + 1]) / 2.0
            bisectors[c] = [np.cos(bisector_angle), np.sin(bisector_angle)]

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            # For each cone, find the neighbor with smallest projection
            # onto the cone bisector
            best_in_cone: dict[int, tuple[float, float, int]] = {}
            # cone_id → (projection_dist, euclidean_dist, neighbor_index)

            for j in range(n):
                if j == i:
                    continue

                diff = points[j] - points[i]
                angle = angle_to_point(points[i], points[j])
                cone_id = assign_cone(angle, cone_bounds)

                # Project diff onto the cone bisector
                proj_dist = float(np.dot(diff, bisectors[cone_id]))

                if proj_dist <= 0:
                    # Point is behind the origin relative to this bisector
                    # This shouldn't normally happen for the correct cone,
                    # but can for edge cases near cone boundaries
                    continue

                eucl_dist = float(dist[i, j])

                if (
                    cone_id not in best_in_cone
                    or proj_dist < best_in_cone[cone_id][0]
                ):
                    best_in_cone[cone_id] = (proj_dist, eucl_dist, j)

            # Add edges
            for cone_id, (proj_d, eucl_d, j) in best_in_cone.items():
                if not G.has_edge(i, j):
                    G.add_edge(
                        i, j,
                        weight=eucl_d,
                        projection_distance=proj_d,
                        cone=cone_id,
                    )

        G.graph["algorithm"] = "theta"
        G.graph["k"] = self.k
        G.graph["actual_stretch"] = compute_stretch_factor(G, dist)

        return G

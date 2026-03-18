"""
Yao Graph.

The Yao graph (also called Yao_k) partitions the plane around each
point into k equal angular cones (sectors of angle 2π/k), and
connects each point to its nearest neighbor within each cone.

For each point p_i:
    1. Divide the plane into k cones of angle 2π/k centered at p_i.
    2. In each non-empty cone, find the closest other point p_j.
    3. Add a directed edge p_i → p_j.

The undirected Yao graph takes the union of all directed edges.

Key properties:
    - At most k·n directed edges (k per node)
    - For k ≥ 9, the Yao graph is a t-spanner with
      t = 1 / (1 − 2·sin(π/k))
    - Sparser than Delaunay for large k
    - O(n² d) construction, O(n log n) possible with range trees

The Yao graph was introduced by Andrew Yao in 1982 and is one of
the foundational constructions in the theory of geometric spanners.

Reference:
    Yao, A.C.-C. (1982). "On constructing minimum spanning trees
    in k-dimensional spaces and related problems." SIAM Journal on
    Computing, 11(4), 721–736.
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


class YaoGraph(GraphBuilder):
    """Nearest neighbor per angular cone → sparse geometric spanner.

    Parameters:
        k: Number of cones (sectors). Higher k = sparser graph,
           lower stretch factor.
    """

    slug = "yao"
    category = "spanners"

    def __init__(self, k: int = 6):
        if k < 2:
            raise ValueError(f"Number of cones k must be ≥ 2, got {k}")
        self.k = k

    @property
    def name(self) -> str:
        return "Yao Graph"

    @property
    def description(self) -> str:
        theta = 2 * np.pi / self.k
        if self.k >= 9:
            t = 1.0 / (1.0 - 2.0 * np.sin(np.pi / self.k))
            return (
                f"Nearest neighbor in each of {self.k} cones "
                f"(θ={np.degrees(theta):.0f}°). Stretch ≤ {t:.2f}."
            )
        return (
            f"Nearest neighbor in each of {self.k} cones "
            f"(θ={np.degrees(theta):.0f}°)."
        )

    @property
    def complexity(self) -> str:
        return "O(kn²) naïve, O(n log n) with range trees"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "k", "Number of angular cones",
                "int", 6, "k ≥ 2; k ≥ 9 guarantees spanner property",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        cone_bounds = cone_partition_angles(self.k)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            # For each cone, find the nearest neighbor
            best_in_cone: dict[int, tuple[float, int]] = {}
            # cone_id → (distance, neighbor_index)

            for j in range(n):
                if j == i:
                    continue

                angle = angle_to_point(points[i], points[j])
                cone_id = assign_cone(angle, cone_bounds)
                d_ij = float(dist[i, j])

                if cone_id not in best_in_cone or d_ij < best_in_cone[cone_id][0]:
                    best_in_cone[cone_id] = (d_ij, j)

            # Add edges to nearest neighbor in each cone
            for cone_id, (d_ij, j) in best_in_cone.items():
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=d_ij, cone=cone_id)

        G.graph["algorithm"] = "yao"
        G.graph["k"] = self.k
        G.graph["actual_stretch"] = compute_stretch_factor(G, dist)

        return G

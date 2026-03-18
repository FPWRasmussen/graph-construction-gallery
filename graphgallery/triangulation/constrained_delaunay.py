"""
Constrained Delaunay Triangulation (CDT).

A constrained Delaunay triangulation honours a set of *required edges*
(constraints) that must appear in the final triangulation, even if they
would not appear in the standard Delaunay triangulation.  The
triangulation is "as Delaunay as possible" given the constraints: every
triangle whose circumcircle interior contains no visible point (visible
across the constraint segments) satisfies the Delaunay criterion.

Use cases:
    - Meshing domains with predefined boundaries
    - Preserving known edges (e.g. inter-cluster bridges)
    - Terrain / GIS triangulation with breaklines

This implementation uses Shewchuk's Triangle library via the ``triangle``
Python package.  If ``triangle`` is not installed, a fallback injects
the constraint edges into the standard scipy Delaunay result.

Reference:
    Shewchuk, J.R. (1996). "Triangle: Engineering a 2D Quality Mesh
    Generator and Delaunay Triangulator." Applied Computational
    Geometry: Towards Geometric Engineering, LNCS 1148, pp. 203–222.
"""

from __future__ import annotations

from typing import Optional, Sequence

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


def _default_constraints(layout: PointLayout) -> np.ndarray:
    """Generate sensible default constraint edges.

    Strategy:
        - If the layout has ≥ 2 clusters, connect the closest pair
          between each pair of clusters (inter-cluster bridges).
        - Add edges along the convex hull of each cluster.

    Args:
        layout: The point layout.

    Returns:
        (m, 2) integer array of constrained edge endpoint indices.
    """
    dist = pairwise_distances(layout.points)
    constraints: list[tuple[int, int]] = []

    if layout.n_clusters >= 2:
        # Inter-cluster bridges: for each pair of clusters, find
        # the 2 closest cross-cluster point pairs
        for c_a in range(layout.n_clusters):
            for c_b in range(c_a + 1, layout.n_clusters):
                mask_a = layout.labels == c_a
                mask_b = layout.labels == c_b
                idx_a = np.where(mask_a)[0]
                idx_b = np.where(mask_b)[0]

                # Cross-cluster distance sub-matrix
                sub_dist = dist[np.ix_(idx_a, idx_b)]

                # Find the 2 shortest cross-cluster edges
                n_bridges = min(2, sub_dist.size)
                flat_indices = np.argpartition(
                    sub_dist.ravel(), n_bridges
                )[:n_bridges]

                for flat_idx in flat_indices:
                    r, c = np.unravel_index(flat_idx, sub_dist.shape)
                    u, v = int(idx_a[r]), int(idx_b[c])
                    constraints.append((min(u, v), max(u, v)))

    # Per-cluster convex hull edges
    from scipy.spatial import ConvexHull

    for c_id in range(layout.n_clusters):
        cluster_global_idx = np.where(layout.labels == c_id)[0]
        if len(cluster_global_idx) < 3:
            continue
        pts = layout.points[cluster_global_idx]
        try:
            hull = ConvexHull(pts)
            for simplex in hull.simplices:
                u = int(cluster_global_idx[simplex[0]])
                v = int(cluster_global_idx[simplex[1]])
                constraints.append((min(u, v), max(u, v)))
        except Exception:
            pass

    # Deduplicate
    constraints = list(set(constraints))

    if not constraints:
        # Fallback: just the single closest pair
        np.fill_diagonal(dist, np.inf)
        flat = np.argmin(dist)
        i, j = np.unravel_index(flat, dist.shape)
        constraints = [(min(int(i), int(j)), max(int(i), int(j)))]

    return np.array(constraints, dtype=np.intp)


def _try_triangle_cdt(
    points: np.ndarray,
    constraint_edges: np.ndarray,
) -> set[tuple[int, int]] | None:
    """Attempt CDT via the ``triangle`` library.

    Returns:
        Set of (i, j) edges with i < j, or None if ``triangle``
        is not available.
    """
    try:
        import triangle as tr
    except ImportError:
        return None

    planar_data = {
        "vertices": points.astype(np.float64),
        "segments": constraint_edges.astype(np.int32),
    }

    # 'p' = triangulate, include all points
    # No quality flags — pure CDT
    result = tr.triangulate(planar_data, "p")

    edges: set[tuple[int, int]] = set()
    for simplex in result.get("triangles", []):
        for a in range(3):
            for b in range(a + 1, 3):
                u, v = int(simplex[a]), int(simplex[b])
                # Only include edges that reference original points
                if u < len(points) and v < len(points):
                    edges.add((min(u, v), max(u, v)))

    return edges


def _fallback_cdt(
    points: np.ndarray,
    constraint_edges: np.ndarray,
) -> set[tuple[int, int]]:
    """Fallback CDT: Delaunay triangulation + forced constraint edges.

    Not a true CDT, but ensures required edges are present.
    """
    from graphgallery.triangulation.delaunay import delaunay_edges

    edges = delaunay_edges(points)

    # Force constraint edges
    for row in constraint_edges:
        u, v = int(row[0]), int(row[1])
        edges.add((min(u, v), max(u, v)))

    return edges


class ConstrainedDelaunayGraph(GraphBuilder):
    """Delaunay triangulation with required (constrained) edges.

    Certain edges are forced to appear in the triangulation. The
    algorithm produces a triangulation that is as close to Delaunay
    as possible while honouring these constraints.

    If the ``triangle`` library is installed, a true CDT is computed.
    Otherwise, constraint edges are injected into the standard Delaunay
    triangulation as a fallback.

    Parameters:
        constraint_edges: Optional (m, 2) array of required edges.
            If None, sensible defaults are generated (inter-cluster
            bridges + per-cluster convex hulls).
    """

    slug = "constrained_delaunay"
    category = "triangulation"

    def __init__(
        self,
        constraint_edges: np.ndarray | None = None,
    ):
        self.constraint_edges = constraint_edges

    @property
    def name(self) -> str:
        return "Constrained Delaunay"

    @property
    def description(self) -> str:
        return (
            "Delaunay triangulation with required edges that must appear. "
            "Uses Shewchuk's Triangle when available."
        )

    @property
    def complexity(self) -> str:
        return "O(n log n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "constraint_edges",
                "Required edges as (m, 2) array or None for auto",
                "ndarray | None",
                None,
                "Each row is (i, j) with 0 ≤ i, j < n",
            ),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if layout.n_points < 3:
            raise ValueError(
                "Constrained Delaunay requires at least 3 points."
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        # Resolve constraint edges
        constraints = (
            self.constraint_edges
            if self.constraint_edges is not None
            else _default_constraints(layout)
        )

        # Try true CDT via triangle, fall back to injection
        edges = _try_triangle_cdt(points, constraints)
        if edges is None:
            edges = _fallback_cdt(points, constraints)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v in edges:
            is_constraint = any(
                (int(constraints[r, 0]) == min(u, v)
                 and int(constraints[r, 1]) == max(u, v))
                for r in range(len(constraints))
            )
            G.add_edge(
                u, v,
                weight=float(dist[u, v]),
                constrained=is_constraint,
            )

        return G

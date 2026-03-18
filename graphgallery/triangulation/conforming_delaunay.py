"""
Conforming Delaunay Triangulation.

A conforming Delaunay triangulation (ConfDT) satisfies both:
    1. All constraint edges are present in the triangulation.
    2. Every triangle in the mesh is truly Delaunay (empty circumcircle).

To achieve both properties simultaneously, *Steiner points* are
inserted along constraint edges or in the interior.  The resulting
mesh typically has more points than the original input.

Quality refinement can also impose a minimum angle constraint
(Ruppert's algorithm), producing meshes suitable for finite element
analysis.

This implementation uses Shewchuk's Triangle library.  If unavailable,
it falls back to a basic Delaunay triangulation of the original points
(no Steiner points).

Reference:
    Ruppert, J. (1995). "A Delaunay Refinement Algorithm for Quality
    2-Dimensional Mesh Generation." Journal of Algorithms, 18(3).
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout


class ConformingDelaunayGraph(GraphBuilder):
    """Delaunay triangulation with Steiner points for full conformity.

    Adds extra points so that constraint edges are present AND every
    triangle satisfies the Delaunay criterion.  Quality refinement
    can enforce a minimum angle.

    Parameters:
        min_angle: Minimum angle constraint in degrees for Ruppert's
            refinement.  Set to 0 to disable quality refinement.
        max_area: Maximum triangle area.  Set to 0 for no constraint.
    """

    slug = "conforming_delaunay"
    category = "triangulation"

    def __init__(
        self,
        min_angle: float = 20.0,
        max_area: float = 0.0,
    ):
        self.min_angle = min_angle
        self.max_area = max_area

    @property
    def name(self) -> str:
        return "Conforming Delaunay"

    @property
    def description(self) -> str:
        return (
            "Adds Steiner points to achieve full Delaunay property "
            "with constraint edges. Supports quality refinement."
        )

    @property
    def complexity(self) -> str:
        return "O(n log n) + refinement iterations"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "min_angle",
                "Minimum angle in degrees (Ruppert refinement)",
                "float",
                20.0,
                "0 ≤ angle ≤ 33 (theoretical max for guaranteed termination)",
            ),
            ParamInfo(
                "max_area",
                "Maximum triangle area (0 = no limit)",
                "float",
                0.0,
                "≥ 0",
            ),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if layout.n_points < 3:
            raise ValueError(
                "Conforming Delaunay requires at least 3 points."
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        """Build a conforming Delaunay triangulation.

        The returned graph may contain **more nodes** than the input
        layout due to Steiner point insertion.  Steiner nodes have
        indices ≥ ``layout.n_points``.  Their coordinates are stored
        in the graph attribute ``G.graph["all_points"]``.

        Returns:
            nx.Graph with attributes:
                - ``G.graph["all_points"]``: (m, 2) array including
                  Steiner points.
                - ``G.graph["n_original"]``: number of original points.
                - ``G.graph["n_steiner"]``: number of added Steiner points.
                - Each node has attribute ``steiner=True/False``.
        """
        self.validate_layout(layout)
        points = layout.points
        n_original = layout.n_points

        try:
            all_points, triangles = self._build_with_triangle(points)
        except ImportError:
            all_points, triangles = self._build_fallback(points)

        n_total = len(all_points)
        n_steiner = n_total - n_original

        # Extract unique edges from triangles
        edges: set[tuple[int, int]] = set()
        for simplex in triangles:
            for a in range(3):
                for b in range(a + 1, 3):
                    u, v = int(simplex[a]), int(simplex[b])
                    edges.add((min(u, v), max(u, v)))

        # Build graph
        G = nx.Graph()
        for i in range(n_total):
            G.add_node(i, steiner=(i >= n_original))

        for u, v in edges:
            d = float(np.linalg.norm(all_points[u] - all_points[v]))
            G.add_edge(u, v, weight=d)

        # Store metadata
        G.graph["all_points"] = all_points
        G.graph["n_original"] = n_original
        G.graph["n_steiner"] = n_steiner

        return G

    def _build_with_triangle(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use Shewchuk's Triangle for conforming refinement.

        Raises:
            ImportError: If ``triangle`` is not installed.
        """
        import triangle as tr

        # Compute convex hull as boundary segments
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        segments = []
        n_hull = len(hull.vertices)
        for i in range(n_hull):
            u = int(hull.vertices[i])
            v = int(hull.vertices[(i + 1) % n_hull])
            segments.append([u, v])

        planar_data = {
            "vertices": points.astype(np.float64),
            "segments": np.array(segments, dtype=np.int32),
        }

        # Build Triangle switch string
        # 'p' = PSLG mode (planar straight-line graph)
        # 'q' = quality refinement with minimum angle
        # 'D' = conforming Delaunay
        # 'a' = maximum area constraint
        switches = "pD"
        if self.min_angle > 0:
            switches += f"q{self.min_angle}"
        if self.max_area > 0:
            switches += f"a{self.max_area}"

        result = tr.triangulate(planar_data, switches)

        all_points = result["vertices"]
        triangles = result["triangles"]

        return all_points, triangles

    def _build_fallback(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: standard scipy Delaunay (no Steiner points)."""
        import warnings
        from scipy.spatial import Delaunay

        warnings.warn(
            "The 'triangle' package is not installed. Falling back to "
            "standard Delaunay triangulation (no Steiner points). "
            "Install with: pip install triangle",
            stacklevel=3,
        )

        tri = Delaunay(points)
        return points.copy(), tri.simplices

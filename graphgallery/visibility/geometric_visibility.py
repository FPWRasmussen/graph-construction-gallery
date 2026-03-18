"""
Geometric Visibility Graph.

In computational geometry, a visibility graph is constructed from a
set of points (and optionally obstacles) in the plane.  Two points
are connected by an edge if the straight-line segment between them
does not pass through any obstacle [[3]].

Without obstacles, every pair of points is visible and the result
is simply the complete graph.  The interesting case arises when
obstacles (polygons) are present: the visibility graph then encodes
which points can "see" each other, and shortest paths in this graph
correspond to shortest obstacle-avoiding paths in the plane.

Applications:
    - Robot motion planning
    - Shortest path with obstacles
    - Architectural analysis (isovist fields)
    - Surveillance and sensor placement

Construction:
    For each pair of points (p_i, p_j):
        Check whether the segment p_i—p_j intersects any obstacle edge.
        If no intersection → add the edge.

For the gallery, we generate small obstacles scattered between the
two clusters to create a visually interesting visibility structure.

Reference:
    de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008).
    "Computational Geometry: Algorithms and Applications." 3rd ed.,
    Springer. Chapter 15: Visibility Graphs.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.visibility._visibility_utils import (
    segments_intersect,
    generate_obstacles_from_layout,
)


class GeometricVisibilityGraph(GraphBuilder):
    """Connect points that can see each other past obstacles.

    If no obstacles are provided, generates random rectangular
    obstacles between the clusters for a visually interesting result.

    Parameters:
        obstacles: List of (m, 2) polygon arrays defining obstacles.
            If None, auto-generated from the layout.
        n_auto_obstacles: Number of auto-generated obstacles.
        obstacle_radius: Size of auto-generated obstacles.
        seed: Random seed for obstacle generation.
    """

    slug = "geometric_visibility"
    category = "visibility"

    def __init__(
        self,
        obstacles: list[np.ndarray] | None = None,
        n_auto_obstacles: int = 5,
        obstacle_radius: float = 0.3,
        seed: int = 42,
    ):
        self.obstacles = obstacles
        self.n_auto_obstacles = n_auto_obstacles
        self.obstacle_radius = obstacle_radius
        self.seed = seed

    @property
    def name(self) -> str:
        return "Geometric Visibility"

    @property
    def description(self) -> str:
        return (
            "Connect points with unobstructed line-of-sight. "
            "Used for shortest-path planning with obstacles."
        )

    @property
    def complexity(self) -> str:
        return "O(n² · E) where E = total obstacle edges"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "obstacles", "List of polygon obstacles or None for auto",
                "list[ndarray] | None", None,
            ),
            ParamInfo(
                "n_auto_obstacles", "Auto-generated obstacle count",
                "int", 5, "≥ 0",
            ),
            ParamInfo(
                "obstacle_radius", "Size of auto obstacles",
                "float", 0.3, "> 0",
            ),
            ParamInfo("seed", "Random seed", "int", 42),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        # Resolve obstacles
        obstacles = self._resolve_obstacles(layout)

        # Extract all obstacle edges (segments)
        obstacle_segments = self._extract_segments(obstacles)

        # Build visibility graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        n_visible = 0
        n_blocked = 0

        for i in range(n):
            for j in range(i + 1, n):
                if self._is_visible(points[i], points[j], obstacle_segments):
                    G.add_edge(i, j, weight=float(dist[i, j]))
                    n_visible += 1
                else:
                    n_blocked += 1

        # Store metadata
        G.graph["algorithm"] = "geometric_visibility"
        G.graph["n_obstacles"] = len(obstacles)
        G.graph["n_obstacle_edges"] = len(obstacle_segments)
        G.graph["n_visible_pairs"] = n_visible
        G.graph["n_blocked_pairs"] = n_blocked
        G.graph["obstacles"] = obstacles

        return G

    def _resolve_obstacles(
        self, layout: PointLayout
    ) -> list[np.ndarray]:
        """Get or generate obstacles."""
        if self.obstacles is not None:
            return self.obstacles

        if self.n_auto_obstacles <= 0:
            return []

        return generate_obstacles_from_layout(
            layout,
            n_obstacles=self.n_auto_obstacles,
            obstacle_radius=self.obstacle_radius,
            seed=self.seed,
        )

    @staticmethod
    def _extract_segments(
        obstacles: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Extract all edge segments from a list of obstacle polygons.

        Returns:
            List of (p1, p2) endpoint pairs for each obstacle edge.
        """
        segments = []
        for polygon in obstacles:
            m = len(polygon)
            for k in range(m):
                p1 = polygon[k]
                p2 = polygon[(k + 1) % m]
                segments.append((p1, p2))
        return segments

    @staticmethod
    def _is_visible(
        p_i: np.ndarray,
        p_j: np.ndarray,
        obstacle_segments: list[tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        """Test if two points can see each other (no obstacle blocks them).

        Args:
            p_i, p_j: The two points to test.
            obstacle_segments: All obstacle edge segments.

        Returns:
            True if the line segment p_i—p_j does not properly intersect
            any obstacle segment.
        """
        for seg_start, seg_end in obstacle_segments:
            if segments_intersect(p_i, p_j, seg_start, seg_end):
                return False
        return True

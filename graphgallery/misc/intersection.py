"""
Intersection Graph.

An intersection graph represents a collection of geometric objects
as nodes, with edges connecting objects that overlap (intersect).
This is one of the most fundamental concepts in computational
geometry and combinatorics.

Common variants:
    - **Circle intersection**: Each point gets a circle; edges if circles overlap
    - **Rectangle intersection**: Axis-aligned bounding boxes
    - **Interval intersection**: 1D intervals on each coordinate axis
    - **Convex hull intersection**: Convex hulls of point subsets

For our gallery, we assign each point a random geometric shape
(circle with random radius, or axis-aligned rectangle with random
extent) and connect overlapping shapes.

Many named graph classes are intersection graphs:
    - Unit disk graphs = intersection of equal-radius circles
    - Interval graphs = intersection of line segments
    - Chordal graphs = intersection of subtrees of a tree

Reference:
    McKee, T.A. & McMorris, F.R. (1999). "Topics in Intersection
    Graph Theory." SIAM Monographs on Discrete Mathematics.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class IntersectionGraph(GraphBuilder):
    """Connect nodes whose assigned geometric objects overlap.

    Parameters:
        shape: Type of geometric object per node.
            - ``"circle"``: Random-radius circles.
            - ``"rectangle"``: Random-extent axis-aligned rectangles.
        radius_mean: Mean radius/extent for shape generation.
        radius_std: Standard deviation of radius/extent.
        seed: Random seed for shape generation.
    """

    slug = "intersection"
    category = "misc"

    def __init__(
        self,
        shape: Literal["circle", "rectangle"] = "circle",
        radius_mean: float = 0.7,
        radius_std: float = 0.3,
        seed: int | None = None,
    ):
        self.shape = shape
        self.radius_mean = radius_mean
        self.radius_std = radius_std
        self.seed = seed

    @property
    def name(self) -> str:
        return "Intersection Graph"

    @property
    def description(self) -> str:
        return (
            f"Connect nodes whose {self.shape}s overlap. "
            f"Mean radius={self.radius_mean}."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n²)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "shape", "Geometric object type", "str", "circle",
                "circle | rectangle",
            ),
            ParamInfo("radius_mean", "Mean radius/extent", "float", 0.7, "> 0"),
            ParamInfo("radius_std", "Radius std deviation", "float", 0.3, "≥ 0"),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points

        seed = self.seed if self.seed is not None else (layout.seed + 401)
        rng = np.random.default_rng(seed)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        if self.shape == "circle":
            radii = np.abs(rng.normal(self.radius_mean, self.radius_std, size=n))
            radii = np.maximum(radii, 0.05)

            for i in range(n):
                G.nodes[i]["radius"] = float(radii[i])

            dist = pairwise_distances(points)
            for i in range(n):
                for j in range(i + 1, n):
                    if dist[i, j] <= radii[i] + radii[j]:
                        overlap = radii[i] + radii[j] - dist[i, j]
                        G.add_edge(
                            i, j,
                            weight=float(dist[i, j]),
                            overlap=float(max(0, overlap)),
                        )

        elif self.shape == "rectangle":
            # Half-extents along x and y
            extents_x = np.abs(rng.normal(
                self.radius_mean, self.radius_std, size=n
            ))
            extents_y = np.abs(rng.normal(
                self.radius_mean, self.radius_std, size=n
            ))
            extents_x = np.maximum(extents_x, 0.05)
            extents_y = np.maximum(extents_y, 0.05)

            for i in range(n):
                G.nodes[i]["extent_x"] = float(extents_x[i])
                G.nodes[i]["extent_y"] = float(extents_y[i])

            for i in range(n):
                for j in range(i + 1, n):
                    # AABB overlap test
                    dx = abs(points[i, 0] - points[j, 0])
                    dy = abs(points[i, 1] - points[j, 1])
                    sum_ex = extents_x[i] + extents_x[j]
                    sum_ey = extents_y[i] + extents_y[j]

                    if dx <= sum_ex and dy <= sum_ey:
                        overlap_x = max(0, sum_ex - dx)
                        overlap_y = max(0, sum_ey - dy)
                        overlap_area = overlap_x * overlap_y

                        d = float(np.linalg.norm(points[i] - points[j]))
                        G.add_edge(
                            i, j,
                            weight=d,
                            overlap_area=float(overlap_area),
                        )

        G.graph["algorithm"] = "intersection"
        G.graph["shape"] = self.shape

        return G

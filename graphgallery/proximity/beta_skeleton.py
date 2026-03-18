"""
β-Skeleton Graph.

A family of proximity graphs parameterized by β > 0 that interpolates
between dense and sparse connectivity:

    β → 0⁺ :  approaches the Complete Graph (all edges)
    β = 1  :  Gabriel Graph
    β = 2  :  Relative Neighborhood Graph
    β → ∞  :  approaches the empty graph (no edges)

For each pair (i, j), a "forbidden region" is defined. If no other
point falls inside it, the edge exists. The shape of the region
depends on β:

    β ≥ 1 (lune-based):
        Intersection of two balls of radius (β/2)·d(i,j), centered at
        points along the segment i–j.

    0 < β < 1 (lens-based):
        Intersection of two balls of radius d(i,j)/(2β), centered on
        the perpendicular bisector of i–j. The lens is thinner than
        the diametral disk, admitting more edges.

Reference:
    Kirkpatrick, D.G. & Radke, J.D. (1985). "A Framework for
    Computational Morphology." Computational Geometry.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class BetaSkeletonGraph(GraphBuilder):
    """Generalized proximity graph parameterized by β.

    Unifies the Gabriel graph (β=1) and RNG (β=2) into a single
    tunable family. Lower β yields denser graphs.

    Parameters:
        beta: Skeleton parameter. Must be > 0.
    """

    slug = "beta_skeleton"
    category = "proximity"

    def __init__(self, beta: float = 1.5):
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        self.beta = beta

    @property
    def name(self) -> str:
        return "β-Skeleton"

    @property
    def description(self) -> str:
        return (
            f"Generalized proximity graph with β={self.beta}. "
            f"β=1 → Gabriel, β=2 → RNG."
        )

    @property
    def complexity(self) -> str:
        return "O(n³)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "beta",
                "Skeleton parameter (1=Gabriel, 2=RNG)",
                "float",
                1.5,
                "β > 0",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                d_ij = dist[i, j]
                if d_ij < 1e-15:
                    # Skip degenerate pairs (coincident points)
                    continue

                c1, c2, radius = self._forbidden_region_params(
                    points[i], points[j], d_ij
                )

                # Edge exists iff no other point lies strictly inside
                # the intersection of the two balls
                region_empty = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    d_to_c1 = np.linalg.norm(points[k] - c1)
                    d_to_c2 = np.linalg.norm(points[k] - c2)
                    if d_to_c1 < radius and d_to_c2 < radius:
                        region_empty = False
                        break

                if region_empty:
                    G.add_edge(i, j, weight=float(d_ij))

        return G

    def _forbidden_region_params(
        self,
        p_i: np.ndarray,
        p_j: np.ndarray,
        d_ij: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute the centers and radius of the forbidden region's two balls.

        Args:
            p_i: Coordinates of point i.
            p_j: Coordinates of point j.
            d_ij: Distance between i and j.

        Returns:
            (c1, c2, radius) — centers of the two balls and their
            shared radius.
        """
        beta = self.beta

        if beta >= 1.0:
            # Lune-based: centers lie on the segment i–j
            #   c1 = (1 - β/2)·p_i + (β/2)·p_j
            #   c2 = (β/2)·p_i + (1 - β/2)·p_j
            #   R  = (β/2)·d_ij
            half_beta = beta / 2.0
            c1 = p_i * (1.0 - half_beta) + p_j * half_beta
            c2 = p_i * half_beta + p_j * (1.0 - half_beta)
            radius = half_beta * d_ij
        else:
            # Lens-based: centers lie on the perpendicular bisector
            #   Both balls pass through p_i and p_j
            #   R = d_ij / (2β)
            midpoint = (p_i + p_j) / 2.0
            direction = p_j - p_i

            # Perpendicular vector (2D rotation by 90°)
            perp = np.array([-direction[1], direction[0]])
            perp_hat = perp / d_ij  # ||direction|| = d_ij

            radius = d_ij / (2.0 * beta)
            offset = (d_ij / 2.0) * np.sqrt(1.0 / beta**2 - 1.0)

            c1 = midpoint + offset * perp_hat
            c2 = midpoint - offset * perp_hat

        return c1, c2, radius

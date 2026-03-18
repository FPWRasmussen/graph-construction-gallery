"""
Sphere of Influence Graph (SIG).

Each point p_i has a "sphere of influence" whose radius r_i equals
the distance to its nearest neighbor. An undirected edge (i, j)
exists if the two spheres overlap:

    d(i, j) < r_i + r_j

This naturally adapts to local point density: in dense regions
the radii are small (tight connections), while in sparse regions
the radii are large (long-range connections).

Reference:
    Toussaint, G.T. (1980). "Pattern Recognition and Geometrical
    Complexity." Proc. 5th ICPR.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class SphereOfInfluenceGraph(GraphBuilder):
    """Connect points whose spheres of influence overlap.

    The radius of each point's sphere equals its nearest-neighbor
    distance, making the graph adaptive to local density.
    """

    slug = "influence"
    category = "proximity"

    @property
    def name(self) -> str:
        return "Sphere of Influence Graph"

    @property
    def description(self) -> str:
        return (
            "Each point's radius = nearest-neighbor distance. "
            "Connect if spheres overlap."
        )

    @property
    def complexity(self) -> str:
        return "O(n²)"

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        # Compute nearest-neighbor distance for each point
        # (set diagonal to inf so a point isn't its own neighbor)
        dist_no_self = dist.copy()
        np.fill_diagonal(dist_no_self, np.inf)
        nn_radius = dist_no_self.min(axis=1)  # shape (n,)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                # Spheres overlap if distance < sum of radii
                if dist[i, j] < nn_radius[i] + nn_radius[j]:
                    G.add_edge(i, j, weight=float(dist[i, j]))

        return G

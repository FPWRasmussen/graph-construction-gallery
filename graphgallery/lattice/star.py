"""
Star Graph.

A star graph S_n has one central "hub" node connected to n-1
"leaf" nodes.  There are no edges between leaf nodes.

Properties:
    - Hub degree: n - 1
    - Leaf degree: 1
    - Total edges: n - 1
    - Planar
    - Bipartite (hub vs. leaves)
    - Diameter: 2 (any leaf can reach any other via the hub)
    - The simplest "hub-and-spoke" topology

Star graphs are important in network science as the extreme
case of degree heterogeneity and as a motif in scale-free networks.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.lattice._layout_utils import (
    ring_positions,
    make_structural_layout,
)


class StarGraph(GraphBuilder):
    """One hub node connected to all other (leaf) nodes.

    Parameters:
        n_override: Explicit node count (overrides layout.n_points).
    """

    slug = "star"
    category = "lattice"

    def __init__(self, n_override: int | None = None):
        self.n_override = n_override

    @property
    def name(self) -> str:
        return "Star Graph"

    @property
    def description(self) -> str:
        return "One hub connected to all n-1 leaf nodes. Diameter = 2."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "n_override", "Explicit node count",
                "int | None", None, "≥ 2",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        n = self.n_override if self.n_override is not None else layout.n_points
        if n < 2:
            raise ValueError(f"Star graph requires n ≥ 2, got {n}")

        # Hub at center, leaves on a ring
        leaf_positions = ring_positions(n - 1, radius=2.0)
        positions = np.vstack([
            [[0.0, 0.0]],  # Hub at origin
            leaf_positions,
        ])

        G = nx.Graph()
        G.add_nodes_from(range(n))

        hub = 0
        for leaf in range(1, n):
            G.add_edge(hub, leaf)

        G.graph["positions"] = positions
        G.graph["hub_node"] = hub
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

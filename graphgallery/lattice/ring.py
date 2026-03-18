"""
Ring / Cycle Graph.

A cycle graph C_n is a graph on n vertices arranged in a circle,
where each vertex is connected to exactly two neighbors (predecessor
and successor).

Properties:
    - Regular (degree 2)
    - Planar
    - 2-connected (removing one vertex leaves a path)
    - Diameter: ⌊n/2⌋
    - Girth (shortest cycle): n
    - Hamiltonian (the graph itself is a Hamilton cycle)
    - The basis for the Watts-Strogatz small-world model
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


class RingGraph(GraphBuilder):
    """Cycle graph where each node connects to its two ring neighbors.

    Parameters:
        n_override: Explicit node count (overrides layout.n_points).
    """

    slug = "ring"
    category = "lattice"

    def __init__(self, n_override: int | None = None):
        self.n_override = n_override

    @property
    def name(self) -> str:
        return "Ring / Cycle Graph"

    @property
    def description(self) -> str:
        return "Each node connected to exactly two neighbors on a circle."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "n_override", "Explicit node count (None=use layout)",
                "int | None", None, "≥ 3",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        n = self.n_override if self.n_override is not None else layout.n_points
        if n < 3:
            raise ValueError(f"Ring graph requires n ≥ 3, got {n}")

        positions = ring_positions(n, radius=2.0)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            G.add_edge(i, (i + 1) % n)

        G.graph["positions"] = positions
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

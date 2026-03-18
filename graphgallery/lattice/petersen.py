"""
Petersen Graph.

The Petersen graph is one of the most famous named graphs in graph
theory.  It has 10 nodes and 15 edges and serves as a counterexample
to many conjectures.

Construction (generalized Petersen graph GP(n, k)):
    - An outer ring of n nodes: 0, 1, ..., n-1
    - An inner ring of n nodes: n, n+1, ..., 2n-1
    - Outer edges: i — (i+1) mod n
    - Inner edges: (n+i) — (n + (i+k) mod n)
    - Spokes: i — (n+i)

The classic Petersen graph is GP(5, 2).

Properties:
    - 10 nodes, 15 edges
    - 3-regular (cubic)
    - Not Hamiltonian (the smallest 3-regular graph without a Hamilton cycle)
    - Not planar (contains K_{3,3} as a minor)
    - Vertex-transitive and edge-transitive
    - Girth: 5
    - Diameter: 2
    - Chromatic number: 3
    - Serves as a counterexample to many conjectures

This builder also supports the generalized Petersen graph GP(n, k).
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


class PetersenGraph(GraphBuilder):
    """The classic Petersen graph GP(5,2) or generalized GP(n,k).

    Parameters:
        n: Number of nodes per ring (classic Petersen: 5).
        k: Inner ring step size (classic Petersen: 2).
    """

    slug = "petersen"
    category = "lattice"

    def __init__(self, n: int = 5, k: int = 2):
        self.n = n
        self.k = k

    @property
    def name(self) -> str:
        if self.n == 5 and self.k == 2:
            return "Petersen Graph"
        return f"Generalized Petersen GP({self.n},{self.k})"

    @property
    def description(self) -> str:
        if self.n == 5 and self.k == 2:
            return (
                "The classic Petersen graph: 10 nodes, 15 edges, 3-regular. "
                "Famous counterexample in graph theory."
            )
        return f"Generalized Petersen graph GP({self.n},{self.k}): 2n nodes, 3n edges."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("n", "Nodes per ring", "int", 5, "n ≥ 3"),
            ParamInfo(
                "k", "Inner ring step size",
                "int", 2, "1 ≤ k < n/2",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        n = self.n
        k = self.k

        if n < 3:
            raise ValueError(f"Petersen graph requires n ≥ 3, got {n}")
        if k < 1 or k >= n / 2:
            raise ValueError(
                f"Step k={k} must satisfy 1 ≤ k < n/2={n / 2}"
            )

        total_nodes = 2 * n

        # Positions: outer ring (larger) and inner ring (smaller)
        outer_radius = 2.0
        inner_radius = 1.0
        outer_pos = ring_positions(n, radius=outer_radius)
        inner_pos = ring_positions(n, radius=inner_radius)

        positions = np.vstack([outer_pos, inner_pos])

        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))

        for i in range(n):
            # Outer ring edges
            G.add_edge(i, (i + 1) % n)

            # Inner ring edges (step k)
            G.add_edge(n + i, n + (i + k) % n)

            # Spoke edges
            G.add_edge(i, n + i)

        # Node metadata
        for i in range(n):
            G.nodes[i]["ring"] = "outer"
        for i in range(n, total_nodes):
            G.nodes[i]["ring"] = "inner"

        G.graph["positions"] = positions
        G.graph["petersen_n"] = n
        G.graph["petersen_k"] = k
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

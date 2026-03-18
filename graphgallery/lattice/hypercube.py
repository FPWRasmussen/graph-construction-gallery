"""
Hypercube Graph Q_d.

The d-dimensional hypercube graph has 2^d nodes, each labeled with
a d-bit binary string.  Two nodes are adjacent iff their labels
differ in exactly one bit (Hamming distance = 1).

Properties:
    - Nodes: 2^d
    - Edges: d · 2^(d-1)
    - Regular (degree d)
    - Bipartite (even vs. odd parity bit strings)
    - Diameter: d
    - Vertex-transitive and edge-transitive
    - Excellent expander properties
    - Used in parallel computing interconnect topologies

For ~30 nodes, d=5 gives Q_5 with 32 nodes.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.lattice._layout_utils import make_structural_layout


class HypercubeGraph(GraphBuilder):
    """d-dimensional hypercube: nodes are bit strings, edges differ by 1 bit.

    Parameters:
        d: Dimension (None = auto from layout.n_points).
    """

    slug = "hypercube"
    category = "lattice"

    def __init__(self, d: int | None = None):
        self.d = d

    @property
    def name(self) -> str:
        return "Hypercube Graph"

    @property
    def description(self) -> str:
        return "Nodes are d-bit strings; adjacent iff Hamming distance = 1."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(d · 2^d)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("d", "Dimension (None=auto)", "int | None", None, "d ≥ 1"),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        d = self._resolve_dim(layout.n_points)
        n = 2 ** d

        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Connect nodes differing by one bit
        for i in range(n):
            for bit in range(d):
                j = i ^ (1 << bit)
                if j > i:  # Avoid duplicate edges
                    G.add_edge(i, j)

        # Store binary labels
        for i in range(n):
            G.nodes[i]["binary"] = format(i, f"0{d}b")

        # Layout: use a spectral-like 2D projection
        positions = self._compute_positions(n, d)

        G.graph["positions"] = positions
        G.graph["dimension"] = d
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

    def _resolve_dim(self, n_target: int) -> int:
        """Choose dimension so 2^d is close to n_target."""
        if self.d is not None:
            return self.d
        return max(1, int(np.round(np.log2(n_target))))

    @staticmethod
    def _compute_positions(n: int, d: int) -> np.ndarray:
        """Compute 2D positions for a d-cube using recursive placement.

        Uses a layout where each dimension adds a displacement vector,
        producing a visually clear hypercube structure.
        """
        if d <= 0:
            return np.array([[0.0, 0.0]])

        # Base displacement vectors: alternate angles for each dimension
        # to spread the layout
        positions = np.zeros((n, 2), dtype=np.float64)

        for bit in range(d):
            angle = np.pi * bit / d + np.pi / 6
            displacement = np.array([np.cos(angle), np.sin(angle)])
            scale = 2.0 ** (d - 1 - bit) * 0.6

            for i in range(n):
                if i & (1 << bit):
                    positions[i] += scale * displacement

        # Center
        positions -= positions.mean(axis=0)

        return positions

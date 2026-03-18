"""
Triangular Lattice Graph.

A triangular lattice is a grid where each cell is divided into two
triangles by adding one diagonal.  Interior nodes have degree 6,
connecting to 4 cardinal and 2 diagonal neighbors.

Properties:
    - Interior degree: 6
    - Planar
    - The dual of the hexagonal lattice
    - Used in finite element meshes, crystallography, and game boards
    - Contains both the 4-connected grid and additional diagonals

Construction:
    Start with a rectangular grid and add the diagonal from top-left
    to bottom-right in each cell (or equivalently, connect each node
    to its (r+1, c+1) neighbor).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.lattice._layout_utils import (
    best_grid_dims,
    make_structural_layout,
)


class TriangularLatticeGraph(GraphBuilder):
    """Grid with diagonal edges → degree-6 triangular tiling.

    Parameters:
        rows: Number of rows (None = auto).
        cols: Number of columns (None = auto).
    """

    slug = "triangular_lattice"
    category = "lattice"

    def __init__(self, rows: int | None = None, cols: int | None = None):
        self.rows = rows
        self.cols = cols

    @property
    def name(self) -> str:
        return "Triangular Lattice"

    @property
    def description(self) -> str:
        return "Grid + diagonals: degree-6 interior nodes. Dual of hexagonal lattice."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("rows", "Grid rows (None=auto)", "int | None", None, "≥ 2"),
            ParamInfo("cols", "Grid columns (None=auto)", "int | None", None, "≥ 2"),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        rows, cols = self._resolve_dims(layout.n_points)
        n = rows * cols

        # Positions: offset even rows slightly for equilateral appearance
        positions = np.zeros((n, 2), dtype=np.float64)
        y_spacing = np.sqrt(3) / 2.0

        def node_id(r: int, c: int) -> int:
            return r * cols + c

        for r in range(rows):
            for c in range(cols):
                x_offset = 0.5 if r % 2 == 1 else 0.0
                positions[node_id(r, c)] = [
                    c + x_offset,
                    (rows - 1 - r) * y_spacing,
                ]

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for r in range(rows):
            for c in range(cols):
                u = node_id(r, c)

                # Right neighbor
                if c + 1 < cols:
                    G.add_edge(u, node_id(r, c + 1))

                # Down neighbor
                if r + 1 < rows:
                    G.add_edge(u, node_id(r + 1, c))

                # Diagonal neighbors (depends on row parity for
                # equilateral triangle tiling)
                if r + 1 < rows:
                    if r % 2 == 0:
                        # Even row: connect down-left
                        if c - 1 >= 0:
                            G.add_edge(u, node_id(r + 1, c - 1))
                    else:
                        # Odd row: connect down-right
                        if c + 1 < cols:
                            G.add_edge(u, node_id(r + 1, c + 1))

        G.graph["positions"] = positions
        G.graph["rows"] = rows
        G.graph["cols"] = cols
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

    def _resolve_dims(self, n_target: int) -> tuple[int, int]:
        """Determine grid dimensions."""
        if self.rows is not None and self.cols is not None:
            return max(2, self.rows), max(2, self.cols)
        rows, cols = best_grid_dims(n_target)
        return max(2, rows), max(2, cols)

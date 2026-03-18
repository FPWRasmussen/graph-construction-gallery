"""
Grid / Lattice Graph (2D).

A 2D grid graph arranges nodes on a rectangular grid where each
interior node connects to its 4 cardinal neighbors (up, down, left,
right).  This is the standard 4-connected lattice used in image
processing, physics (Ising model), and pathfinding.

Properties:
    - Nodes: rows × cols
    - Interior degree: 4
    - Edge/corner degree: 3 / 2
    - Planar
    - Bipartite (checkerboard coloring)
    - Diameter: rows + cols - 2

Optional 8-connectivity adds diagonal neighbors.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.lattice._layout_utils import (
    grid_positions,
    best_grid_dims,
    make_structural_layout,
)


class GridGraph(GraphBuilder):
    """2D rectangular grid graph with 4- or 8-connectivity.

    Parameters:
        rows: Number of rows (auto-computed from n if None).
        cols: Number of columns (auto-computed from n if None).
        eight_connected: If True, include diagonal neighbors (8-conn).
    """

    slug = "grid_2d"
    category = "lattice"

    def __init__(
        self,
        rows: int | None = None,
        cols: int | None = None,
        eight_connected: bool = False,
    ):
        self.rows = rows
        self.cols = cols
        self.eight_connected = eight_connected

    @property
    def name(self) -> str:
        conn = "8" if self.eight_connected else "4"
        return f"Grid Graph ({conn}-connected)"

    @property
    def description(self) -> str:
        return "Regular 2D rectangular lattice with cardinal (and optional diagonal) neighbors."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("rows", "Grid rows (None=auto)", "int | None", None, "≥ 1"),
            ParamInfo("cols", "Grid columns (None=auto)", "int | None", None, "≥ 1"),
            ParamInfo(
                "eight_connected", "Include diagonal neighbors",
                "bool", False,
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        rows, cols = self._resolve_dims(layout.n_points)
        n = rows * cols
        positions = grid_positions(rows, cols)

        def node_id(r: int, c: int) -> int:
            return r * cols + c

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for r in range(rows):
            for c in range(cols):
                u = node_id(r, c)

                # 4-connectivity: right and down
                if c + 1 < cols:
                    G.add_edge(u, node_id(r, c + 1))
                if r + 1 < rows:
                    G.add_edge(u, node_id(r + 1, c))

                # 8-connectivity: diagonals
                if self.eight_connected:
                    if r + 1 < rows and c + 1 < cols:
                        G.add_edge(u, node_id(r + 1, c + 1))
                    if r + 1 < rows and c - 1 >= 0:
                        G.add_edge(u, node_id(r + 1, c - 1))

        G.graph["positions"] = positions
        G.graph["rows"] = rows
        G.graph["cols"] = cols
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

    def _resolve_dims(self, n_target: int) -> tuple[int, int]:
        """Determine grid dimensions."""
        if self.rows is not None and self.cols is not None:
            return self.rows, self.cols
        if self.rows is not None:
            return self.rows, max(1, int(np.ceil(n_target / self.rows)))
        if self.cols is not None:
            return max(1, int(np.ceil(n_target / self.cols))), self.cols
        return best_grid_dims(n_target)

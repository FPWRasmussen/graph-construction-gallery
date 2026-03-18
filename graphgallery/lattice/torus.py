"""
Torus Graph.

A torus graph is a 2D grid with wrap-around edges: the top row
connects to the bottom row, and the left column connects to the
right column.  Equivalently, it is the Cartesian product of two
cycle graphs: C_rows □ C_cols.

Properties:
    - Regular (degree 4)
    - Not planar (for rows, cols ≥ 3)
    - No boundary effects (every node is equivalent)
    - Vertex-transitive
    - Commonly used in physics (periodic boundary conditions),
      parallel computing (mesh interconnects), and cellular automata

Layout note: Drawn as a flat grid with curved wrap-around edges
for clarity, rather than attempting a 3D torus visualization.
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


class TorusGraph(GraphBuilder):
    """2D grid with periodic (wrap-around) boundary conditions.

    Parameters:
        rows: Number of rows (None = auto).
        cols: Number of columns (None = auto).
    """

    slug = "torus"
    category = "lattice"

    def __init__(self, rows: int | None = None, cols: int | None = None):
        self.rows = rows
        self.cols = cols

    @property
    def name(self) -> str:
        return "Torus Graph"

    @property
    def description(self) -> str:
        return "2D grid with wrap-around edges (periodic boundaries). Degree 4 everywhere."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("rows", "Grid rows (None=auto)", "int | None", None, "≥ 3"),
            ParamInfo("cols", "Grid columns (None=auto)", "int | None", None, "≥ 3"),
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

                # Right neighbor (with wrap)
                right = node_id(r, (c + 1) % cols)
                if right != u:
                    G.add_edge(u, right)

                # Down neighbor (with wrap)
                down = node_id((r + 1) % rows, c)
                if down != u:
                    G.add_edge(u, down)

        # Mark wrap-around edges for styling
        for u, v in G.edges():
            r_u, c_u = divmod(u, cols)
            r_v, c_v = divmod(v, cols)
            is_wrap = (
                abs(c_u - c_v) > 1 or  # Horizontal wrap
                abs(r_u - r_v) > 1      # Vertical wrap
            )
            G[u][v]["wrap"] = is_wrap

        G.graph["positions"] = positions
        G.graph["rows"] = rows
        G.graph["cols"] = cols
        G.graph["structural_layout"] = make_structural_layout(positions)

        return G

    def _resolve_dims(self, n_target: int) -> tuple[int, int]:
        """Determine torus dimensions (minimum 3×3)."""
        if self.rows is not None and self.cols is not None:
            return max(3, self.rows), max(3, self.cols)
        rows, cols = best_grid_dims(n_target)
        return max(3, rows), max(3, cols)

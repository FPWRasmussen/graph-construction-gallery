"""
Hexagonal (Honeycomb) Lattice Graph.

A hexagonal lattice consists of regular hexagonal tiles where each
vertex belongs to exactly 3 hexagons.  Every interior node has
degree 3 (trivalent/cubic).

Properties:
    - Degree: 3 (interior), 2 (boundary)
    - Planar
    - Bipartite
    - Used in chemistry (graphene, benzene), board games (hex grids),
      cellular networks, and antenna placement

Construction:
    We generate a "brick wall" pattern where even rows are offset
    by half a spacing unit, creating the honeycomb topology.
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


class HexagonalLatticeGraph(GraphBuilder):
    """Honeycomb lattice with degree-3 interior nodes.

    Parameters:
        rows: Number of hex rows (None = auto).
        cols: Number of hex columns (None = auto).
    """

    slug = "hexagonal"
    category = "lattice"

    def __init__(self, rows: int | None = None, cols: int | None = None):
        self.rows = rows
        self.cols = cols

    @property
    def name(self) -> str:
        return "Hexagonal Lattice"

    @property
    def description(self) -> str:
        return "Honeycomb tiling: degree-3 interior nodes. Planar and bipartite."

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("rows", "Hex rows (None=auto)", "int | None", None, "≥ 2"),
            ParamInfo("cols", "Hex columns (None=auto)", "int | None", None, "≥ 2"),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        rows, cols = self._resolve_dims(layout.n_points)

        G = nx.Graph()
        positions: dict[int, tuple[float, float]] = {}
        node_grid: dict[tuple[int, int], int] = {}
        node_id = 0

        # Generate honeycomb nodes and positions
        # Each "cell" row has two rows of nodes
        x_spacing = 1.0
        y_spacing = np.sqrt(3) / 2.0

        for r in range(rows):
            for c in range(cols):
                x = c * x_spacing
                y = r * y_spacing

                # Offset odd rows
                if r % 2 == 1:
                    x += 0.5

                node_grid[(r, c)] = node_id
                positions[node_id] = (x, y)
                G.add_node(node_id)
                node_id += 1

        # Connect edges: honeycomb pattern
        for r in range(rows):
            for c in range(cols):
                u = node_grid[(r, c)]

                # Horizontal neighbor (right)
                if c + 1 < cols:
                    v = node_grid[(r, c + 1)]
                    G.add_edge(u, v)

                # Vertical neighbors depend on row parity
                if r + 1 < rows:
                    # For honeycomb: connect to specific neighbors
                    # in the next row based on column parity
                    if r % 2 == 0:
                        # Even row → connect down-left and down (same col)
                        if (r + 1, c) in node_grid:
                            G.add_edge(u, node_grid[(r + 1, c)])
                        # Only connect down-left for even columns
                        if c > 0 and (r + 1, c - 1) in node_grid:
                            # Only for specific pattern
                            if c % 2 == 0:
                                G.add_edge(u, node_grid[(r + 1, c - 1)])
                    else:
                        # Odd row → connect down and down-right
                        if (r + 1, c) in node_grid:
                            G.add_edge(u, node_grid[(r + 1, c)])
                        if c + 1 < cols and (r + 1, c + 1) in node_grid:
                            if c % 2 == 0:
                                G.add_edge(u, node_grid[(r + 1, c + 1)])

        # Rebuild as a clean position array
        n = G.number_of_nodes()
        pos_array = np.zeros((n, 2), dtype=np.float64)
        for nid, (x, y) in positions.items():
            pos_array[nid] = [x, y]

        G.graph["positions"] = pos_array
        G.graph["rows"] = rows
        G.graph["cols"] = cols
        G.graph["structural_layout"] = make_structural_layout(pos_array)

        return G

    def _resolve_dims(self, n_target: int) -> tuple[int, int]:
        """Determine hex grid dimensions."""
        if self.rows is not None and self.cols is not None:
            return max(2, self.rows), max(2, self.cols)
        rows, cols = best_grid_dims(n_target)
        return max(2, rows), max(2, cols)

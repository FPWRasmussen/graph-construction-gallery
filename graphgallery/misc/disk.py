"""
Disk Graph (Unit Disk Graph).

In a disk graph, each node has a "disk" of a fixed radius r centered
at its position.  An edge exists between two nodes if and only if
their disks overlap, which is equivalent to:

    d(p_i, p_j) ≤ 2r

(since two disks of radius r overlap iff the centers are within 2r.)

In the **unit disk graph** (UDG), all disks have radius 1, so the
connectivity threshold is distance ≤ 2.  This builder also supports
**variable-radius** disks where each node's radius depends on local
density (its nearest-neighbor distance), similar to the sphere of
influence graph but using a different radius formula.

Applications:
    - Wireless sensor networks (transmission range = disk radius)
    - Ad-hoc network modeling
    - Continuum percolation theory
    - Interference modeling in cellular networks
    - Geographic network analysis

The unit disk graph is a fundamental model in wireless networking
and has been extensively studied in computational geometry and
combinatorial optimization.

Reference:
    Clark, B.N., Colbourn, C.J., & Johnson, D.S. (1990). "Unit
    disk graphs." Discrete Mathematics, 86(1–3), 165–177.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class DiskGraph(GraphBuilder):
    """Connect nodes whose disks overlap (distance ≤ 2r).

    Supports uniform radius and adaptive per-node radius.

    Parameters:
        r: Disk radius.  The connectivity threshold is 2r.
        mode: ``"uniform"`` (all same radius) or ``"adaptive"``
            (radius proportional to k-th nearest neighbor distance).
        adaptive_k: k-th neighbor for adaptive radius.
        adaptive_scale: Multiplier for the adaptive radius.
    """

    slug = "disk"
    category = "misc"

    def __init__(
        self,
        r: float = 0.5,
        mode: Literal["uniform", "adaptive"] = "uniform",
        adaptive_k: int = 3,
        adaptive_scale: float = 0.5,
    ):
        self.r = r
        self.mode = mode
        self.adaptive_k = adaptive_k
        self.adaptive_scale = adaptive_scale

    @property
    def name(self) -> str:
        if self.mode == "uniform":
            return "Disk Graph"
        return "Adaptive Disk Graph"

    @property
    def description(self) -> str:
        if self.mode == "uniform":
            return (
                f"Unit disk model: connect if distance ≤ {2 * self.r:.2f} "
                f"(radius r={self.r})."
            )
        return (
            f"Adaptive disk: per-node radius from {self.adaptive_k}-th "
            f"neighbor distance × {self.adaptive_scale}."
        )

    @property
    def complexity(self) -> str:
        return "O(n²)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("r", "Disk radius (uniform mode)", "float", 0.5, "r > 0"),
            ParamInfo(
                "mode", "Radius mode", "str", "uniform",
                "uniform | adaptive",
            ),
            ParamInfo("adaptive_k", "k-th neighbor for adaptive", "int", 3, "k ≥ 1"),
            ParamInfo("adaptive_scale", "Adaptive radius multiplier", "float", 0.5, "> 0"),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        # Compute per-node radii
        if self.mode == "adaptive":
            radii = self._compute_adaptive_radii(dist, n)
        else:
            radii = np.full(n, self.r, dtype=np.float64)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            G.nodes[i]["radius"] = float(radii[i])

        for i in range(n):
            for j in range(i + 1, n):
                # Disks overlap if distance ≤ r_i + r_j
                if dist[i, j] <= radii[i] + radii[j]:
                    G.add_edge(i, j, weight=float(dist[i, j]))

        G.graph["algorithm"] = "disk"
        G.graph["mode"] = self.mode
        G.graph["radii"] = radii.tolist()

        return G

    def _compute_adaptive_radii(
        self, dist: np.ndarray, n: int
    ) -> np.ndarray:
        """Compute per-node adaptive radii from k-th neighbor distances."""
        from graphgallery.points import k_nearest_indices

        k = min(self.adaptive_k, n - 1)
        knn = k_nearest_indices(dist, k)

        radii = np.array([
            dist[i, knn[i, -1]] * self.adaptive_scale
            for i in range(n)
        ], dtype=np.float64)

        return np.maximum(radii, 1e-10)

"""
Horizontal Visibility Graph (HVG).

The horizontal visibility graph is a simplified variant of the
natural visibility graph [[4]].  Instead of checking whether
intermediate points lie below a sloped line, it checks whether
they lie below a horizontal line at the height of the shorter
endpoint:

    Points i and j are horizontally visible iff for all k (i < k < j):
        y_k < min(y_i, y_j)

Geometrically, imagine each data point as a bar in a bar chart.
Two bars can "see" each other horizontally if you can draw a
horizontal line at the height of the shorter bar without hitting
any intermediate bar.

Key properties:
    - Always a subgraph of the NVG (stricter visibility criterion)
    - Always connected (adjacent points are always visible)
    - Simpler and faster to compute than NVG
    - Random i.i.d. series → degree distribution P(k) = (1/3)(2/3)^(k-2)
    - The mean degree of a random HVG is exactly 4
    - Useful for distinguishing random from chaotic series

The HVG was introduced by Luque et al. (2009) as a computationally
simpler alternative to the NVG that still captures essential
dynamical features.

Reference:
    Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009).
    "Horizontal visibility graphs: Exact results for random time
    series." Physical Review E, 80(4), 046103.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.visibility._visibility_utils import (
    layout_to_time_series,
    original_index,
    horizontal_visibility_test,
)


class HorizontalVisibilityGraph(GraphBuilder):
    """Time series → graph via horizontal line-of-sight visibility.

    Simpler than natural visibility: intermediate points must lie
    below the minimum of the two endpoint values.

    Parameters:
        directed: If True, edges point forward in time only.
        use_original_indices: If True, node indices match original layout.
    """

    slug = "horizontal_visibility"
    category = "visibility"

    def __init__(
        self,
        directed: bool = False,
        use_original_indices: bool = True,
    ):
        self.directed = directed
        self.use_original_indices = use_original_indices

    @property
    def name(self) -> str:
        return "Horizontal Visibility Graph"

    @property
    def description(self) -> str:
        return (
            "Simplified visibility: intermediate points must lie "
            "below min(y_i, y_j). Always a subgraph of the NVG."
        )

    @property
    def is_directed(self) -> bool:
        return self.directed

    @property
    def complexity(self) -> str:
        return "O(n²) naïve, O(n log n) with stack-based approach"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "directed", "Forward-in-time edges only",
                "bool", False,
            ),
            ParamInfo(
                "use_original_indices", "Map nodes to original layout indices",
                "bool", True,
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points

        # Convert layout to time series
        times, values, sort_order = layout_to_time_series(layout)

        # Choose graph type
        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from(range(n))

        # Store time series data as node attributes
        for sorted_idx in range(n):
            orig_idx = original_index(sort_order, sorted_idx)
            node_id = orig_idx if self.use_original_indices else sorted_idx
            G.nodes[node_id]["time"] = float(times[sorted_idx])
            G.nodes[node_id]["value"] = float(values[sorted_idx])
            G.nodes[node_id]["sorted_index"] = sorted_idx

        # Optimized horizontal visibility using a stack
        # For each point, find the farthest visible point to the right
        # using monotone stack properties
        #
        # However, for correctness with all pairs, we use the O(n²) approach
        # and include all visible pairs, not just the farthest.
        for i in range(n):
            for j in range(i + 1, n):
                if horizontal_visibility_test(values, i, j):
                    if self.use_original_indices:
                        u = original_index(sort_order, i)
                        v = original_index(sort_order, j)
                    else:
                        u, v = i, j

                    dt = abs(float(times[j] - times[i]))
                    G.add_edge(u, v, weight=dt, time_gap=j - i)

        G.graph["algorithm"] = "horizontal_visibility"
        G.graph["sort_order"] = sort_order.tolist()
        G.graph["times"] = times.tolist()
        G.graph["values"] = values.tolist()

        return G

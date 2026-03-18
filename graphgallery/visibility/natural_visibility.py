"""
Natural Visibility Graph (NVG).

The natural visibility graph converts a time series into a graph
where each data point becomes a node and edges represent geometric
visibility between points [[8]].

Given a time series {(t_1, y_1), (t_2, y_2), ..., (t_n, y_n)},
two points (t_i, y_i) and (t_j, y_j) are connected if for every
intermediate point k (i < k < j):

    y_k < y_i + (y_j − y_i) · (t_k − t_i) / (t_j − t_i)

Geometrically, this means all intermediate bars in a bar chart
representation lie below the straight line connecting the tops of
bars i and j [[9]].

Key properties:
    - Connected (every adjacent pair is visible)
    - Invariant under affine transformations of the time axis
    - Invariant under rescaling of values
    - Periodic series → regular graphs
    - Random series → scale-free graphs
    - The degree distribution inherits the series dynamics

The NVG was introduced by Lacasa et al. (2008) as a bridge between
time series analysis and complex network theory [[8]].

For our gallery, we sort the canonical point layout by x-coordinate
and treat the y-coordinates as the time series values.

Reference:
    Lacasa, L., Luque, B., Ballesteros, F., Luque, J., & Nuño, J.C.
    (2008). "From time series to complex networks: The visibility
    graph." Proc. National Academy of Sciences, 105(13), 4972–4975.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.visibility._visibility_utils import (
    layout_to_time_series,
    original_index,
    natural_visibility_test,
)


class NaturalVisibilityGraph(GraphBuilder):
    """Time series → graph via natural (line-of-sight) visibility.

    Points sorted by x-coordinate form a time series.  Two points
    are connected if all intermediate points lie below the line
    connecting them.

    Parameters:
        directed: If True, produce a directed graph where edges point
            forward in time only.
        use_original_indices: If True, node indices match the original
            layout.  If False, nodes are indexed in sorted time order.
    """

    slug = "natural_visibility"
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
        return "Natural Visibility Graph"

    @property
    def description(self) -> str:
        return (
            "Time series → graph: connect points with unobstructed "
            "line-of-sight over intermediate values."
        )

    @property
    def is_directed(self) -> bool:
        return self.directed

    @property
    def complexity(self) -> str:
        return "O(n²) naïve, O(n log n) with divide-and-conquer"

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

        # Store the time series data as node attributes
        for sorted_idx in range(n):
            orig_idx = original_index(sort_order, sorted_idx)
            node_id = orig_idx if self.use_original_indices else sorted_idx
            G.nodes[node_id]["time"] = float(times[sorted_idx])
            G.nodes[node_id]["value"] = float(values[sorted_idx])
            G.nodes[node_id]["sorted_index"] = sorted_idx

        # Test visibility for all pairs
        for i in range(n):
            # Only need to check j > i (forward in sorted time)
            for j in range(i + 1, n):
                if natural_visibility_test(times, values, i, j):
                    # Map to node indices
                    if self.use_original_indices:
                        u = original_index(sort_order, i)
                        v = original_index(sort_order, j)
                    else:
                        u, v = i, j

                    # Edge weight = time difference
                    dt = abs(float(times[j] - times[i]))
                    G.add_edge(u, v, weight=dt, time_gap=j - i)

        G.graph["algorithm"] = "natural_visibility"
        G.graph["sort_order"] = sort_order.tolist()
        G.graph["times"] = times.tolist()
        G.graph["values"] = values.tolist()

        return G

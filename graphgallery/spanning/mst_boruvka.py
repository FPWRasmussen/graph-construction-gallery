"""
Minimum Spanning Tree — Borůvka's Algorithm.

Borůvka's algorithm (also called Sollin's algorithm) is the oldest
MST algorithm (1926).  It works by simultaneously growing multiple
tree fragments in parallel:

    1. Start with n single-vertex components.
    2. In each round, for every component find the cheapest edge
       connecting it to a *different* component.
    3. Add all such cheapest edges (merging components).
    4. Repeat until a single component (the MST) remains.

The number of components at least halves each round, so at most
O(log n) rounds are needed, each taking O(m) time.

Strategy:  component-centric, parallel-friendly, O(m log n).

Borůvka's is historically significant and naturally parallelizable,
making it relevant for distributed and GPU-based MST computation.

Reference:
    Borůvka, O. (1926). "O jistém problému minimálním."
    Práce Moravské Přírodovědecké Společnosti, 3, 37–58.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class MSTBoruvkaGraph(GraphBuilder):
    """Minimum Spanning Tree via Borůvka's parallel component-merging.

    Each round finds the cheapest outgoing edge for every component
    and merges.  Components halve each round → O(log n) rounds.
    """

    slug = "mst_boruvka"
    category = "spanning"

    @property
    def name(self) -> str:
        return "MST (Borůvka's)"

    @property
    def description(self) -> str:
        return (
            "Parallel-friendly MST: each round merges components via "
            "their cheapest outgoing edge. O(log n) rounds."
        )

    @property
    def complexity(self) -> str:
        return "O(m log n), naturally parallelizable"

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        # Component label for each vertex (initially each is its own)
        component = np.arange(n, dtype=np.intp)
        mst_edges: list[tuple[int, int, float]] = []

        max_rounds = int(np.ceil(np.log2(n))) + 2  # Safety bound

        for _round in range(max_rounds):
            if len(mst_edges) >= n - 1:
                break

            # For each component, find its cheapest outgoing edge
            # cheapest[comp_id] = (weight, u, v) or None
            n_components = len(set(component))
            if n_components <= 1:
                break

            cheapest: dict[int, tuple[float, int, int] | None] = {}

            for i in range(n):
                for j in range(i + 1, n):
                    ci, cj = component[i], component[j]
                    if ci == cj:
                        continue  # Same component

                    w = float(dist[i, j])

                    # Update cheapest for component ci
                    if ci not in cheapest or cheapest[ci] is None:
                        cheapest[ci] = (w, i, j)
                    elif w < cheapest[ci][0]:
                        cheapest[ci] = (w, i, j)

                    # Update cheapest for component cj
                    if cj not in cheapest or cheapest[cj] is None:
                        cheapest[cj] = (w, i, j)
                    elif w < cheapest[cj][0]:
                        cheapest[cj] = (w, i, j)

            if not cheapest:
                break

            # Add the cheapest edges and merge components
            edges_added_this_round: set[tuple[int, int]] = set()

            for comp_id, edge_info in cheapest.items():
                if edge_info is None:
                    continue

                w, u, v = edge_info
                edge_key = (min(u, v), max(u, v))

                if edge_key in edges_added_this_round:
                    continue  # Already added from the other component's side

                # Check that they are still in different components
                # (may have been merged earlier this round)
                if component[u] == component[v]:
                    continue

                edges_added_this_round.add(edge_key)
                mst_edges.append((u, v, w))

                # Merge: relabel the smaller component
                old_comp = component[v]
                new_comp = component[u]
                if old_comp != new_comp:
                    mask = component == old_comp
                    component[mask] = new_comp

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v, w in mst_edges:
            G.add_edge(u, v, weight=w)

        G.graph["algorithm"] = "boruvka"
        G.graph["total_weight"] = sum(w for _, _, w in mst_edges)

        return G

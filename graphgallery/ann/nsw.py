"""
Navigable Small World (NSW) Graph.

The NSW graph is built by incrementally inserting nodes and connecting
each new node to its approximate nearest neighbors found via greedy
search on the graph built so far.  Early-inserted nodes tend to
become long-range "highway" edges, while later nodes get short-range
local connections — naturally creating a navigable small-world
structure.

Construction:
    For each new node q (inserted in random order):
    1. Use greedy search on the current graph to find the f closest
       nodes to q.
    2. Add bidirectional edges from q to each of the f found neighbors.

Search:
    - Start from a random (or designated) entry point.
    - Greedily move to the neighbor closest to the query.
    - Multi-start and beam search improve recall.

NSW achieves polylogarithmic search complexity on uniformly
distributed data, with the greedy search exploiting the small-world
property for efficient navigation.

Reference:
    Malkov, Y., Ponomarenko, A., Logvinov, A., & Krylov, V. (2014).
    "Approximate nearest neighbor algorithm based on navigable small
    world graphs." Information Systems, 45, 61–68.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.ann._ann_utils import greedy_search, euclidean_distance


class NSWGraph(GraphBuilder):
    """Navigable Small World graph via incremental greedy insertion.

    Early nodes become long-range hubs; later nodes get local edges.
    The result is a single-layer navigable graph.

    Parameters:
        f: Number of friends (neighbors) per insertion.
        ef_construction: Beam width during construction search.
        seed: Random seed for insertion order.
    """

    slug = "nsw"
    category = "ann"

    def __init__(
        self,
        f: int = 5,
        ef_construction: int = 16,
        seed: int | None = None,
    ):
        self.f = f
        self.ef_construction = ef_construction
        self.seed = seed

    @property
    def name(self) -> str:
        return "Navigable Small World"

    @property
    def description(self) -> str:
        return (
            f"Incremental insertion with greedy search. "
            f"f={self.f} friends per node."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n · f · log n) expected"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("f", "Friends per insertion", "int", 5, "f ≥ 1"),
            ParamInfo(
                "ef_construction", "Search beam width during build",
                "int", 16, "≥ f",
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        seed = self.seed if self.seed is not None else (layout.seed + 200)
        rng = np.random.default_rng(seed)

        # Random insertion order
        insertion_order = rng.permutation(n)

        # Adjacency: node → set of neighbors
        adj: dict[int, set[int]] = {i: set() for i in range(n)}

        # Insert nodes one by one
        inserted: set[int] = set()

        for step, node in enumerate(insertion_order):
            if step == 0:
                # First node: no neighbors yet
                inserted.add(node)
                continue

            if step < self.f + 1:
                # Not enough nodes for full search — connect to all existing
                for existing in inserted:
                    adj[node].add(existing)
                    adj[existing].add(node)
                inserted.add(node)
                continue

            # Greedy search from a random entry point
            entry = int(rng.choice(list(inserted)))

            results = greedy_search(
                query=points[node],
                entry_point=entry,
                points=points,
                adj=adj,
                ef=max(self.ef_construction, self.f),
            )

            # Connect to the f nearest found
            neighbors_to_add = [
                idx for _, idx in results[:self.f]
                if idx != node
            ]

            for nbr in neighbors_to_add:
                adj[node].add(nbr)
                adj[nbr].add(node)

            inserted.add(node)

        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    G.add_edge(u, v, weight=float(dist[u, v]))

        G.graph["algorithm"] = "nsw"
        G.graph["f"] = self.f
        G.graph["entry_point"] = int(insertion_order[0])

        return G

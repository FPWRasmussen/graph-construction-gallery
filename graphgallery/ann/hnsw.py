"""
Hierarchical Navigable Small World (HNSW) Graph.

HNSW extends NSW with a multi-layer hierarchy inspired by skip lists.
Each node is assigned a random level ℓ drawn from an exponential
distribution.  The graph has L layers (0 being the densest, bottom
layer):

    - Layer 0: contains ALL nodes (the base layer).
    - Layer ℓ > 0: contains only nodes with level ≥ ℓ.

Each layer is an independent NSW graph.  During search:
    1. Enter at the top layer via the single entry point.
    2. Greedily search within each layer.
    3. Drop to the next layer and repeat.
    4. Perform a wide beam search on layer 0.

This hierarchical structure provides O(log n) search complexity
with high probability, making HNSW one of the fastest ANN algorithms
in practice [[5]].

The output of this builder is the **base layer (layer 0)** graph,
with layer assignments stored as node attributes.

Reference:
    Malkov, Y.A. & Yashunin, D.A. (2020). "Efficient and Robust
    Approximate Nearest Neighbor Search Using Hierarchical Navigable
    Small World Graphs." IEEE TPAMI, 42(4), 824–836.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.ann._ann_utils import greedy_search, euclidean_distance


class HNSWGraph(GraphBuilder):
    """Multi-layer hierarchical navigable small world graph.

    Builds a layered NSW structure.  The returned graph is the
    union of all layers, with layer information on each edge.

    Parameters:
        M: Maximum number of connections per node per layer.
        M0: Maximum connections on layer 0 (typically 2*M).
        ef_construction: Beam width during construction.
        mL: Level generation factor (1/ln(M)).
        seed: Random seed.
    """

    slug = "hnsw"
    category = "ann"

    def __init__(
        self,
        M: int = 5,
        M0: int | None = None,
        ef_construction: int = 32,
        mL: float | None = None,
        seed: int | None = None,
    ):
        self.M = M
        self.M0 = M0 if M0 is not None else 2 * M
        self.ef_construction = ef_construction
        self.mL = mL if mL is not None else 1.0 / math.log(M) if M > 1 else 1.0
        self.seed = seed

    @property
    def name(self) -> str:
        return "HNSW"

    @property
    def description(self) -> str:
        return (
            f"Hierarchical NSW: M={self.M}, ef={self.ef_construction}. "
            f"Multi-layer skip-list-inspired ANN graph."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n · M · log n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("M", "Max connections per layer", "int", 5, "M ≥ 2"),
            ParamInfo("M0", "Max connections on layer 0", "int", 10, "≥ M"),
            ParamInfo(
                "ef_construction", "Construction beam width",
                "int", 32, "≥ M",
            ),
            ParamInfo("mL", "Level multiplier", "float", "1/ln(M)"),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        seed = self.seed if self.seed is not None else (layout.seed + 201)
        rng = np.random.default_rng(seed)

        # Assign random levels to each node
        levels = np.array([
            self._random_level(rng) for _ in range(n)
        ], dtype=np.intp)
        max_level = int(levels.max())

        # Per-layer adjacency: layer → {node → set of neighbors}
        layer_adj: list[dict[int, set[int]]] = [
            {i: set() for i in range(n) if levels[i] >= layer}
            for layer in range(max_level + 1)
        ]

        # Entry point: node with highest level
        entry_point = int(np.argmax(levels))

        # Insertion order
        insertion_order = rng.permutation(n)
        inserted: set[int] = set()

        for node in insertion_order:
            node_level = int(levels[node])

            if not inserted:
                inserted.add(node)
                continue

            # Phase 1: Traverse from top layer to node_level + 1
            # using greedy search (ef=1) to find a good entry
            current_entry = entry_point

            for layer in range(max_level, node_level, -1):
                if node not in layer_adj[layer]:
                    # Node doesn't exist at this layer
                    if current_entry in layer_adj[layer]:
                        results = greedy_search(
                            query=points[node],
                            entry_point=current_entry,
                            points=points,
                            adj=layer_adj[layer],
                            ef=1,
                        )
                        if results:
                            current_entry = results[0][1]

            # Phase 2: Insert at layers node_level down to 0
            for layer in range(min(node_level, max_level), -1, -1):
                if current_entry not in layer_adj[layer]:
                    # Find any node in this layer as entry
                    layer_nodes = list(layer_adj[layer].keys())
                    if not layer_nodes:
                        continue
                    current_entry = layer_nodes[0]

                max_conn = self.M0 if layer == 0 else self.M
                ef = max(self.ef_construction, max_conn)

                results = greedy_search(
                    query=points[node],
                    entry_point=current_entry,
                    points=points,
                    adj=layer_adj[layer],
                    ef=ef,
                )

                # Select neighbors (simple: take closest up to max_conn)
                neighbors = [
                    idx for _, idx in results[:max_conn]
                    if idx != node and idx in layer_adj[layer]
                ]

                for nbr in neighbors:
                    layer_adj[layer][node].add(nbr)
                    layer_adj[layer][nbr].add(node)

                    # Prune if neighbor exceeds max connections
                    if len(layer_adj[layer][nbr]) > max_conn:
                        self._shrink_connections(
                            nbr, layer_adj[layer], points, max_conn
                        )

                if results:
                    current_entry = results[0][1]

            inserted.add(node)

        # Build NetworkX graph from union of all layers
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            G.nodes[i]["level"] = int(levels[i])

        for layer in range(max_level + 1):
            for u, neighbors in layer_adj[layer].items():
                for v in neighbors:
                    if v > u:
                        if G.has_edge(u, v):
                            # Add this layer to the edge's layer set
                            existing = G[u][v].get("layers", set())
                            existing.add(layer)
                            G[u][v]["layers"] = existing
                        else:
                            G.add_edge(
                                u, v,
                                weight=float(dist[u, v]),
                                layers={layer},
                            )

        G.graph["algorithm"] = "hnsw"
        G.graph["M"] = self.M
        G.graph["M0"] = self.M0
        G.graph["max_level"] = max_level
        G.graph["entry_point"] = entry_point

        return G

    def _random_level(self, rng: np.random.Generator) -> int:
        """Draw a random level from an exponential distribution."""
        r = rng.random()
        if r == 0:
            r = 1e-15
        return int(-math.log(r) * self.mL)

    @staticmethod
    def _shrink_connections(
        node: int,
        adj: dict[int, set[int]],
        points: np.ndarray,
        max_conn: int,
    ) -> None:
        """Reduce a node's connections to max_conn by keeping closest."""
        neighbors = list(adj[node])
        distances = [
            (euclidean_distance(points[node], points[nbr]), nbr)
            for nbr in neighbors
        ]
        distances.sort(key=lambda x: x[0])

        # Keep only the closest max_conn
        keep = {nbr for _, nbr in distances[:max_conn]}
        remove = set(neighbors) - keep

        for nbr in remove:
            adj[node].discard(nbr)
            adj[nbr].discard(node)

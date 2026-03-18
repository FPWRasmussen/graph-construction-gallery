"""
Vamana Graph (DiskANN).

Vamana is the graph construction algorithm behind Microsoft's DiskANN
system, designed for billion-scale ANN search with bounded memory.
It builds a degree-bounded, navigable graph using a two-pass
construction with **robust pruning** [[3]]:

    Pass 1 (random initialization):
        - Start with a random R-regular graph.
        - For each node (in random order), perform greedy search to
          find approximate neighbors, then apply robust pruning.

    Pass 2 (refinement with higher α):
        - Repeat with a higher pruning parameter α to improve
          long-range connectivity.

Robust pruning ensures angular diversity among neighbors: a candidate
is rejected if a closer already-selected neighbor "covers" the same
angular direction (within factor α).

Key properties:
    - Bounded out-degree R
    - High recall with a single entry point (medoid)
    - α parameter controls long-range edge density
    - Designed for disk-resident data (SSD-friendly access patterns)
    - Search from a single medoid entry point

Reference:
    Subramanya, S.J., Devvrit, F., Simhadri, H.V., Krishnaswamy, R.,
    & Kadekodi, R. (2019). "DiskANN: Fast Accurate Billion-point
    Nearest Neighbor Search on a Single Node." NeurIPS 2019.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.ann._ann_utils import (
    greedy_search,
    robust_prune,
    euclidean_distance,
)


class VamanaGraph(GraphBuilder):
    """Degree-bounded navigable graph via robust pruning (DiskANN).

    Parameters:
        R: Maximum out-degree (neighbor list size).
        alpha: Pruning distance factor (≥ 1.0). Higher = more
            long-range edges.
        L: Search list size during construction.
        n_passes: Number of construction passes (1 or 2).
        seed: Random seed.
    """

    slug = "vamana"
    category = "ann"

    def __init__(
        self,
        R: int = 5,
        alpha: float = 1.2,
        L: int = 20,
        n_passes: int = 2,
        seed: int | None = None,
    ):
        self.R = R
        self.alpha = alpha
        self.L = L
        self.n_passes = n_passes
        self.seed = seed

    @property
    def name(self) -> str:
        return "Vamana (DiskANN)"

    @property
    def description(self) -> str:
        return (
            f"Degree-bounded graph (R={self.R}) with robust pruning "
            f"(α={self.alpha}). Medoid entry point."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n · L · R · passes)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("R", "Maximum out-degree", "int", 5, "R ≥ 1"),
            ParamInfo(
                "alpha", "Pruning factor",
                "float", 1.2, "α ≥ 1.0",
            ),
            ParamInfo("L", "Construction search list size", "int", 20, "L ≥ R"),
            ParamInfo("n_passes", "Number of build passes", "int", 2, "1 or 2"),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        seed = self.seed if self.seed is not None else (layout.seed + 202)
        rng = np.random.default_rng(seed)

        # Find the medoid (point closest to centroid) as entry point
        centroid = points.mean(axis=0)
        medoid = int(np.argmin(
            np.linalg.norm(points - centroid, axis=1)
        ))

        # Initialize with a random R-regular-ish graph
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            others = [j for j in range(n) if j != i]
            chosen = rng.choice(
                others, size=min(self.R, len(others)), replace=False
            )
            for j in chosen:
                adj[i].add(int(j))
                adj[int(j)].add(i)

        # Construction passes
        for pass_num in range(self.n_passes):
            # Use higher alpha on the second pass for long-range edges
            alpha = self.alpha if pass_num == 0 else self.alpha * 1.2

            # Random node visitation order
            order = rng.permutation(n)

            for node in order:
                # Greedy search to find approximate nearest neighbors
                results = greedy_search(
                    query=points[node],
                    entry_point=medoid,
                    points=points,
                    adj=adj,
                    ef=self.L,
                )

                # Gather candidates: search results + current neighbors
                candidate_set: dict[int, float] = {}
                for d_nq, idx in results:
                    if idx != node:
                        candidate_set[idx] = d_nq
                for nbr in adj[node]:
                    if nbr not in candidate_set:
                        candidate_set[nbr] = euclidean_distance(
                            points[node], points[nbr]
                        )

                candidates = [
                    (d, idx) for idx, d in candidate_set.items()
                ]

                # Robust pruning
                new_neighbors = robust_prune(
                    node, candidates, points, alpha, self.R
                )

                # Update adjacency: remove old, add new
                old_neighbors = adj[node].copy()
                adj[node] = set(new_neighbors)

                # Add reverse edges
                for nbr in new_neighbors:
                    adj[nbr].add(node)

                    # If neighbor now exceeds R, prune it too
                    if len(adj[nbr]) > self.R:
                        nbr_candidates = [
                            (euclidean_distance(points[nbr], points[nb2]), nb2)
                            for nb2 in adj[nbr] if nb2 != nbr
                        ]
                        pruned = robust_prune(
                            nbr, nbr_candidates, points, alpha, self.R
                        )
                        adj[nbr] = set(pruned)

        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for u in range(n):
            for v in adj[u]:
                if v > u:
                    G.add_edge(u, v, weight=float(dist[u, v]))

        G.graph["algorithm"] = "vamana"
        G.graph["R"] = self.R
        G.graph["alpha"] = self.alpha
        G.graph["medoid"] = medoid
        G.graph["n_passes"] = self.n_passes

        return G

"""
NN-Descent Graph.

NN-Descent is an iterative algorithm for constructing an approximate
k-NN graph.  It exploits the principle that "a neighbor of a neighbor
is likely also a neighbor":

    1. Initialize each node's k-NN list with random neighbors.
    2. Repeat until convergence:
       a. For each node, explore the neighbors of its current neighbors.
       b. If a neighbor-of-neighbor is closer than the current farthest
          k-NN entry, swap it in.
    3. The number of updates decreases each iteration, converging
       rapidly (typically 5–15 iterations).

NN-Descent is remarkably efficient: O(n^1.14) empirical complexity
for typical datasets, making it one of the fastest methods for
building exact or near-exact k-NN graphs.

Reference:
    Dong, W., Moses, C., & Li, K. (2011). "Efficient k-nearest
    neighbor graph construction for generic similarity measures."
    Proc. 20th International Conference on World Wide Web (WWW),
    pp. 577–586.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.ann._ann_utils import euclidean_distance


class NNDescentGraph(GraphBuilder):
    """Approximate k-NN graph via iterative neighbor-of-neighbor exploration.

    Initializes randomly and refines by checking neighbors-of-neighbors.
    Converges rapidly to a high-recall k-NN graph.

    Parameters:
        k: Number of nearest neighbors per node.
        max_iterations: Maximum refinement iterations.
        delta: Early stopping threshold (fraction of updates).
        rho: Sampling rate for neighbor exploration.
        seed: Random seed.
    """

    slug = "nn_descent"
    category = "ann"

    def __init__(
        self,
        k: int = 5,
        max_iterations: int = 20,
        delta: float = 0.001,
        rho: float = 1.0,
        seed: int | None = None,
    ):
        self.k = k
        self.max_iterations = max_iterations
        self.delta = delta
        self.rho = rho
        self.seed = seed

    @property
    def name(self) -> str:
        return "NN-Descent"

    @property
    def description(self) -> str:
        return (
            f"Iterative k-NN refinement (k={self.k}). "
            f"'Neighbor of neighbor is likely a neighbor.'"
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n^1.14) empirical, O(n·k²·iters) worst case"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("k", "Number of nearest neighbors", "int", 5, "k ≥ 1"),
            ParamInfo(
                "max_iterations", "Max refinement iterations",
                "int", 20, "≥ 1",
            ),
            ParamInfo(
                "delta", "Early stopping threshold",
                "float", 0.001, "0 < δ < 1",
            ),
            ParamInfo(
                "rho", "Neighbor sampling rate",
                "float", 1.0, "0 < ρ ≤ 1",
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def validate_layout(self, layout: PointLayout) -> None:
        super().validate_layout(layout)
        if self.k >= layout.n_points:
            raise ValueError(
                f"k={self.k} must be < n_points={layout.n_points}"
            )

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist_matrix = pairwise_distances(points)

        seed = self.seed if self.seed is not None else (layout.seed + 203)
        rng = np.random.default_rng(seed)

        k = self.k

        # Initialize: random k neighbors per node
        # Store as a max-heap per node: [(-distance, neighbor_id), ...]
        # Max-heap so we can efficiently find and replace the farthest neighbor
        nn_lists: list[list[tuple[float, int]]] = []
        for i in range(n):
            others = [j for j in range(n) if j != i]
            chosen = rng.choice(others, size=min(k, len(others)), replace=False)
            heap = []
            for j in chosen:
                # Negative distance for max-heap via heapq (min-heap)
                heap.append((-float(dist_matrix[i, j]), int(j)))
            # Heapify
            import heapq
            heapq.heapify(heap)
            nn_lists.append(heap)

        # Track which neighbors are "new" (not yet used for exploration)
        is_new: list[set[int]] = [
            {int(j) for _, j in nn_lists[i]} for i in range(n)
        ]

        total_updates = 0

        for iteration in range(self.max_iterations):
            updates = 0

            # Build reverse neighbor lists
            reverse_nn: dict[int, set[int]] = {i: set() for i in range(n)}
            for i in range(n):
                for _, j in nn_lists[i]:
                    reverse_nn[j].add(i)

            # For each node, explore neighbors-of-neighbors
            for u in range(n):
                # Collect "new" neighbors for exploration
                new_neighbors: list[int] = []
                for _, j in nn_lists[u]:
                    if j in is_new[u]:
                        new_neighbors.append(j)

                # Sample subset if rho < 1
                if self.rho < 1.0 and len(new_neighbors) > 1:
                    sample_size = max(1, int(self.rho * len(new_neighbors)))
                    new_neighbors = rng.choice(
                        new_neighbors, size=sample_size, replace=False
                    ).tolist()

                # Also include reverse neighbors
                all_candidates: set[int] = set(new_neighbors)
                for j in new_neighbors:
                    for _, jj in nn_lists[j]:
                        if jj != u:
                            all_candidates.add(jj)

                # Check each candidate
                for candidate in all_candidates:
                    if candidate == u:
                        continue

                    d_uc = float(dist_matrix[u, candidate])

                    # Check if candidate is closer than current farthest
                    if len(nn_lists[u]) < k:
                        import heapq
                        heapq.heappush(nn_lists[u], (-d_uc, candidate))
                        is_new[u].add(candidate)
                        updates += 1
                    elif d_uc < -nn_lists[u][0][0]:
                        # Check if candidate is already in the list
                        current_ids = {j for _, j in nn_lists[u]}
                        if candidate not in current_ids:
                            import heapq
                            heapq.heapreplace(nn_lists[u], (-d_uc, candidate))
                            is_new[u].add(candidate)
                            updates += 1

                # Clear "new" flags for explored neighbors
                is_new[u] = set()

            total_updates += updates

            # Early stopping
            update_rate = updates / (n * k) if n * k > 0 else 0
            if update_rate < self.delta:
                break

        # Build graph from final nn_lists (undirected)
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for u in range(n):
            for neg_d, v in nn_lists[u]:
                if v > u:
                    G.add_edge(u, v, weight=float(-neg_d))
                elif not G.has_edge(v, u):
                    G.add_edge(v, u, weight=float(-neg_d))

        G.graph["algorithm"] = "nn_descent"
        G.graph["k"] = k
        G.graph["iterations"] = iteration + 1
        G.graph["total_updates"] = total_updates

        return G

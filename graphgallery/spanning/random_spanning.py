"""
Random Spanning Tree — Wilson's Algorithm.

Wilson's algorithm generates a **uniformly random spanning tree**
(UST) from any connected graph.  It uses *loop-erased random walks*:

    1. Pick an arbitrary root vertex r and mark it as part of the tree.
    2. Pick any non-tree vertex u and start a random walk from u.
    3. Whenever the walk revisits a vertex, erase the loop created
       (keeping only the first visit to each vertex).
    4. When the walk hits the tree, add the loop-erased path to the tree.
    5. Repeat until all vertices are in the tree.

The expected running time is proportional to the mean hitting time
of the random walk, which is O(n²) for complete graphs and dense
expanders, but can be much worse for sparse or poorly-connected graphs.

Uniform spanning trees have beautiful mathematical properties:
    - The marginal probability of an edge is its effective resistance.
    - They connect to the Matrix-Tree theorem and determinantal
      point processes.
    - They provide natural random subgraph samples for testing.

Reference:
    Wilson, D.B. (1996). "Generating random spanning trees more
    quickly than the cover time." Proceedings of the 28th Annual
    ACM Symposium on Theory of Computing (STOC), pp. 296–303.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class RandomSpanningTreeGraph(GraphBuilder):
    """Uniformly random spanning tree via Wilson's loop-erased random walk.

    Samples a spanning tree uniformly at random from the complete
    Euclidean graph on the input points.

    Parameters:
        seed: Random seed for the walk (default: layout seed + 7).
        use_weights: If True, the random walk transitions are biased
            by inverse distance (closer neighbors more likely). If
            False, the walk is unweighted (uniform neighbor selection).
    """

    slug = "random_spanning"
    category = "spanning"

    def __init__(self, seed: int | None = None, use_weights: bool = False):
        self.seed = seed
        self.use_weights = use_weights

    @property
    def name(self) -> str:
        return "Random Spanning Tree"

    @property
    def description(self) -> str:
        return (
            "Uniformly random spanning tree via Wilson's loop-erased "
            "random walk algorithm."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n²) expected for complete graphs (cover time)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "seed",
                "Random seed (None = layout seed + 7)",
                "int | None",
                None,
            ),
            ParamInfo(
                "use_weights",
                "Bias walk by inverse distance",
                "bool",
                False,
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        dist = pairwise_distances(layout.points)

        seed = self.seed if self.seed is not None else (layout.seed + 7)
        rng = np.random.default_rng(seed)

        # Precompute transition probabilities for each vertex
        transition_probs = self._build_transition_probs(dist, n, rng)

        # --- Wilson's algorithm ---
        in_tree = np.zeros(n, dtype=bool)
        # next_in_path[v] = the successor of v on the current
        # loop-erased walk toward the tree
        next_in_path = np.full(n, -1, dtype=np.intp)

        # Root: first vertex
        root = 0
        in_tree[root] = True

        tree_edges: list[tuple[int, int]] = []

        for start in range(n):
            if in_tree[start]:
                continue

            # Phase 1: Random walk from start until we hit the tree.
            # Record next_in_path to keep only the loop-erased version.
            current = start
            while not in_tree[current]:
                next_vertex = self._step(current, n, transition_probs, rng)
                next_in_path[current] = next_vertex
                current = next_vertex

            # Phase 2: Follow the loop-erased path from start to tree,
            # adding each edge.
            current = start
            while not in_tree[current]:
                in_tree[current] = True
                nxt = next_in_path[current]
                tree_edges.append((current, nxt))
                current = nxt

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v in tree_edges:
            G.add_edge(u, v, weight=float(dist[u, v]))

        G.graph["algorithm"] = "wilson"
        G.graph["total_weight"] = sum(
            float(dist[u, v]) for u, v in tree_edges
        )

        return G

    def _build_transition_probs(
        self,
        dist: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build the transition probability matrix.

        Returns:
            (n, n) array where row i sums to 1.  Self-loops are zero.
        """
        if self.use_weights:
            # Inverse distance weighting (closer = more likely)
            with np.errstate(divide="ignore"):
                inv_dist = np.where(dist > 0, 1.0 / dist, 0.0)
            np.fill_diagonal(inv_dist, 0.0)
            row_sums = inv_dist.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # Prevent division by zero
            return inv_dist / row_sums
        else:
            # Uniform: equal probability for all other vertices
            probs = np.ones((n, n), dtype=np.float64)
            np.fill_diagonal(probs, 0.0)
            return probs / probs.sum(axis=1, keepdims=True)

    def _step(
        self,
        current: int,
        n: int,
        transition_probs: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Take one step of the random walk from current."""
        return int(rng.choice(n, p=transition_probs[current]))

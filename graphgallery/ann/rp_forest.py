"""
Random Projection Forest Graph.

A random projection (RP) forest partitions the point set using
random hyperplane splits, creating a collection of randomized
space-partitioning trees.  Points that frequently end up in the
same leaf across multiple trees are considered likely neighbors.

Construction:
    1. Build T random projection trees:
       a. At each internal node, pick a random hyperplane (direction).
       b. Split points by which side of the hyperplane they fall on.
       c. Continue until leaves have ≤ leaf_size points.
    2. For each pair of points co-occurring in the same leaf,
       record them as candidate neighbors.
    3. Build a k-NN graph from the candidate pairs.

RP forests are the basis of the Annoy library (Spotify) and serve
as an initialization method for algorithms like NN-Descent.

Key properties:
    - O(n · T · log n) construction
    - Quality improves with more trees T
    - No distance computations during tree building (only projections)
    - Good for high-dimensional data where tree-based methods
      (KD-tree) degrade

Reference:
    Dasgupta, S. & Freund, Y. (2008). "Random Projection Trees and
    Low Dimensional Manifolds." Proc. 40th ACM STOC, pp. 537–546.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


@dataclass
class _RPNode:
    """Node in a random projection tree."""
    indices: np.ndarray          # Point indices at this node
    split_vector: np.ndarray | None = None   # Random direction
    split_threshold: float = 0.0
    left: Optional[_RPNode] = None
    right: Optional[_RPNode] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _build_rp_tree(
    points: np.ndarray,
    indices: np.ndarray,
    leaf_size: int,
    rng: np.random.Generator,
) -> _RPNode:
    """Recursively build a random projection tree.

    Args:
        points: (n, d) full point array.
        indices: Indices of points at this node.
        leaf_size: Maximum points per leaf.
        rng: Random number generator.

    Returns:
        Root _RPNode.
    """
    node = _RPNode(indices=indices)

    if len(indices) <= leaf_size:
        return node

    # Random split direction (unit vector)
    d = points.shape[1]
    direction = rng.standard_normal(d)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return node
    direction /= norm

    # Project points onto the direction
    projections = points[indices] @ direction

    # Split at median
    threshold = float(np.median(projections))

    left_mask = projections <= threshold
    right_mask = ~left_mask

    # Avoid empty splits
    if not left_mask.any() or not right_mask.any():
        return node

    node.split_vector = direction
    node.split_threshold = threshold
    node.left = _build_rp_tree(points, indices[left_mask], leaf_size, rng)
    node.right = _build_rp_tree(points, indices[right_mask], leaf_size, rng)

    return node


def _collect_leaves(node: _RPNode) -> list[np.ndarray]:
    """Collect all leaf index sets from an RP tree."""
    if node.is_leaf:
        return [node.indices]
    leaves = []
    if node.left is not None:
        leaves.extend(_collect_leaves(node.left))
    if node.right is not None:
        leaves.extend(_collect_leaves(node.right))
    return leaves


class RPForestGraph(GraphBuilder):
    """k-NN graph constructed from random projection forest co-occurrences.

    Points sharing leaves across multiple trees become candidate
    neighbors.

    Parameters:
        k: Number of neighbors per node.
        n_trees: Number of random projection trees.
        leaf_size: Maximum leaf size.
        seed: Random seed.
    """

    slug = "rp_forest"
    category = "ann"

    def __init__(
        self,
        k: int = 5,
        n_trees: int = 10,
        leaf_size: int = 5,
        seed: int | None = None,
    ):
        self.k = k
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.seed = seed

    @property
    def name(self) -> str:
        return "RP-Forest Graph"

    @property
    def description(self) -> str:
        return (
            f"k-NN via {self.n_trees} random projection trees "
            f"(leaf_size={self.leaf_size})."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n · T · log n)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("k", "Neighbors per node", "int", 5, "k ≥ 1"),
            ParamInfo("n_trees", "Number of RP trees", "int", 10, "T ≥ 1"),
            ParamInfo("leaf_size", "Max leaf size", "int", 5, "≥ 2"),
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

        seed = self.seed if self.seed is not None else (layout.seed + 204)
        rng = np.random.default_rng(seed)

        # Count co-occurrences across all trees
        cooccurrence: defaultdict[tuple[int, int], int] = defaultdict(int)

        indices = np.arange(n, dtype=np.intp)

        for tree_idx in range(self.n_trees):
            tree = _build_rp_tree(points, indices, self.leaf_size, rng)
            leaves = _collect_leaves(tree)

            for leaf_indices in leaves:
                # All pairs in this leaf are candidate neighbors
                leaf_list = leaf_indices.tolist()
                for a_idx in range(len(leaf_list)):
                    for b_idx in range(a_idx + 1, len(leaf_list)):
                        u, v = leaf_list[a_idx], leaf_list[b_idx]
                        key = (min(u, v), max(u, v))
                        cooccurrence[key] += 1

        # Build k-NN from candidates, ranked by distance
        # For each node, collect all candidate neighbors and pick closest k
        candidates: dict[int, list[tuple[float, int]]] = {
            i: [] for i in range(n)
        }

        for (u, v), count in cooccurrence.items():
            d = float(dist_matrix[u, v])
            candidates[u].append((d, v))
            candidates[v].append((d, u))

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for u in range(n):
            # Sort by distance, take top k
            candidates[u].sort(key=lambda x: x[0])
            seen: set[int] = set()
            count = 0
            for d, v in candidates[u]:
                if v in seen:
                    continue
                seen.add(v)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=d)
                count += 1
                if count >= self.k:
                    break

        G.graph["algorithm"] = "rp_forest"
        G.graph["n_trees"] = self.n_trees
        G.graph["leaf_size"] = self.leaf_size
        G.graph["n_candidates"] = len(cooccurrence)

        return G

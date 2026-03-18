"""
WSPD Spanner (Well-Separated Pair Decomposition).

A Well-Separated Pair Decomposition (WSPD) partitions the set of
all point pairs into groups (A_i, B_i) such that within each group,
all points in A_i are "far" from all points in B_i relative to their
internal diameters.  Formally, a pair (A, B) is s-well-separated if:

    dist(A, B) ≥ s · max(diam(A), diam(B))

where s > 0 is the separation factor.

A WSPD spanner selects one representative edge per well-separated
pair:  pick any a ∈ A and b ∈ B and add edge (a, b).  The resulting
graph is a (1 + ε)-spanner where ε depends on s.

Construction via split tree (fair split tree):
    1. Build a split tree by recursively splitting the bounding box
       along its longest axis at the midpoint.
    2. Enumerate well-separated pairs by walking the split tree.
    3. For each pair, add a representative edge.

Properties:
    - O(s^d · n) pairs and edges for d-dimensional points
    - Stretch factor t = 1 + 4/(s - 2) for s > 2 (in 2D)
    - O(n log n + n·s²) construction time
    - Theoretical foundation for many spanner algorithms

Reference:
    Callahan, P.B. & Kosaraju, S.R. (1995). "A decomposition of
    multidimensional point sets with applications to k-nearest-
    neighbors and n-body potential fields." JACM, 42(1), 67–90.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances
from graphgallery.spanners._spanner_utils import compute_stretch_factor


# ---------------------------------------------------------------------------
# Split tree data structure
# ---------------------------------------------------------------------------

@dataclass
class _SplitNode:
    """Node in a fair split tree for WSPD construction."""

    point_indices: np.ndarray  # Indices into the original point array
    bbox_min: np.ndarray       # Bounding box minimum (2,)
    bbox_max: np.ndarray       # Bounding box maximum (2,)
    left: Optional[_SplitNode] = None
    right: Optional[_SplitNode] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def diameter(self) -> float:
        """Bounding box diameter (L∞ → L2 approximation)."""
        return float(np.linalg.norm(self.bbox_max - self.bbox_min))

    @property
    def center(self) -> np.ndarray:
        return (self.bbox_min + self.bbox_max) / 2.0

    @property
    def size(self) -> int:
        return len(self.point_indices)

    def representative(self) -> int:
        """Return one representative point index."""
        return int(self.point_indices[0])


def _build_split_tree(
    points: np.ndarray,
    indices: np.ndarray,
) -> _SplitNode:
    """Recursively build a fair split tree.

    Splits along the longest bounding box axis at the midpoint.

    Args:
        points: (n, 2) full point array.
        indices: Indices of points in this subtree.

    Returns:
        Root _SplitNode.
    """
    pts = points[indices]
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)

    node = _SplitNode(
        point_indices=indices,
        bbox_min=bbox_min.copy(),
        bbox_max=bbox_max.copy(),
    )

    if len(indices) <= 1:
        return node

    # Split along the longest axis
    extents = bbox_max - bbox_min
    split_axis = int(np.argmax(extents))
    mid = (bbox_min[split_axis] + bbox_max[split_axis]) / 2.0

    left_mask = pts[:, split_axis] <= mid
    right_mask = ~left_mask

    # Handle edge cases where all points fall on one side
    if not left_mask.any():
        left_mask[0] = True
        right_mask[0] = False
    elif not right_mask.any():
        right_mask[-1] = True
        left_mask[-1] = False

    node.left = _build_split_tree(points, indices[left_mask])
    node.right = _build_split_tree(points, indices[right_mask])

    return node


def _node_distance(a: _SplitNode, b: _SplitNode) -> float:
    """Minimum distance between two bounding boxes."""
    # Clamp-based bbox distance
    dx = max(0.0, max(a.bbox_min[0] - b.bbox_max[0],
                      b.bbox_min[0] - a.bbox_max[0]))
    dy = max(0.0, max(a.bbox_min[1] - b.bbox_max[1],
                      b.bbox_min[1] - a.bbox_max[1]))
    return np.sqrt(dx * dx + dy * dy)


def _find_wspd_pairs(
    u: _SplitNode,
    v: _SplitNode,
    s: float,
    pairs: list[tuple[_SplitNode, _SplitNode]],
) -> None:
    """Recursively enumerate s-well-separated pairs.

    Args:
        u, v: Split tree nodes to compare.
        s: Separation factor.
        pairs: Accumulator list for discovered pairs.
    """
    if u.size == 0 or v.size == 0:
        return

    # Check if (u, v) is s-well-separated
    dist_uv = _node_distance(u, v)
    max_diam = max(u.diameter, v.diameter)

    if max_diam < 1e-15 or dist_uv >= s * max_diam:
        # Well-separated: record the pair
        pairs.append((u, v))
        return

    # Not well-separated: split the larger node and recurse
    if u.diameter >= v.diameter:
        if u.is_leaf:
            # Can't split further — force accept
            pairs.append((u, v))
            return
        _find_wspd_pairs(u.left, v, s, pairs)
        _find_wspd_pairs(u.right, v, s, pairs)
    else:
        if v.is_leaf:
            pairs.append((u, v))
            return
        _find_wspd_pairs(u, v.left, s, pairs)
        _find_wspd_pairs(u, v.right, s, pairs)


class WSPDSpannerGraph(GraphBuilder):
    """Spanner via Well-Separated Pair Decomposition.

    Builds a split tree, enumerates well-separated pairs, and adds
    one representative edge per pair.

    Parameters:
        s: Separation factor.  Higher s = sparser graph, larger stretch.
            Stretch factor ≈ 1 + 4/(s-2) for s > 2.
    """

    slug = "wspd_spanner"
    category = "spanners"

    def __init__(self, s: float = 4.0):
        if s <= 0:
            raise ValueError(f"Separation factor s must be > 0, got {s}")
        self.s = s

    @property
    def name(self) -> str:
        return "WSPD Spanner"

    @property
    def description(self) -> str:
        if self.s > 2:
            t = 1.0 + 4.0 / (self.s - 2.0)
            return (
                f"Well-Separated Pair Decomposition spanner (s={self.s}). "
                f"Theoretical stretch ≤ {t:.2f}."
            )
        return f"WSPD spanner with separation factor s={self.s}."

    @property
    def complexity(self) -> str:
        return "O(n log n + n·s²)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "s", "Separation factor",
                "float", 4.0, "s > 0; s > 2 for theoretical bounds",
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        points = layout.points
        dist = pairwise_distances(points)

        # Build split tree
        indices = np.arange(n, dtype=np.intp)
        root = _build_split_tree(points, indices)

        # Enumerate well-separated pairs
        pairs: list[tuple[_SplitNode, _SplitNode]] = []

        if root.left is not None and root.right is not None:
            _find_wspd_pairs(root.left, root.right, self.s, pairs)
            # Also need pairs within each subtree
            self._enumerate_internal(root.left, self.s, pairs)
            self._enumerate_internal(root.right, self.s, pairs)

        # Add one representative edge per pair
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for u_node, v_node in pairs:
            a = u_node.representative()
            b = v_node.representative()
            if a != b and not G.has_edge(a, b):
                G.add_edge(a, b, weight=float(dist[a, b]))

        G.graph["algorithm"] = "wspd_spanner"
        G.graph["separation_factor"] = self.s
        G.graph["n_pairs"] = len(pairs)
        G.graph["actual_stretch"] = compute_stretch_factor(G, dist)

        return G

    def _enumerate_internal(
        self,
        node: _SplitNode,
        s: float,
        pairs: list[tuple[_SplitNode, _SplitNode]],
    ) -> None:
        """Recursively enumerate well-separated pairs within a subtree."""
        if node.is_leaf or node.size <= 1:
            return

        if node.left is not None and node.right is not None:
            _find_wspd_pairs(node.left, node.right, s, pairs)
            self._enumerate_internal(node.left, s, pairs)
            self._enumerate_internal(node.right, s, pairs)

"""
LSH-Based Graph (Locality-Sensitive Hashing).

Locality-Sensitive Hashing (LSH) maps similar points to the same
hash bucket with high probability.  By building multiple independent
hash tables, points that are true neighbors are very likely to
collide in at least one table.

This builder uses **random hyperplane LSH** (SimHash), which is
suited for cosine/Euclidean similarity:

    h(x) = sign(⟨r, x⟩)

where r is a random Gaussian vector.  Multiple bits form a hash code,
and points with the same code in any table become candidate neighbors.

Construction:
    1. Generate L hash tables, each with b random hyperplane bits.
    2. Hash all points into each table.
    3. For each bucket with > 1 point, form candidate pairs.
    4. Build a k-NN graph from the candidates using true distances.

Key properties:
    - Sublinear query time: O(n^(1/c)) for c-approximate NN
    - Quality controlled by n_tables (L) and n_bits (b)
    - Simple to implement and parallelize
    - Foundational technique in the E2LSH and FALCONN libraries

Reference:
    Indyk, P. & Motwani, R. (1998). "Approximate Nearest Neighbors:
    Towards Removing the Curse of Dimensionality." Proc. 30th ACM
    STOC, pp. 604–613.

    Charikar, M.S. (2002). "Similarity Estimation Techniques from
    Rounding Algorithms." Proc. 34th ACM STOC, pp. 380–388.
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout, pairwise_distances


class LSHGraph(GraphBuilder):
    """k-NN graph via Locality-Sensitive Hashing with random hyperplanes.

    Multiple hash tables with random hyperplane bits identify
    candidate neighbor pairs.  True distances resolve the final
    k-NN graph.

    Parameters:
        k: Number of neighbors per node.
        n_tables: Number of independent hash tables (L).
        n_bits: Number of hyperplane bits per hash code (b).
        seed: Random seed.
    """

    slug = "lsh"
    category = "ann"

    def __init__(
        self,
        k: int = 5,
        n_tables: int = 10,
        n_bits: int = 8,
        seed: int | None = None,
    ):
        self.k = k
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.seed = seed

    @property
    def name(self) -> str:
        return "LSH-Based Graph"

    @property
    def description(self) -> str:
        return (
            f"k-NN via {self.n_tables} hash tables × {self.n_bits} bits. "
            f"Random hyperplane LSH."
        )

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n · L · b + candidates)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo("k", "Neighbors per node", "int", 5, "k ≥ 1"),
            ParamInfo("n_tables", "Number of hash tables (L)", "int", 10, "L ≥ 1"),
            ParamInfo("n_bits", "Hash bits per table (b)", "int", 8, "b ≥ 1"),
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
        d = layout.points.shape[1]
        points = layout.points
        dist_matrix = pairwise_distances(points)

        seed = self.seed if self.seed is not None else (layout.seed + 205)
        rng = np.random.default_rng(seed)

        # Generate random hyperplanes for each table
        # Shape: (n_tables, n_bits, d)
        hyperplanes = rng.standard_normal((self.n_tables, self.n_bits, d))

        # Candidate pairs from hash collisions
        candidate_set: set[tuple[int, int]] = set()

        for table_idx in range(self.n_tables):
            H = hyperplanes[table_idx]  # (n_bits, d)

            # Compute hash codes: sign of dot products
            # projections shape: (n, n_bits)
            projections = points @ H.T
            hash_codes = (projections >= 0).astype(np.int8)

            # Convert each hash code to a tuple for bucketing
            buckets: defaultdict[tuple, list[int]] = defaultdict(list)
            for i in range(n):
                code = tuple(hash_codes[i].tolist())
                buckets[code].append(i)

            # All pairs within each bucket are candidates
            for bucket_points in buckets.values():
                if len(bucket_points) < 2:
                    continue
                for a_idx in range(len(bucket_points)):
                    for b_idx in range(a_idx + 1, len(bucket_points)):
                        u, v = bucket_points[a_idx], bucket_points[b_idx]
                        candidate_set.add((min(u, v), max(u, v)))

        # Build k-NN from candidates
        neighbors: dict[int, list[tuple[float, int]]] = {
            i: [] for i in range(n)
        }

        for u, v in candidate_set:
            d_uv = float(dist_matrix[u, v])
            neighbors[u].append((d_uv, v))
            neighbors[v].append((d_uv, u))

        # If a node has no candidates (hash isolation), add brute-force
        # fallback with a few random neighbors
        for i in range(n):
            if not neighbors[i]:
                # Fallback: add k random neighbors
                others = rng.choice(
                    [j for j in range(n) if j != i],
                    size=min(self.k, n - 1),
                    replace=False,
                )
                for j in others:
                    d_ij = float(dist_matrix[i, int(j)])
                    neighbors[i].append((d_ij, int(j)))

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for u in range(n):
            # Sort by distance, take top k
            neighbors[u].sort(key=lambda x: x[0])
            seen: set[int] = set()
            count = 0
            for d_uv, v in neighbors[u]:
                if v in seen:
                    continue
                seen.add(v)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=d_uv)
                count += 1
                if count >= self.k:
                    break

        G.graph["algorithm"] = "lsh"
        G.graph["n_tables"] = self.n_tables
        G.graph["n_bits"] = self.n_bits
        G.graph["n_candidates"] = len(candidate_set)

        return G

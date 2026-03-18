"""
Shared utilities for ANN graph construction.

Provides greedy search, neighbor management, and recall computation
used across multiple ANN graph algorithms.
"""

from __future__ import annotations

import heapq

import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def greedy_search(
    query: np.ndarray,
    entry_point: int,
    points: np.ndarray,
    adj: dict[int, set[int]],
    ef: int = 1,
) -> list[tuple[float, int]]:
    """Greedy beam search on a navigable graph.

    Starting from ``entry_point``, greedily explores neighbors to
    find the closest nodes to ``query``.  Uses a beam width of
    ``ef`` (search expansion factor) to maintain a candidate set.

    Args:
        query: (d,) query vector.
        entry_point: Starting node index.
        points: (n, d) array of all point coordinates.
        adj: Adjacency dict mapping node → set of neighbors.
        ef: Beam width / search list size.

    Returns:
        List of (distance, node_index) sorted by distance,
        containing up to ``ef`` nearest results.
    """
    # visited set to avoid re-processing
    visited: set[int] = set()

    # candidates: min-heap of (distance, node) — nodes to explore
    d_entry = euclidean_distance(query, points[entry_point])
    candidates: list[tuple[float, int]] = [(d_entry, entry_point)]

    # results: max-heap of (-distance, node) — best found so far
    results: list[tuple[float, int]] = [(-d_entry, entry_point)]

    visited.add(entry_point)

    while candidates:
        d_c, c = heapq.heappop(candidates)

        # If the closest candidate is farther than our worst result,
        # and we have enough results, stop
        if len(results) >= ef and d_c > -results[0][0]:
            break

        # Explore neighbors of c
        for neighbor in adj.get(c, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)

            d_n = euclidean_distance(query, points[neighbor])

            # Add to results if better than worst, or if we need more
            if len(results) < ef:
                heapq.heappush(candidates, (d_n, neighbor))
                heapq.heappush(results, (-d_n, neighbor))
            elif d_n < -results[0][0]:
                heapq.heappush(candidates, (d_n, neighbor))
                heapq.heapreplace(results, (-d_n, neighbor))

    # Convert results to sorted (distance, node) list
    result_list = [(-neg_d, node) for neg_d, node in results]
    result_list.sort(key=lambda x: x[0])

    return result_list


def robust_prune(
    node: int,
    candidates: list[tuple[float, int]],
    points: np.ndarray,
    alpha: float,
    R: int,
) -> list[int]:
    """Robust pruning procedure (used in Vamana / DiskANN).

    Selects up to R neighbors from candidates such that no selected
    neighbor is "dominated" by a closer already-selected neighbor.
    A candidate c is dominated by selected neighbor p if:

        α · d(p, c) ≤ d(node, c)

    This ensures angular diversity in the neighbor set, improving
    search navigability.

    Args:
        node: The node whose neighbors we are selecting.
        candidates: List of (distance_to_node, candidate_index).
        points: (n, d) point coordinates.
        alpha: Pruning parameter (≥ 1.0). Higher = more aggressive pruning.
        R: Maximum number of neighbors.

    Returns:
        List of selected neighbor indices (up to R).
    """
    # Sort candidates by distance to node
    candidates = sorted(candidates, key=lambda x: x[0])

    selected: list[int] = []

    for d_nc, c in candidates:
        if c == node:
            continue
        if len(selected) >= R:
            break

        # Check if c is dominated by any already-selected neighbor
        dominated = False
        for p in selected:
            d_pc = euclidean_distance(points[p], points[c])
            if alpha * d_pc <= d_nc:
                dominated = True
                break

        if not dominated:
            selected.append(c)

    return selected


def compute_recall(
    approx_neighbors: list[list[int]],
    exact_neighbors: np.ndarray,
    k: int,
) -> float:
    """Compute recall@k of approximate vs. exact neighbors.

    Args:
        approx_neighbors: List of approximate neighbor index lists.
        exact_neighbors: (n, k) array of exact k-NN indices.
        k: Number of neighbors to evaluate.

    Returns:
        Mean recall in [0, 1].
    """
    n = len(approx_neighbors)
    total_recall = 0.0

    for i in range(n):
        exact_set = set(exact_neighbors[i, :k].tolist())
        approx_set = set(approx_neighbors[i][:k])
        total_recall += len(exact_set & approx_set) / k

    return total_recall / n

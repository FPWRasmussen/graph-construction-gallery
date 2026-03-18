"""
Shared utilities for geometric spanner construction.

Provides angular cone partitioning, stretch factor verification,
and shortest-path helpers used across multiple spanner algorithms.
"""

from __future__ import annotations

import heapq
from typing import Sequence

import networkx as nx
import numpy as np


def compute_stretch_factor(
    G: nx.Graph,
    dist_matrix: np.ndarray,
) -> float:
    """Compute the exact stretch factor (dilation) of a graph.

    The stretch factor is the maximum ratio of shortest-path distance
    in G to Euclidean distance, over all pairs of vertices:

        t = max_{p,q}  δ_G(p, q) / |pq|

    Args:
        G: A weighted NetworkX graph.
        dist_matrix: (n, n) Euclidean distance matrix.

    Returns:
        The stretch factor t ≥ 1.  Returns inf if G is disconnected.
    """
    n = dist_matrix.shape[0]

    # Compute all-pairs shortest paths in the graph
    try:
        sp_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    except nx.NetworkXError:
        return float("inf")

    max_stretch = 1.0

    for i in range(n):
        if i not in sp_lengths:
            return float("inf")
        for j in range(i + 1, n):
            euclidean = dist_matrix[i, j]
            if euclidean < 1e-15:
                continue

            if j not in sp_lengths[i]:
                return float("inf")

            graph_dist = sp_lengths[i][j]
            stretch = graph_dist / euclidean
            if stretch > max_stretch:
                max_stretch = stretch

    return max_stretch


def verify_t_spanner(
    G: nx.Graph,
    dist_matrix: np.ndarray,
    t: float,
) -> tuple[bool, float]:
    """Check whether a graph is a valid t-spanner.

    Args:
        G: The spanner graph.
        dist_matrix: Euclidean distance matrix.
        t: Target stretch factor.

    Returns:
        (is_valid, actual_stretch) tuple.
    """
    actual = compute_stretch_factor(G, dist_matrix)
    return actual <= t + 1e-9, actual


def cone_partition_angles(k: int) -> np.ndarray:
    """Compute the boundary angles for k equal cones centered at each point.

    Cones partition the plane into k sectors of angle 2π/k each.
    Cone i covers the angular range [i·2π/k, (i+1)·2π/k).

    Args:
        k: Number of cones.

    Returns:
        (k+1,) array of boundary angles in [0, 2π].
    """
    return np.linspace(0, 2 * np.pi, k + 1)


def angle_to_point(origin: np.ndarray, target: np.ndarray) -> float:
    """Compute the angle from origin to target in [0, 2π).

    Args:
        origin: (2,) coordinates.
        target: (2,) coordinates.

    Returns:
        Angle in radians in [0, 2π).
    """
    diff = target - origin
    angle = np.arctan2(diff[1], diff[0])
    if angle < 0:
        angle += 2 * np.pi
    return angle


def assign_cone(angle: float, cone_boundaries: np.ndarray) -> int:
    """Determine which cone an angle falls into.

    Args:
        angle: Angle in [0, 2π).
        cone_boundaries: (k+1,) sorted boundary array.

    Returns:
        Cone index in [0, k-1].
    """
    k = len(cone_boundaries) - 1
    # Binary search for the cone
    idx = int(np.searchsorted(cone_boundaries, angle, side="right")) - 1
    return max(0, min(idx, k - 1))


def dijkstra_single_source(
    adj: dict[int, list[tuple[int, float]]],
    source: int,
    n: int,
) -> np.ndarray:
    """Single-source Dijkstra on an adjacency list.

    Args:
        adj: adjacency list mapping node → [(neighbor, weight), ...].
        source: Source node index.
        n: Total number of nodes.

    Returns:
        (n,) array of shortest-path distances from source.
        Unreachable nodes have distance inf.
    """
    dist = np.full(n, np.inf)
    dist[source] = 0.0
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj.get(u, []):
            new_d = d + w
            if new_d < dist[v]:
                dist[v] = new_d
                heapq.heappush(heap, (new_d, v))

    return dist

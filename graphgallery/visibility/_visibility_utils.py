"""
Shared utilities for visibility graph construction.

Provides line-of-sight tests, time-series preparation, segment
intersection checks, and analysis helpers used across the three
visibility graph algorithms.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from graphgallery.points import PointLayout


# ---------------------------------------------------------------------------
# Time-series preparation
# ---------------------------------------------------------------------------

def layout_to_time_series(
    layout: PointLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a 2D point layout into a time series by sorting on x.

    Points are sorted by x-coordinate.  The sorted x-values become
    the "time" axis and the sorted y-values become the "series values".

    Args:
        layout: The point layout to convert.

    Returns:
        (times, values, sort_order) where:
            - times:  (n,) sorted x-coordinates
            - values: (n,) y-coordinates in sorted order
            - sort_order: (n,) index array mapping sorted → original indices
    """
    sort_order = np.argsort(layout.points[:, 0])
    times = layout.points[sort_order, 0]
    values = layout.points[sort_order, 1]
    return times, values, sort_order


def original_index(sort_order: np.ndarray, sorted_idx: int) -> int:
    """Map a sorted-order index back to the original layout index.

    Args:
        sort_order: Index array from layout_to_time_series.
        sorted_idx: Index in the sorted array.

    Returns:
        Corresponding index in the original layout.
    """
    return int(sort_order[sorted_idx])


# ---------------------------------------------------------------------------
# Geometric line-of-sight tests
# ---------------------------------------------------------------------------

def segments_intersect(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    epsilon: float = 1e-10,
) -> bool:
    """Test whether line segments (p1,p2) and (p3,p4) properly intersect.

    Uses the cross-product orientation test.  Touching at endpoints
    is NOT counted as intersection (proper intersection only).

    Args:
        p1, p2: Endpoints of first segment (each (2,) arrays).
        p3, p4: Endpoints of second segment.
        epsilon: Numerical tolerance for collinearity.

    Returns:
        True if the segments properly intersect.
    """
    d1 = _cross_2d(p3, p4, p1)
    d2 = _cross_2d(p3, p4, p2)
    d3 = _cross_2d(p1, p2, p3)
    d4 = _cross_2d(p1, p2, p4)

    if ((d1 > epsilon and d2 < -epsilon) or (d1 < -epsilon and d2 > epsilon)) and \
       ((d3 > epsilon and d4 < -epsilon) or (d3 < -epsilon and d4 > epsilon)):
        return True

    return False


def _cross_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the 2D cross product (b-a) × (c-a).

    Positive if c is counter-clockwise from (a→b).
    Negative if clockwise.
    Zero if collinear.
    """
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def point_in_polygon(
    point: np.ndarray,
    polygon: np.ndarray,
) -> bool:
    """Test if a point lies inside a polygon using ray casting.

    Args:
        point: (2,) array.
        polygon: (m, 2) array of polygon vertices (closed or open).

    Returns:
        True if point is inside the polygon.
    """
    n = len(polygon)
    inside = False
    px, py = point

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


# ---------------------------------------------------------------------------
# Natural Visibility test
# ---------------------------------------------------------------------------

def natural_visibility_test(
    times: np.ndarray,
    values: np.ndarray,
    i: int,
    j: int,
) -> bool:
    """Test whether data points i and j have natural visibility.

    Two points (t_i, y_i) and (t_j, y_j) in a time series are
    mutually visible if for every intermediate point k (i < k < j):

        y_k < y_i + (y_j - y_i) · (t_k - t_i) / (t_j - t_i)

    That is, all intermediate points lie strictly below the straight
    line connecting (t_i, y_i) and (t_j, y_j) [[8]].

    Args:
        times: (n,) sorted time values.
        values: (n,) series values.
        i: Index of the first point (i < j).
        j: Index of the second point.

    Returns:
        True if the two points are naturally visible.
    """
    if abs(j - i) <= 1:
        return True  # Adjacent points are always visible

    t_i, y_i = times[i], values[i]
    t_j, y_j = times[j], values[j]
    dt = t_j - t_i

    if abs(dt) < 1e-15:
        return False  # Same time → not visible

    slope = (y_j - y_i) / dt

    for k in range(i + 1, j):
        t_k = times[k]
        y_k = values[k]

        # Height of the line at t_k
        line_height = y_i + slope * (t_k - t_i)

        if y_k >= line_height:
            return False  # Point k blocks visibility

    return True


# ---------------------------------------------------------------------------
# Horizontal Visibility test
# ---------------------------------------------------------------------------

def horizontal_visibility_test(
    values: np.ndarray,
    i: int,
    j: int,
) -> bool:
    """Test whether data points i and j have horizontal visibility.

    Two points (t_i, y_i) and (t_j, y_j) are horizontally visible
    if for every intermediate point k (i < k < j):

        y_k < min(y_i, y_j)

    This is a simplified version of the natural visibility test that
    uses a horizontal line instead of a sloped line [[4]].

    Args:
        values: (n,) series values.
        i: Index of the first point (i < j).
        j: Index of the second point.

    Returns:
        True if the two points are horizontally visible.
    """
    if abs(j - i) <= 1:
        return True  # Adjacent points are always visible

    threshold = min(values[i], values[j])

    for k in range(i + 1, j):
        if values[k] >= threshold:
            return False  # Point k blocks horizontal visibility

    return True


# ---------------------------------------------------------------------------
# Obstacle generation for geometric visibility
# ---------------------------------------------------------------------------

def generate_obstacles_from_layout(
    layout: PointLayout,
    n_obstacles: int = 4,
    obstacle_radius: float = 0.3,
    n_vertices: int = 4,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate rectangular obstacles scattered between clusters.

    Creates obstacles positioned between and around the point clusters
    to produce an interesting geometric visibility graph.

    Args:
        layout: The point layout (used for bounding box).
        n_obstacles: Number of obstacles to generate.
        obstacle_radius: Half-size of each obstacle.
        n_vertices: Vertices per obstacle (4 = rectangle).
        seed: Random seed.

    Returns:
        List of (m, 2) arrays, each defining an obstacle polygon.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = layout.bounding_box(padding=0.2)

    obstacles = []
    for _ in range(n_obstacles):
        # Random center
        cx = rng.uniform(xmin + obstacle_radius, xmax - obstacle_radius)
        cy = rng.uniform(ymin + obstacle_radius, ymax - obstacle_radius)

        # Random rectangle (slight rotation)
        angle = rng.uniform(0, np.pi / 4)
        r = obstacle_radius * rng.uniform(0.5, 1.0)

        if n_vertices == 4:
            # Axis-aligned rectangle with slight rotation
            corners = np.array([
                [-r, -r],
                [r, -r],
                [r, r],
                [-r, r],
            ], dtype=np.float64)

            # Rotate
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = corners @ rot.T

            # Translate
            corners += np.array([cx, cy])
        else:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False) + angle
            corners = np.column_stack([
                cx + r * np.cos(angles),
                cy + r * np.sin(angles),
            ])

        obstacles.append(corners)

    return obstacles

"""
Shared layout utilities for structured graphs.

Provides helper functions for generating canonical 2D positions
for lattice and structured graph topologies, and for overriding
the PointLayout so visualization works seamlessly.
"""

from __future__ import annotations

import numpy as np

from graphgallery.points import PointLayout, ClusterSpec


def make_structural_layout(
    positions: np.ndarray,
    label: str = "Structured",
    seed: int = 42,
) -> PointLayout:
    """Wrap an (n, 2) position array as a single-cluster PointLayout.

    This allows structured graphs (which define their own geometry)
    to work with the standard visualization pipeline.

    Args:
        positions: (n, 2) array of node coordinates.
        label: Display label for the single pseudo-cluster.
        seed: Seed value for the layout metadata.

    Returns:
        A :class:`PointLayout` with one cluster containing all nodes.
    """
    n = positions.shape[0]
    spec = ClusterSpec(
        n_points=n,
        center=(float(positions[:, 0].mean()), float(positions[:, 1].mean())),
        std=0.0,
        label=label,
    )
    return PointLayout(
        points=positions,
        labels=np.zeros(n, dtype=np.intp),
        cluster_specs=(spec,),
        seed=seed,
    )


def grid_positions(rows: int, cols: int) -> np.ndarray:
    """Generate (rows × cols) grid positions.

    Returns:
        (rows*cols, 2) array with (col, row) coordinates.  Row 0 is
        at the top (y = rows - 1).
    """
    positions = []
    for r in range(rows):
        for c in range(cols):
            positions.append([float(c), float(rows - 1 - r)])
    return np.array(positions, dtype=np.float64)


def ring_positions(n: int, radius: float = 1.0) -> np.ndarray:
    """Generate n positions equally spaced on a circle.

    Args:
        n: Number of nodes.
        radius: Circle radius.

    Returns:
        (n, 2) array of positions.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Start from the top (π/2) and go clockwise
    angles = np.pi / 2 - angles
    return np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
    ])


def best_grid_dims(n: int) -> tuple[int, int]:
    """Find the most square grid dimensions for ~n nodes.

    Returns:
        (rows, cols) such that rows * cols is close to n and
        rows ≤ cols.
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols

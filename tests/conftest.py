"""
Shared pytest fixtures for the Graph Construction Gallery test suite.

Provides canonical layouts, precomputed distance matrices, and helper
functions used across all test modules.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from graphgallery.points import (
    PointLayout,
    ClusterSpec,
    make_two_cluster_layout,
    make_single_cluster_layout,
    make_uniform_layout,
    pairwise_distances,
    k_nearest_indices,
)


# ---------------------------------------------------------------------------
# Import all subpackages to populate the registry
# ---------------------------------------------------------------------------

_SUBPACKAGES = [
    "graphgallery.proximity",
    "graphgallery.triangulation",
    "graphgallery.spanning",
    "graphgallery.random_models",
    "graphgallery.spanners",
    "graphgallery.ann",
    "graphgallery.kernel",
    "graphgallery.visibility",
    "graphgallery.data_driven",
    "graphgallery.misc",
]


@pytest.fixture(scope="session", autouse=True)
def import_all_subpackages():
    """Import every subpackage so all builders are registered."""
    for pkg in _SUBPACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Layout fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def layout() -> PointLayout:
    """The canonical two-cluster layout (30 points, seed=42)."""
    return make_two_cluster_layout(seed=42)


@pytest.fixture(scope="session")
def small_layout() -> PointLayout:
    """A minimal layout for fast tests (8 points)."""
    return make_two_cluster_layout(seed=42)  # Use same but tests can slice


@pytest.fixture(scope="session")
def tiny_layout() -> PointLayout:
    """An extremely small layout for edge-case tests (5 points)."""
    spec = ClusterSpec(n_points=5, center=(0.0, 0.0), std=1.0, label="Tiny")
    from graphgallery.points import make_layout
    return make_layout(clusters=(spec,), seed=99)


@pytest.fixture(scope="session")
def uniform_layout() -> PointLayout:
    """A uniform random layout (30 points)."""
    return make_uniform_layout(n_points=30, seed=42)


@pytest.fixture(scope="session")
def single_cluster_layout() -> PointLayout:
    """A single Gaussian cluster (30 points)."""
    return make_single_cluster_layout(n_points=30, seed=42)


# ---------------------------------------------------------------------------
# Precomputed data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dist_matrix(layout: PointLayout) -> np.ndarray:
    """Precomputed pairwise distance matrix for the canonical layout."""
    return pairwise_distances(layout.points)


@pytest.fixture(scope="session")
def exact_knn_5(dist_matrix: np.ndarray) -> np.ndarray:
    """Exact 5-NN indices for the canonical layout."""
    return k_nearest_indices(dist_matrix, 5, exclude_self=True)


@pytest.fixture(scope="session")
def exact_knn_3(dist_matrix: np.ndarray) -> np.ndarray:
    """Exact 3-NN indices for the canonical layout."""
    return k_nearest_indices(dist_matrix, 3, exclude_self=True)


# ---------------------------------------------------------------------------
# Helper assertion functions
# ---------------------------------------------------------------------------

def assert_valid_graph(
    G: nx.Graph,
    n_expected: int | None = None,
    min_edges: int = 0,
    max_edges: int | None = None,
    connected: bool = False,
    directed: bool = False,
    simple: bool = True,
) -> None:
    """Assert basic structural properties of a graph.

    Args:
        G: The graph to check.
        n_expected: Expected number of nodes (None = skip check).
        min_edges: Minimum expected edges.
        max_edges: Maximum expected edges (None = skip).
        connected: If True, assert the graph is connected.
        directed: If True, assert the graph is directed.
        simple: If True, assert no self-loops.
    """
    if directed:
        assert isinstance(G, nx.DiGraph), f"Expected DiGraph, got {type(G)}"
    else:
        assert isinstance(G, (nx.Graph, nx.DiGraph)), f"Expected Graph, got {type(G)}"

    if n_expected is not None:
        assert G.number_of_nodes() == n_expected, (
            f"Expected {n_expected} nodes, got {G.number_of_nodes()}"
        )

    assert G.number_of_edges() >= min_edges, (
        f"Expected ≥ {min_edges} edges, got {G.number_of_edges()}"
    )

    if max_edges is not None:
        assert G.number_of_edges() <= max_edges, (
            f"Expected ≤ {max_edges} edges, got {G.number_of_edges()}"
        )

    if connected:
        if G.is_directed():
            assert nx.is_weakly_connected(G), "Graph should be weakly connected"
        else:
            assert nx.is_connected(G), "Graph should be connected"

    if simple:
        assert nx.number_of_selfloops(G) == 0, (
            f"Graph has {nx.number_of_selfloops(G)} self-loops"
        )


def assert_weighted_edges(
    G: nx.Graph,
    weight_key: str = "weight",
    min_weight: float = 0.0,
    max_weight: float | None = None,
) -> None:
    """Assert that all edges have valid weights."""
    for u, v, data in G.edges(data=True):
        assert weight_key in data, f"Edge ({u},{v}) missing '{weight_key}'"
        w = data[weight_key]
        assert w >= min_weight, f"Edge ({u},{v}) weight {w} < {min_weight}"
        if max_weight is not None:
            assert w <= max_weight + 1e-9, (
                f"Edge ({u},{v}) weight {w} > {max_weight}"
            )


def assert_symmetric(G: nx.Graph) -> None:
    """Assert that an undirected graph has no directional issues."""
    assert not G.is_directed(), "Expected undirected graph"
    for u, v in G.edges():
        assert G.has_edge(v, u), f"Edge ({u},{v}) but not ({v},{u})"


def assert_degree_bounds(
    G: nx.Graph,
    min_degree: int = 0,
    max_degree: int | None = None,
    exact_degree: int | None = None,
) -> None:
    """Assert degree bounds for all nodes."""
    for node, deg in G.degree():
        if exact_degree is not None:
            assert deg == exact_degree, (
                f"Node {node}: degree {deg} != {exact_degree}"
            )
        assert deg >= min_degree, (
            f"Node {node}: degree {deg} < {min_degree}"
        )
        if max_degree is not None:
            assert deg <= max_degree, (
                f"Node {node}: degree {deg} > {max_degree}"
            )


def assert_subgraph(G_sub: nx.Graph, G_super: nx.Graph) -> None:
    """Assert that G_sub is an edge-subgraph of G_super."""
    for u, v in G_sub.edges():
        assert G_super.has_edge(u, v) or G_super.has_edge(v, u), (
            f"Edge ({u},{v}) in sub but not in super"
        )


def assert_edges_respect_distance(
    G: nx.Graph,
    dist_matrix: np.ndarray,
    max_distance: float,
) -> None:
    """Assert all edges connect points within a distance threshold."""
    for u, v in G.edges():
        d = dist_matrix[u, v]
        assert d <= max_distance + 1e-9, (
            f"Edge ({u},{v}) distance {d:.4f} > {max_distance}"
        )

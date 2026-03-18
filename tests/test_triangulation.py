"""Tests for the triangulation-based graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.triangulation import (
    DelaunayGraph,
    ConstrainedDelaunayGraph,
    all_triangulation_builders,
)
from tests.conftest import assert_valid_graph, assert_weighted_edges, assert_subgraph


class TestDelaunayGraph:
    def test_basic(self, layout):
        G = DelaunayGraph().build(layout)
        n = layout.n_points
        assert_valid_graph(G, n_expected=n, min_edges=n, max_edges=3 * n - 6)

    def test_connected(self, layout):
        G = DelaunayGraph().build(layout)
        assert nx.is_connected(G)

    def test_planar_edge_bound(self, layout):
        G = DelaunayGraph().build(layout)
        assert G.number_of_edges() <= 3 * layout.n_points - 6

    def test_weighted(self, layout):
        G = DelaunayGraph().build(layout)
        assert_weighted_edges(G, min_weight=0.0)


class TestConstrainedDelaunayGraph:
    def test_basic(self, layout):
        G = ConstrainedDelaunayGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=30)

    def test_custom_constraints(self, layout):
        constraints = np.array([[0, 29], [5, 25]], dtype=np.intp)
        G = ConstrainedDelaunayGraph(constraint_edges=constraints).build(layout)
        assert G.has_edge(0, 29) or G.has_edge(29, 0)
        assert G.has_edge(5, 25) or G.has_edge(25, 5)

    def test_constraint_edges_marked(self, layout):
        G = ConstrainedDelaunayGraph().build(layout)
        constrained_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("constrained", False)
        ]
        assert len(constrained_edges) > 0


class TestAllTriangulationBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_triangulation_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() >= layout.n_points
            assert G.number_of_edges() > 0

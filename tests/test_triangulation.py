"""Tests for the triangulation-based graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.triangulation import (
    DelaunayGraph,
    ConstrainedDelaunayGraph,
    ConformingDelaunayGraph,
    WeightedTriangulationGraph,
    VoronoiDualGraph,
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


class TestConformingDelaunayGraph:
    def test_basic(self, layout):
        G = ConformingDelaunayGraph().build(layout)
        assert_valid_graph(G, min_edges=30)
        assert G.number_of_nodes() >= layout.n_points

    def test_steiner_points_flagged(self, layout):
        G = ConformingDelaunayGraph().build(layout)
        n_steiner = sum(
            1 for _, d in G.nodes(data=True) if d.get("steiner", False)
        )
        assert G.graph.get("n_steiner", 0) == n_steiner

    def test_all_points_stored(self, layout):
        G = ConformingDelaunayGraph().build(layout)
        if "all_points" in G.graph:
            assert G.graph["all_points"].shape[0] == G.number_of_nodes()


class TestWeightedTriangulationGraph:
    def test_basic(self, layout):
        G = WeightedTriangulationGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_equal_weights_matches_delaunay(self, layout):
        equal_weights = np.ones(layout.n_points)
        G_wt = WeightedTriangulationGraph(weights=equal_weights).build(layout)
        G_del = DelaunayGraph().build(layout)
        # Should be very similar (possibly identical)
        wt_edges = set(tuple(sorted(e)) for e in G_wt.edges())
        del_edges = set(tuple(sorted(e)) for e in G_del.edges())
        overlap = len(wt_edges & del_edges)
        assert overlap > len(del_edges) * 0.8

    def test_power_distance_stored(self, layout):
        G = WeightedTriangulationGraph().build(layout)
        for u, v, d in G.edges(data=True):
            assert "power_distance" in d


class TestVoronoiDualGraph:
    def test_basic(self, layout):
        G = VoronoiDualGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=30)

    def test_matches_delaunay(self, layout):
        G_vor = VoronoiDualGraph().build(layout)
        G_del = DelaunayGraph().build(layout)
        vor_edges = set(tuple(sorted(e)) for e in G_vor.edges())
        del_edges = set(tuple(sorted(e)) for e in G_del.edges())
        assert vor_edges == del_edges

    def test_voronoi_stored(self, layout):
        G = VoronoiDualGraph(store_voronoi=True).build(layout)
        assert "voronoi" in G.graph

    def test_finite_only(self, layout):
        G_all = VoronoiDualGraph(finite_only=False).build(layout)
        G_fin = VoronoiDualGraph(finite_only=True).build(layout)
        assert G_fin.number_of_edges() <= G_all.number_of_edges()


class TestAllTriangulationBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_triangulation_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() >= layout.n_points
            assert G.number_of_edges() > 0

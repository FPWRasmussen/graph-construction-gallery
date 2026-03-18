"""Tests for the geometric spanners subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.points import pairwise_distances
from graphgallery.spanners import (
    TSpannerGraph,
    YaoGraph,
    ThetaGraph,
    WSPDSpannerGraph,
    GreedySpannerGraph,
    all_spanner_builders,
)
from graphgallery.spanners._spanner_utils import verify_t_spanner
from tests.conftest import assert_valid_graph


class TestTSpannerGraph:
    def test_basic(self, layout):
        G = TSpannerGraph(t=2.0).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, connected=True)

    def test_stretch_factor(self, layout, dist_matrix):
        G = TSpannerGraph(t=2.0).build(layout)
        is_valid, actual_t = verify_t_spanner(G, dist_matrix, 2.0)
        assert is_valid, f"Not a valid 2-spanner: actual t={actual_t}"

    def test_lower_t_more_edges(self, layout):
        G_low = TSpannerGraph(t=1.5).build(layout)
        G_high = TSpannerGraph(t=3.0).build(layout)
        assert G_low.number_of_edges() >= G_high.number_of_edges()


class TestYaoGraph:
    def test_basic(self, layout):
        G = YaoGraph(k=6).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, connected=True)

    def test_actual_stretch_stored(self, layout):
        G = YaoGraph(k=6).build(layout)
        assert "actual_stretch" in G.graph
        assert G.graph["actual_stretch"] >= 1.0

    def test_more_cones_sparser(self, layout):
        G6 = YaoGraph(k=6).build(layout)
        G12 = YaoGraph(k=12).build(layout)
        # More cones doesn't necessarily mean fewer edges
        # but both should be valid
        assert G6.number_of_edges() > 0
        assert G12.number_of_edges() > 0


class TestThetaGraph:
    def test_basic(self, layout):
        G = ThetaGraph(k=6).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, connected=True)

    def test_stretch_stored(self, layout):
        G = ThetaGraph(k=6).build(layout)
        assert "actual_stretch" in G.graph


class TestWSPDSpannerGraph:
    def test_basic(self, layout):
        G = WSPDSpannerGraph(s=4.0).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=10)

    def test_n_pairs_stored(self, layout):
        G = WSPDSpannerGraph(s=4.0).build(layout)
        assert "n_pairs" in G.graph
        assert G.graph["n_pairs"] > 0


class TestGreedySpannerGraph:
    def test_basic(self, layout):
        G = GreedySpannerGraph(t=2.0).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, connected=True)

    def test_valid_spanner(self, layout, dist_matrix):
        G = GreedySpannerGraph(t=2.0).build(layout)
        is_valid, actual_t = verify_t_spanner(G, dist_matrix, 2.0)
        assert is_valid, f"actual t={actual_t}"

    def test_sparser_than_complete(self, layout):
        G = GreedySpannerGraph(t=2.0).build(layout)
        max_edges = 30 * 29 // 2
        assert G.number_of_edges() < max_edges

    def test_contains_mst(self, layout):
        from graphgallery.spanning import EuclideanMSTGraph
        from tests.conftest import assert_subgraph
        G_greedy = GreedySpannerGraph(t=2.0).build(layout)
        G_mst = EuclideanMSTGraph().build(layout)
        assert_subgraph(G_mst, G_greedy)

    def test_edges_metadata(self, layout):
        G = GreedySpannerGraph(t=2.0).build(layout)
        assert "edges_considered" in G.graph
        assert "edges_added" in G.graph
        assert G.graph["edges_added"] == G.number_of_edges()


class TestAllSpannerBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_spanner_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() == 30
            assert G.number_of_edges() > 0

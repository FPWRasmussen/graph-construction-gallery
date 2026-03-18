"""Tests for the visibility graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.visibility import (
    GeometricVisibilityGraph,
    NaturalVisibilityGraph,
    HorizontalVisibilityGraph,
    all_visibility_builders,
)
from graphgallery.visibility._visibility_utils import layout_to_time_series
from tests.conftest import assert_valid_graph, assert_subgraph


class TestGeometricVisibilityGraph:
    def test_no_obstacles_is_complete(self, layout):
        G = GeometricVisibilityGraph(n_auto_obstacles=0).build(layout)
        n = layout.n_points
        assert G.number_of_edges() == n * (n - 1) // 2

    def test_obstacles_reduce_edges(self, layout):
        G_full = GeometricVisibilityGraph(n_auto_obstacles=0).build(layout)
        G_obs = GeometricVisibilityGraph(
            n_auto_obstacles=5, obstacle_radius=0.4, seed=42
        ).build(layout)
        assert G_obs.number_of_edges() < G_full.number_of_edges()

    def test_blocked_pairs_counted(self, layout):
        G = GeometricVisibilityGraph(
            n_auto_obstacles=5, seed=42
        ).build(layout)
        assert G.graph["n_blocked_pairs"] > 0
        total = G.graph["n_visible_pairs"] + G.graph["n_blocked_pairs"]
        assert total == layout.n_points * (layout.n_points - 1) // 2

    def test_obstacles_stored(self, layout):
        G = GeometricVisibilityGraph(
            n_auto_obstacles=3, seed=42
        ).build(layout)
        assert len(G.graph["obstacles"]) == 3


class TestNaturalVisibilityGraph:
    def test_basic(self, layout):
        G = NaturalVisibilityGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, connected=True)

    def test_adjacent_points_connected(self, layout):
        G = NaturalVisibilityGraph().build(layout)
        _, _, sort_order = layout_to_time_series(layout)
        for i in range(len(sort_order) - 1):
            u = int(sort_order[i])
            v = int(sort_order[i + 1])
            assert G.has_edge(u, v), f"Adjacent pair ({u},{v}) missing"

    def test_directed_mode(self, layout):
        G = NaturalVisibilityGraph(directed=True).build(layout)
        assert G.is_directed()


class TestHorizontalVisibilityGraph:
    def test_basic(self, layout):
        G = HorizontalVisibilityGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, connected=True)

    def test_subgraph_of_nvg(self, layout):
        G_hvg = HorizontalVisibilityGraph().build(layout)
        G_nvg = NaturalVisibilityGraph().build(layout)
        assert_subgraph(G_hvg, G_nvg)

    def test_fewer_edges_than_nvg(self, layout):
        G_hvg = HorizontalVisibilityGraph().build(layout)
        G_nvg = NaturalVisibilityGraph().build(layout)
        assert G_hvg.number_of_edges() <= G_nvg.number_of_edges()

    def test_adjacent_connected(self, layout):
        G = HorizontalVisibilityGraph().build(layout)
        _, _, sort_order = layout_to_time_series(layout)
        for i in range(len(sort_order) - 1):
            u = int(sort_order[i])
            v = int(sort_order[i + 1])
            assert G.has_edge(u, v)


class TestAllVisibilityBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_visibility_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() == 30
            assert G.number_of_edges() > 0

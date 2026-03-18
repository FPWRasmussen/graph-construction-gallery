"""Tests for the spanning tree-based graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.spanning import (
    MSTPrimGraph,
    MSTKruskalGraph,
    MSTBoruvkaGraph,
    EuclideanMSTGraph,
    RandomSpanningTreeGraph,
    KMSTOverlayGraph,
    all_spanning_builders,
)
from tests.conftest import assert_valid_graph, assert_weighted_edges


class TestMSTAlgorithmsAgree:
    """All three MST algorithms must produce the same total weight."""

    def test_same_total_weight(self, layout):
        builders = [MSTPrimGraph(), MSTKruskalGraph(), MSTBoruvkaGraph()]
        weights = []
        for b in builders:
            G = b.build(layout)
            w = sum(d["weight"] for _, _, d in G.edges(data=True))
            weights.append(round(w, 8))
        assert weights[0] == weights[1] == weights[2]

    def test_same_edge_count(self, layout):
        builders = [MSTPrimGraph(), MSTKruskalGraph(), MSTBoruvkaGraph()]
        counts = [b.build(layout).number_of_edges() for b in builders]
        assert all(c == 29 for c in counts)  # n - 1 = 29

    def test_emst_agrees(self, layout):
        G_emst = EuclideanMSTGraph().build(layout)
        G_kruskal = MSTKruskalGraph().build(layout)
        w_emst = round(sum(d["weight"] for _, _, d in G_emst.edges(data=True)), 8)
        w_kruskal = round(sum(d["weight"] for _, _, d in G_kruskal.edges(data=True)), 8)
        assert w_emst == w_kruskal


class TestMSTPrimGraph:
    def test_basic(self, layout):
        G = MSTPrimGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, max_edges=29, connected=True)

    def test_is_tree(self, layout):
        G = MSTPrimGraph().build(layout)
        assert nx.is_tree(G)

    def test_different_start_same_weight(self, layout):
        G0 = MSTPrimGraph(start_vertex=0).build(layout)
        G15 = MSTPrimGraph(start_vertex=15).build(layout)
        w0 = sum(d["weight"] for _, _, d in G0.edges(data=True))
        w15 = sum(d["weight"] for _, _, d in G15.edges(data=True))
        assert abs(w0 - w15) < 1e-9


class TestMSTKruskalGraph:
    def test_basic(self, layout):
        G = MSTKruskalGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, max_edges=29, connected=True)
        assert nx.is_tree(G)


class TestMSTBoruvkaGraph:
    def test_basic(self, layout):
        G = MSTBoruvkaGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, max_edges=29, connected=True)
        assert nx.is_tree(G)


class TestEuclideanMSTGraph:
    def test_basic(self, layout):
        G = EuclideanMSTGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, max_edges=29, connected=True)
        assert nx.is_tree(G)

    def test_subgraph_of_delaunay(self, layout):
        from graphgallery.triangulation import DelaunayGraph
        from tests.conftest import assert_subgraph
        G_emst = EuclideanMSTGraph().build(layout)
        G_del = DelaunayGraph().build(layout)
        assert_subgraph(G_emst, G_del)


class TestRandomSpanningTreeGraph:
    def test_basic(self, layout):
        G = RandomSpanningTreeGraph(seed=123).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, max_edges=29, connected=True)
        assert nx.is_tree(G)

    def test_different_seeds_different_trees(self, layout):
        G1 = RandomSpanningTreeGraph(seed=1).build(layout)
        G2 = RandomSpanningTreeGraph(seed=2).build(layout)
        e1 = set(tuple(sorted(e)) for e in G1.edges())
        e2 = set(tuple(sorted(e)) for e in G2.edges())
        assert e1 != e2  # Very likely different

    def test_not_deterministic(self):
        builder = RandomSpanningTreeGraph()
        assert not builder.is_deterministic


class TestKMSTOverlayGraph:
    def test_basic(self, layout):
        G = KMSTOverlayGraph(k=3).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, connected=True)

    def test_more_edges_than_single_mst(self, layout):
        G_mst = MSTKruskalGraph().build(layout)
        G_overlay = KMSTOverlayGraph(k=3).build(layout)
        assert G_overlay.number_of_edges() >= G_mst.number_of_edges()

    def test_k1_similar_to_mst(self, layout):
        G_k1 = KMSTOverlayGraph(k=1, noise_scale=0.0).build(layout)
        assert G_k1.number_of_edges() == 29


class TestAllSpanningBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_spanning_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() == 30
            assert G.number_of_edges() >= 29

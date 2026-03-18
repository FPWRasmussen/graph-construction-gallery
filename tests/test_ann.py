"""Tests for the approximate nearest neighbor graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.points import pairwise_distances, k_nearest_indices
from graphgallery.ann import (
    NSWGraph,
    HNSWGraph,
    VamanaGraph,
    NNDescentGraph,
    RPForestGraph,
    LSHGraph,
    all_ann_builders,
)
from graphgallery.ann._ann_utils import compute_recall
from tests.conftest import assert_valid_graph


class TestNSWGraph:
    def test_basic(self, layout):
        G = NSWGraph(f=5, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_connected(self, layout):
        G = NSWGraph(f=5, seed=42).build(layout)
        assert nx.is_connected(G)


class TestHNSWGraph:
    def test_basic(self, layout):
        G = HNSWGraph(M=5, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_level_attributes(self, layout):
        G = HNSWGraph(M=5, seed=42).build(layout)
        for i in range(30):
            assert "level" in G.nodes[i]
            assert G.nodes[i]["level"] >= 0

    def test_entry_point(self, layout):
        G = HNSWGraph(M=5, seed=42).build(layout)
        assert "entry_point" in G.graph
        ep = G.graph["entry_point"]
        assert 0 <= ep < 30


class TestVamanaGraph:
    def test_basic(self, layout):
        G = VamanaGraph(R=5, alpha=1.2, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_medoid_stored(self, layout):
        G = VamanaGraph(R=5, seed=42).build(layout)
        assert "medoid" in G.graph


class TestNNDescentGraph:
    def test_basic(self, layout):
        G = NNDescentGraph(k=5, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_convergence(self, layout):
        G = NNDescentGraph(k=5, max_iterations=20, seed=42).build(layout)
        assert "iterations" in G.graph
        assert G.graph["iterations"] <= 20

    def test_high_recall(self, layout, dist_matrix, exact_knn_5):
        G = NNDescentGraph(k=5, max_iterations=20, seed=42).build(layout)
        approx_nn = []
        for i in range(layout.n_points):
            nbrs = sorted(G.neighbors(i), key=lambda j: dist_matrix[i, j])[:5]
            approx_nn.append(nbrs)
        recall = compute_recall(approx_nn, exact_knn_5, 5)
        assert recall > 0.7, f"Recall too low: {recall}"


class TestRPForestGraph:
    def test_basic(self, layout):
        G = RPForestGraph(k=5, n_trees=10, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=10)

    def test_n_candidates(self, layout):
        G = RPForestGraph(k=5, n_trees=10, seed=42).build(layout)
        assert "n_candidates" in G.graph


class TestLSHGraph:
    def test_basic(self, layout):
        G = LSHGraph(k=5, n_tables=10, n_bits=8, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=10)

    def test_n_candidates(self, layout):
        G = LSHGraph(k=5, seed=42).build(layout)
        assert "n_candidates" in G.graph


class TestAllANNBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_ann_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() == 30
            assert G.number_of_edges() > 0

    def test_all_nondeterministic(self):
        for builder in all_ann_builders():
            assert not builder.is_deterministic

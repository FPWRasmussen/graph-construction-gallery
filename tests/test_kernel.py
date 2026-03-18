"""Tests for the kernel & similarity-based graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.kernel import (
    GaussianRBFGraph,
    CosineSimilarityGraph,
    ThresholdedSimilarityGraph,
    AdaptiveBandwidthGraph,
    JaccardSimilarityGraph,
    all_kernel_builders,
)
from tests.conftest import assert_valid_graph, assert_weighted_edges


class TestGaussianRBFGraph:
    def test_basic(self, layout):
        G = GaussianRBFGraph(sigma=1.0, threshold=0.1).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=10)

    def test_weights_in_unit_range(self, layout):
        G = GaussianRBFGraph(sigma=1.0, threshold=0.0).build(layout)
        assert_weighted_edges(G, min_weight=0.0, max_weight=1.0)

    def test_threshold_reduces_edges(self, layout):
        G_low = GaussianRBFGraph(sigma=1.0, threshold=0.1).build(layout)
        G_high = GaussianRBFGraph(sigma=1.0, threshold=0.5).build(layout)
        assert G_high.number_of_edges() <= G_low.number_of_edges()

    def test_auto_sigma(self, layout):
        G = GaussianRBFGraph(sigma=None, threshold=0.1).build(layout)
        assert "sigma" in G.graph
        assert G.graph["sigma"] > 0


class TestCosineSimilarityGraph:
    def test_basic(self, layout):
        G = CosineSimilarityGraph(threshold=0.5).build(layout)
        assert_valid_graph(G, n_expected=30)

    def test_weights_bounded(self, layout):
        G = CosineSimilarityGraph(threshold=0.0, weighted=True).build(layout)
        for u, v, d in G.edges(data=True):
            assert -1.0 <= d["weight"] <= 1.0 + 1e-9


class TestThresholdedSimilarityGraph:
    @pytest.mark.parametrize("measure", [
        "gaussian", "cosine", "polynomial", "laplacian", "sigmoid",
        "inverse_distance",
    ])
    def test_all_measures(self, layout, measure):
        G = ThresholdedSimilarityGraph(
            measure=measure, threshold=0.3
        ).build(layout)
        assert_valid_graph(G, n_expected=30)

    def test_unknown_measure(self, layout):
        with pytest.raises(ValueError, match="Unknown"):
            ThresholdedSimilarityGraph(measure="bogus").build(layout)


class TestAdaptiveBandwidthGraph:
    def test_basic(self, layout):
        G = AdaptiveBandwidthGraph(k_bandwidth=7, threshold=0.05).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=10)

    def test_per_point_sigma_varies(self, layout):
        G = AdaptiveBandwidthGraph(k_bandwidth=7).build(layout)
        sigmas = G.graph["per_point_sigma"]
        assert max(sigmas) > min(sigmas)

    def test_weights_in_unit_range(self, layout):
        G = AdaptiveBandwidthGraph(k_bandwidth=7, threshold=0.0).build(layout)
        assert_weighted_edges(G, min_weight=0.0, max_weight=1.0)


class TestJaccardSimilarityGraph:
    @pytest.mark.parametrize("method", [
        "spatial_bin", "neighbor_fingerprint", "multi_threshold",
    ])
    def test_all_methods(self, layout, method):
        G = JaccardSimilarityGraph(
            method=method, threshold=0.1
        ).build(layout)
        assert_valid_graph(G, n_expected=30)

    def test_weights_in_unit_range(self, layout):
        G = JaccardSimilarityGraph(
            method="spatial_bin", threshold=0.0
        ).build(layout)
        if G.number_of_edges() > 0:
            assert_weighted_edges(G, min_weight=0.0, max_weight=1.0)


class TestAllKernelBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_kernel_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() == 30

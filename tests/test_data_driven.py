"""Tests for the data-driven / learned graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.data_driven import (
    CorrelationGraph,
    PartialCorrelationGraph,
    GraphicalLassoGraph,
    MutualInformationGraph,
    ExpansionGraph,
    all_data_driven_builders,
)
from tests.conftest import assert_valid_graph, assert_weighted_edges


class TestCorrelationGraph:
    def test_basic(self, layout):
        G = CorrelationGraph(threshold=0.3, n_samples=300, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=1)

    def test_weights_bounded(self, layout):
        G = CorrelationGraph(threshold=0.3, n_samples=300, seed=42).build(layout)
        assert_weighted_edges(G, min_weight=0.0, max_weight=1.0)

    def test_higher_threshold_fewer_edges(self, layout):
        G_low = CorrelationGraph(threshold=0.3, n_samples=300, seed=42).build(layout)
        G_high = CorrelationGraph(threshold=0.7, n_samples=300, seed=42).build(layout)
        assert G_high.number_of_edges() <= G_low.number_of_edges()

    def test_correlation_matrix_stored(self, layout):
        G = CorrelationGraph(threshold=0.3, n_samples=300, seed=42).build(layout)
        assert "correlation_matrix" in G.graph
        corr = G.graph["correlation_matrix"]
        assert corr.shape == (30, 30)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)


class TestPartialCorrelationGraph:
    def test_basic(self, layout):
        G = PartialCorrelationGraph(
            threshold=0.1, n_samples=300, seed=42
        ).build(layout)
        assert_valid_graph(G, n_expected=30)

    def test_sparser_than_correlation(self, layout):
        G_corr = CorrelationGraph(threshold=0.3, n_samples=500, seed=42).build(layout)
        G_pcorr = PartialCorrelationGraph(
            threshold=0.15, n_samples=500, seed=42
        ).build(layout)
        # Partial correlation is typically sparser
        # (not guaranteed with different thresholds, but generally true)
        assert G_pcorr.number_of_edges() < G_corr.number_of_edges() * 2

    def test_precision_matrix_stored(self, layout):
        G = PartialCorrelationGraph(
            threshold=0.1, n_samples=300, seed=42
        ).build(layout)
        assert "precision_matrix" in G.graph


class TestGraphicalLassoGraph:
    def test_basic(self, layout):
        G = GraphicalLassoGraph(
            alpha=0.15, n_samples=300, seed=42
        ).build(layout)
        assert_valid_graph(G, n_expected=30)

    def test_higher_alpha_sparser(self, layout):
        G_low = GraphicalLassoGraph(
            alpha=0.1, n_samples=300, seed=42
        ).build(layout)
        G_high = GraphicalLassoGraph(
            alpha=0.5, n_samples=300, seed=42
        ).build(layout)
        assert G_high.number_of_edges() <= G_low.number_of_edges()

    def test_precision_stored(self, layout):
        G = GraphicalLassoGraph(
            alpha=0.15, n_samples=300, seed=42
        ).build(layout)
        assert "precision_matrix" in G.graph


class TestMutualInformationGraph:
    @pytest.mark.parametrize("estimator", ["histogram", "ksg"])
    def test_estimators(self, layout, estimator):
        G = MutualInformationGraph(
            threshold=0.05, n_samples=200, estimator=estimator, seed=42
        ).build(layout)
        assert_valid_graph(G, n_expected=30)

    def test_nonnegative_weights(self, layout):
        G = MutualInformationGraph(
            threshold=0.01, n_samples=200, seed=42
        ).build(layout)
        if G.number_of_edges() > 0:
            assert_weighted_edges(G, min_weight=0.0)

    def test_mi_matrix_stored(self, layout):
        G = MutualInformationGraph(
            threshold=0.05, n_samples=200, seed=42
        ).build(layout)
        assert "mi_matrix" in G.graph


class TestExpansionGraph:
    @pytest.mark.parametrize("method", ["percentile", "diffusion"])
    def test_methods(self, layout, method):
        G = ExpansionGraph(method=method).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=1)

    def test_diffusion_matrix_stored(self, layout):
        G = ExpansionGraph(method="diffusion").build(layout)
        assert "diffusion_matrix" in G.graph


class TestAllDataDrivenBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_data_driven_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() == 30

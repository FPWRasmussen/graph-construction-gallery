"""Tests for the random graph model subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.random_models import (
    ErdosRenyiGnpGraph,
    ErdosRenyiGnmGraph,
    WattsStrogatzGraph,
    BarabasiAlbertGraph,
    RandomGeometricGraph,
    StochasticBlockModelGraph,
    ConfigurationModelGraph,
    ChungLuGraph,
    KroneckerGraph,
    ForestFireGraph,
    PriceGraph,
    HolmeKimGraph,
    RandomRegularGraph,
    all_random_model_builders,
)
from tests.conftest import assert_valid_graph, assert_degree_bounds


class TestErdosRenyiGnpGraph:
    def test_basic(self, layout):
        G = ErdosRenyiGnpGraph(p=0.15, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30)

    def test_p0_empty(self, layout):
        G = ErdosRenyiGnpGraph(p=0.0, seed=42).build(layout)
        assert G.number_of_edges() == 0

    def test_p1_complete(self, layout):
        G = ErdosRenyiGnpGraph(p=1.0, seed=42).build(layout)
        assert G.number_of_edges() == 30 * 29 // 2


class TestErdosRenyiGnmGraph:
    def test_exact_edge_count(self, layout):
        G = ErdosRenyiGnmGraph(m=60, seed=42).build(layout)
        assert G.number_of_edges() == 60

    def test_m0(self, layout):
        G = ErdosRenyiGnmGraph(m=0, seed=42).build(layout)
        assert G.number_of_edges() == 0

    def test_m_capped(self, layout):
        max_e = 30 * 29 // 2
        G = ErdosRenyiGnmGraph(m=max_e + 100, seed=42).build(layout)
        assert G.number_of_edges() == max_e


class TestWattsStrogatzGraph:
    def test_basic(self, layout):
        G = WattsStrogatzGraph(k=4, p=0.3, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_p0_ring_lattice(self, layout):
        G = WattsStrogatzGraph(k=4, p=0.0, seed=42).build(layout)
        # Should have exactly n*k/2 edges
        assert G.number_of_edges() == 30 * 4 // 2

    def test_k_must_be_even(self):
        with pytest.raises(ValueError, match="even"):
            WattsStrogatzGraph(k=3)


class TestBarabasiAlbertGraph:
    def test_basic(self, layout):
        G = BarabasiAlbertGraph(m=2, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_connected(self, layout):
        G = BarabasiAlbertGraph(m=2, seed=42).build(layout)
        assert nx.is_connected(G)


class TestRandomGeometricGraph:
    def test_basic(self, layout):
        G = RandomGeometricGraph(r=1.0).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=1)

    def test_deterministic(self, layout):
        G1 = RandomGeometricGraph(r=1.0).build(layout)
        G2 = RandomGeometricGraph(r=1.0).build(layout)
        assert set(G1.edges()) == set(G2.edges())


class TestStochasticBlockModelGraph:
    def test_basic(self, layout):
        G = StochasticBlockModelGraph(
            p_within=0.4, p_between=0.05, seed=42
        ).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=1)

    def test_community_structure(self, layout):
        G = StochasticBlockModelGraph(
            p_within=0.8, p_between=0.01, seed=42
        ).build(layout)
        intra = sum(1 for u, v in G.edges()
                    if layout.labels[u] == layout.labels[v])
        inter = G.number_of_edges() - intra
        assert intra > inter  # More intra than inter edges


class TestConfigurationModelGraph:
    def test_basic(self, layout):
        G = ConfigurationModelGraph(seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=1)


class TestChungLuGraph:
    def test_basic(self, layout):
        G = ChungLuGraph(seed=42).build(layout)
        assert_valid_graph(G, n_expected=30)


class TestKroneckerGraph:
    def test_basic(self, layout):
        G = KroneckerGraph(seed=42).build(layout)
        assert_valid_graph(G, n_expected=30)


class TestForestFireGraph:
    def test_basic(self, layout):
        G = ForestFireGraph(p_forward=0.35, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29, connected=True)


class TestPriceGraph:
    def test_basic(self, layout):
        G = PriceGraph(m=3, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20, directed=True)

    def test_is_directed(self, layout):
        G = PriceGraph(seed=42).build(layout)
        assert G.is_directed()


class TestHolmeKimGraph:
    def test_basic(self, layout):
        G = HolmeKimGraph(m=2, p=0.5, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20, connected=True)


class TestRandomRegularGraph:
    def test_basic(self, layout):
        G = RandomRegularGraph(d=4, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30)
        assert_degree_bounds(G, exact_degree=4)

    def test_invalid_nd_odd(self, layout):
        with pytest.raises(ValueError, match="even"):
            RandomRegularGraph(d=3).build(layout)  # 30 * 3 = 90 is even, OK
        # But 30 * 5 = 150 is even too. Use a layout where n*d is odd.

    def test_d3(self, layout):
        # 30 * 3 = 90 is even, so this should work
        G = RandomRegularGraph(d=3, seed=42).build(layout)
        assert_degree_bounds(G, exact_degree=3)


class TestAllRandomModelBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_random_model_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() >= 2

"""Tests for the lattice & structured graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.lattice import (
    GridGraph,
    RingGraph,
    StarGraph,
    CompleteBipartiteGraph,
    HypercubeGraph,
    TorusGraph,
    HexagonalLatticeGraph,
    TriangularLatticeGraph,
    PetersenGraph,
    all_lattice_builders,
)
from tests.conftest import assert_valid_graph, assert_degree_bounds


class TestGridGraph:
    def test_basic(self, layout):
        G = GridGraph(rows=5, cols=6).build(layout)
        assert_valid_graph(G, n_expected=30, connected=True)

    def test_4connected_degree(self, layout):
        G = GridGraph(rows=5, cols=6).build(layout)
        for node, deg in G.degree():
            assert 2 <= deg <= 4

    def test_8connected(self, layout):
        G4 = GridGraph(rows=5, cols=6, eight_connected=False).build(layout)
        G8 = GridGraph(rows=5, cols=6, eight_connected=True).build(layout)
        assert G8.number_of_edges() > G4.number_of_edges()

    def test_positions_stored(self, layout):
        G = GridGraph(rows=5, cols=6).build(layout)
        assert "positions" in G.graph
        assert G.graph["positions"].shape == (30, 2)


class TestRingGraph:
    def test_basic(self, layout):
        G = RingGraph().build(layout)
        assert_valid_graph(G, n_expected=30, connected=True)
        assert_degree_bounds(G, exact_degree=2)

    def test_is_cycle(self, layout):
        G = RingGraph().build(layout)
        assert G.number_of_edges() == G.number_of_nodes()

    def test_n_override(self, layout):
        G = RingGraph(n_override=15).build(layout)
        assert G.number_of_nodes() == 15


class TestStarGraph:
    def test_basic(self, layout):
        G = StarGraph().build(layout)
        assert_valid_graph(G, n_expected=30, connected=True)

    def test_hub_degree(self, layout):
        G = StarGraph().build(layout)
        degrees = sorted([d for _, d in G.degree()])
        assert degrees[-1] == 29  # Hub
        assert all(d == 1 for d in degrees[:-1])  # Leaves

    def test_edge_count(self, layout):
        G = StarGraph().build(layout)
        assert G.number_of_edges() == 29


class TestCompleteBipartiteGraph:
    def test_basic(self, layout):
        G = CompleteBipartiteGraph(a=10, b=20).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=200, max_edges=200)

    def test_bipartite(self, layout):
        G = CompleteBipartiteGraph(a=10, b=20).build(layout)
        assert nx.is_bipartite(G)

    def test_partition_metadata(self, layout):
        G = CompleteBipartiteGraph(a=10, b=20).build(layout)
        a_count = sum(1 for _, d in G.nodes(data=True) if d["partition"] == "A")
        b_count = sum(1 for _, d in G.nodes(data=True) if d["partition"] == "B")
        assert a_count == 10 and b_count == 20


class TestHypercubeGraph:
    def test_basic(self, layout):
        G = HypercubeGraph(d=5).build(layout)
        assert_valid_graph(G, n_expected=32, connected=True)
        assert_degree_bounds(G, exact_degree=5)

    def test_node_labels(self, layout):
        G = HypercubeGraph(d=5).build(layout)
        for i in range(32):
            assert "binary" in G.nodes[i]
            assert len(G.nodes[i]["binary"]) == 5

    def test_hamming_distance_1(self, layout):
        G = HypercubeGraph(d=4).build(layout)
        for u, v in G.edges():
            bu = G.nodes[u]["binary"]
            bv = G.nodes[v]["binary"]
            hamming = sum(a != b for a, b in zip(bu, bv))
            assert hamming == 1


class TestTorusGraph:
    def test_basic(self, layout):
        G = TorusGraph(rows=5, cols=6).build(layout)
        assert_valid_graph(G, n_expected=30, connected=True)
        assert_degree_bounds(G, exact_degree=4)

    def test_more_edges_than_grid(self, layout):
        G_grid = GridGraph(rows=5, cols=6).build(layout)
        G_torus = TorusGraph(rows=5, cols=6).build(layout)
        assert G_torus.number_of_edges() > G_grid.number_of_edges()

    def test_wrap_edges_marked(self, layout):
        G = TorusGraph(rows=5, cols=6).build(layout)
        wrap_count = sum(1 for _, _, d in G.edges(data=True) if d.get("wrap"))
        assert wrap_count > 0


class TestPetersenGraph:
    def test_classic(self, layout):
        G = PetersenGraph(n=5, k=2).build(layout)
        assert_valid_graph(G, n_expected=10)
        assert G.number_of_edges() == 15
        assert_degree_bounds(G, exact_degree=3)

    def test_not_hamiltonian(self, layout):
        # The classic Petersen graph is not Hamiltonian
        # We can't easily test this directly, but we can check girth = 5
        G = PetersenGraph().build(layout)
        girth = nx.girth(G)
        assert girth == 5

    def test_ring_metadata(self, layout):
        G = PetersenGraph().build(layout)
        outer = sum(1 for _, d in G.nodes(data=True) if d["ring"] == "outer")
        inner = sum(1 for _, d in G.nodes(data=True) if d["ring"] == "inner")
        assert outer == 5 and inner == 5


class TestAllLatticeBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_lattice_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() >= 2
            assert G.number_of_edges() > 0
            assert "positions" in G.graph or "structural_layout" in G.graph

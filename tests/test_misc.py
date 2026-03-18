"""Tests for the miscellaneous graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.points import pairwise_distances, k_nearest_indices
from graphgallery.misc import (
    PowerDiagramGraph,
    KDTreeNeighborGraph,
    BallTreeNeighborGraph,
    DiskGraph,
    IntersectionGraph,
    DeBruijnGraph,
    CayleyGraph,
    all_misc_builders,
)
from tests.conftest import (
    assert_valid_graph,
    assert_weighted_edges,
    assert_degree_bounds,
)


class TestPowerDiagramGraph:
    def test_basic(self, layout):
        G = PowerDiagramGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_power_distance_stored(self, layout):
        G = PowerDiagramGraph().build(layout)
        for u, v, d in G.edges(data=True):
            assert "power_distance" in d


class TestKDTreeNeighborGraph:
    def test_basic(self, layout):
        G = KDTreeNeighborGraph(k=5).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=30)

    def test_matches_exact_knn(self, layout, dist_matrix, exact_knn_5):
        G = KDTreeNeighborGraph(k=5).build(layout)
        for i in range(30):
            for j in exact_knn_5[i]:
                assert G.has_edge(i, int(j)) or G.has_edge(int(j), i), \
                    f"Missing exact k-NN edge ({i}, {j})"


class TestBallTreeNeighborGraph:
    def test_knn_matches_kdtree(self, layout):
        G_kd = KDTreeNeighborGraph(k=5).build(layout)
        G_bt = BallTreeNeighborGraph(k=5, mode="knn").build(layout)
        kd_edges = set(tuple(sorted(e)) for e in G_kd.edges())
        bt_edges = set(tuple(sorted(e)) for e in G_bt.edges())
        assert kd_edges == bt_edges

    def test_radius_mode(self, layout, dist_matrix):
        r = 1.0
        G = BallTreeNeighborGraph(radius=r, mode="radius").build(layout)
        for u, v in G.edges():
            assert dist_matrix[u, v] <= r + 1e-9


class TestDiskGraph:
    def test_basic(self, layout):
        G = DiskGraph(r=0.5).build(layout)
        assert_valid_graph(G, n_expected=30)

    def test_large_radius_complete(self, layout):
        G = DiskGraph(r=50.0).build(layout)
        assert G.number_of_edges() == 30 * 29 // 2

    def test_tiny_radius_sparse(self, layout):
        G = DiskGraph(r=0.001).build(layout)
        assert G.number_of_edges() <= 5

    def test_distance_bound(self, layout, dist_matrix):
        r = 0.5
        G = DiskGraph(r=r).build(layout)
        for u, v in G.edges():
            assert dist_matrix[u, v] <= 2 * r + 1e-9

    def test_adaptive_mode(self, layout):
        G = DiskGraph(mode="adaptive", adaptive_k=3).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=1)


class TestIntersectionGraph:
    @pytest.mark.parametrize("shape", ["circle", "rectangle"])
    def test_shapes(self, layout, shape):
        G = IntersectionGraph(shape=shape, radius_mean=0.7, seed=42).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=1)

    def test_circle_radius_metadata(self, layout):
        G = IntersectionGraph(shape="circle", seed=42).build(layout)
        for i in range(30):
            assert "radius" in G.nodes[i]

    def test_rectangle_extent_metadata(self, layout):
        G = IntersectionGraph(shape="rectangle", seed=42).build(layout)
        for i in range(30):
            assert "extent_x" in G.nodes[i]
            assert "extent_y" in G.nodes[i]


class TestDeBruijnGraph:
    def test_basic_b25(self, layout):
        G = DeBruijnGraph(k=2, n=5).build(layout)
        assert G.number_of_nodes() == 32

    def test_b33(self, layout):
        G = DeBruijnGraph(k=3, n=3).build(layout)
        assert G.number_of_nodes() == 27

    def test_node_labels(self, layout):
        G = DeBruijnGraph(k=2, n=3).build(layout)
        for i in range(G.number_of_nodes()):
            label = G.nodes[i]["label"]
            assert len(label) == 3
            assert all(c in "01" for c in label)

    def test_positions_stored(self, layout):
        G = DeBruijnGraph(k=2, n=5).build(layout)
        assert "positions" in G.graph


class TestCayleyGraph:
    def test_cyclic_generators_1(self, layout):
        G = CayleyGraph(group="cyclic", n=15, generators=[1]).build(layout)
        assert G.number_of_nodes() == 15
        assert_degree_bounds(G, exact_degree=2)  # Cycle graph

    def test_cyclic_generators_1_2(self, layout):
        G = CayleyGraph(group="cyclic", n=15).build(layout)
        assert G.number_of_nodes() == 15
        # With generators {1, 2, -1, -2}, degree = 4
        assert_degree_bounds(G, exact_degree=4)

    def test_dihedral(self, layout):
        G = CayleyGraph(group="dihedral", n=8).build(layout)
        assert G.number_of_nodes() == 16  # |D_8| = 2*8

    def test_symmetric_s3(self, layout):
        G = CayleyGraph(group="symmetric", n=3).build(layout)
        assert G.number_of_nodes() == 6  # 3! = 6

    def test_symmetric_s4(self, layout):
        G = CayleyGraph(group="symmetric", n=4).build(layout)
        assert G.number_of_nodes() == 24  # 4! = 24

    def test_positions_stored(self, layout):
        G = CayleyGraph(group="cyclic", n=15).build(layout)
        assert "positions" in G.graph


class TestAllMiscBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_misc_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() >= 2
            assert G.number_of_edges() > 0

"""Tests for the proximity & distance-based graph subpackage."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.points import PointLayout, pairwise_distances, k_nearest_indices
from graphgallery.proximity import (
    CompleteGraph,
    SymmetricKNNGraph,
    MutualKNNGraph,
    EpsilonNeighborhoodGraph,
    GabrielGraph,
    RelativeNeighborhoodGraph,
    BetaSkeletonGraph,
    UrquhartGraph,
    SphereOfInfluenceGraph,
    all_proximity_builders,
)
from tests.conftest import (
    assert_valid_graph,
    assert_weighted_edges,
    assert_symmetric,
    assert_subgraph,
    assert_edges_respect_distance,
)


class TestCompleteGraph:
    def test_basic(self, layout):
        G = CompleteGraph().build(layout)
        n = layout.n_points
        assert_valid_graph(G, n_expected=n, min_edges=n * (n - 1) // 2)

    def test_edge_count(self, layout):
        G = CompleteGraph().build(layout)
        assert G.number_of_edges() == 30 * 29 // 2

    def test_weighted(self, layout, dist_matrix):
        G = CompleteGraph(weighted=True).build(layout)
        assert_weighted_edges(G, min_weight=0.0)
        for u, v, d in G.edges(data=True):
            assert abs(d["weight"] - dist_matrix[u, v]) < 1e-9

    def test_unweighted(self, layout):
        G = CompleteGraph(weighted=False).build(layout)
        for u, v, d in G.edges(data=True):
            assert "weight" not in d


class TestSymmetricKNNGraph:
    def test_basic(self, layout):
        G = SymmetricKNNGraph(k=5).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=30)
        assert_symmetric(G)

    def test_more_edges_than_mutual(self, layout):
        G_sym = SymmetricKNNGraph(k=5).build(layout)
        G_mut = MutualKNNGraph(k=5).build(layout)
        assert G_sym.number_of_edges() >= G_mut.number_of_edges()

    def test_superset_of_mutual(self, layout):
        G_sym = SymmetricKNNGraph(k=5).build(layout)
        G_mut = MutualKNNGraph(k=5).build(layout)
        assert_subgraph(G_mut, G_sym)


class TestMutualKNNGraph:
    def test_basic(self, layout):
        G = MutualKNNGraph(k=5).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=1)
        assert_symmetric(G)

    def test_mutual_condition(self, layout, dist_matrix, exact_knn_5):
        G = MutualKNNGraph(k=5).build(layout)
        knn_sets = [set(exact_knn_5[i].tolist()) for i in range(30)]
        for u, v in G.edges():
            assert v in knn_sets[u] and u in knn_sets[v]


class TestEpsilonNeighborhoodGraph:
    def test_basic(self, layout):
        G = EpsilonNeighborhoodGraph(epsilon=1.2).build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=1)

    def test_distance_bound(self, layout, dist_matrix):
        eps = 1.2
        G = EpsilonNeighborhoodGraph(epsilon=eps).build(layout)
        assert_edges_respect_distance(G, dist_matrix, eps)

    def test_completeness(self, layout, dist_matrix):
        eps = 1.2
        G = EpsilonNeighborhoodGraph(epsilon=eps).build(layout)
        for i in range(30):
            for j in range(i + 1, 30):
                if dist_matrix[i, j] <= eps:
                    assert G.has_edge(i, j)

    def test_large_epsilon_is_complete(self, layout):
        G = EpsilonNeighborhoodGraph(epsilon=100.0).build(layout)
        assert G.number_of_edges() == 30 * 29 // 2

    def test_tiny_epsilon_is_sparse(self, layout):
        G = EpsilonNeighborhoodGraph(epsilon=0.01).build(layout)
        assert G.number_of_edges() <= 5


class TestGabrielGraph:
    def test_basic(self, layout):
        G = GabrielGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=29)

    def test_subgraph_of_delaunay(self, layout):
        from graphgallery.triangulation import DelaunayGraph
        G_gab = GabrielGraph().build(layout)
        G_del = DelaunayGraph().build(layout)
        assert_subgraph(G_gab, G_del)

    def test_supergraph_of_rng(self, layout):
        G_gab = GabrielGraph().build(layout)
        G_rng = RelativeNeighborhoodGraph().build(layout)
        assert_subgraph(G_rng, G_gab)


class TestRelativeNeighborhoodGraph:
    def test_basic(self, layout):
        G = RelativeNeighborhoodGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_contains_mst(self, layout):
        from graphgallery.spanning import EuclideanMSTGraph
        G_rng = RelativeNeighborhoodGraph().build(layout)
        G_mst = EuclideanMSTGraph().build(layout)
        assert_subgraph(G_mst, G_rng)


class TestBetaSkeletonGraph:
    def test_beta_1_equals_gabriel(self, layout):
        G_beta1 = BetaSkeletonGraph(beta=1.0).build(layout)
        G_gab = GabrielGraph().build(layout)
        assert set(G_beta1.edges()) == set(G_gab.edges())

    def test_higher_beta_is_sparser(self, layout):
        G_low = BetaSkeletonGraph(beta=1.0).build(layout)
        G_high = BetaSkeletonGraph(beta=2.0).build(layout)
        assert G_high.number_of_edges() <= G_low.number_of_edges()

    def test_invalid_beta(self):
        with pytest.raises(ValueError):
            BetaSkeletonGraph(beta=0.0)


class TestUrquhartGraph:
    def test_basic(self, layout):
        G = UrquhartGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)

    def test_subgraph_of_delaunay(self, layout):
        from graphgallery.triangulation import DelaunayGraph
        G_urq = UrquhartGraph().build(layout)
        G_del = DelaunayGraph().build(layout)
        assert_subgraph(G_urq, G_del)


class TestSphereOfInfluenceGraph:
    def test_basic(self, layout):
        G = SphereOfInfluenceGraph().build(layout)
        assert_valid_graph(G, n_expected=30, min_edges=20)


class TestAllProximityBuilders:
    def test_all_build_successfully(self, layout):
        for builder in all_proximity_builders():
            G = builder.build(layout)
            assert G.number_of_nodes() > 0
            assert G.number_of_edges() >= 0

    def test_all_have_metadata(self):
        for builder in all_proximity_builders():
            assert builder.name
            assert builder.slug
            assert builder.category == "proximity"

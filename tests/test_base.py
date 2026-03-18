"""Tests for the base module: registry, GraphBuilder, BuildResult, ParamInfo."""

import numpy as np
import networkx as nx
import pytest

from graphgallery.base import (
    GraphBuilder,
    BuildResult,
    ParamInfo,
    registry,
    all_builders,
    get_builder,
    list_categories,
    list_algorithms,
    CATEGORY_DIRECTORY_MAP,
    CATEGORY_DISPLAY_NAMES,
)
from graphgallery.points import make_two_cluster_layout


class TestRegistry:
    def test_registry_not_empty(self):
        assert len(registry()) > 0

    def test_all_builders_returns_list(self):
        builders = all_builders()
        assert isinstance(builders, list)
        assert len(builders) > 50  # We have 74 algorithms

    def test_get_builder_valid(self):
        cls = get_builder("proximity", "knn")
        assert issubclass(cls, GraphBuilder)

    def test_get_builder_invalid(self):
        with pytest.raises(KeyError):
            get_builder("nonexistent", "fake")

    def test_list_categories(self):
        cats = list_categories()
        assert "proximity" in cats
        assert "spanning" in cats
        assert len(cats) >= 11

    def test_list_algorithms(self):
        algos = list_algorithms("proximity")
        assert "knn" in algos
        assert "complete" in algos

    def test_list_algorithms_invalid_category(self):
        with pytest.raises(KeyError):
            list_algorithms("nonexistent")


class TestGraphBuilder:
    def test_all_builders_have_required_attrs(self):
        for cls in all_builders():
            b = cls()
            assert b.name, f"{cls} has empty name"
            assert b.slug, f"{cls} has empty slug"
            assert b.category, f"{cls} has empty category"

    def test_build_and_record(self):
        layout = make_two_cluster_layout()
        cls = get_builder("proximity", "complete")
        builder = cls()
        result = builder.build_and_record(layout)
        assert isinstance(result, BuildResult)
        assert result.n_nodes == 30
        assert result.n_edges > 0
        assert result.build_time_ms >= 0

    def test_params_property(self):
        cls = get_builder("proximity", "knn")
        builder = cls(k=7)
        assert builder.params["k"] == 7

    def test_repr(self):
        cls = get_builder("proximity", "knn")
        builder = cls(k=7)
        assert "k=7" in repr(builder)

    def test_str(self):
        cls = get_builder("proximity", "knn")
        builder = cls(k=7)
        s = str(builder)
        assert "proximity" in s
        assert "knn" in s

    def test_info(self):
        cls = get_builder("proximity", "knn")
        builder = cls(k=5)
        info = builder.info()
        assert "k-Nearest" in info or "knn" in info
        assert "proximity" in info

    def test_image_path(self):
        cls = get_builder("proximity", "knn")
        builder = cls()
        path = builder.image_path
        assert "01_proximity" in path
        assert "knn.png" in path

    def test_validate_layout(self):
        from graphgallery.points import PointLayout, ClusterSpec, make_layout
        cls = get_builder("proximity", "complete")
        builder = cls()
        # Valid
        layout = make_two_cluster_layout()
        builder.validate_layout(layout)  # Should not raise
        # Invalid: single point
        tiny_spec = ClusterSpec(n_points=1, center=(0, 0), std=1.0)
        tiny = make_layout(clusters=(tiny_spec,))
        with pytest.raises(ValueError):
            builder.validate_layout(tiny)


class TestBuildResult:
    def test_summary(self):
        G = nx.Graph()
        G.add_nodes_from(range(5))
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])
        result = BuildResult(
            graph=G,
            builder_name="Test",
            builder_slug="test",
            category="test_cat",
            params={"k": 5},
            build_time_ms=1.23,
        )
        s = result.summary()
        assert "5 nodes" in s
        assert "3 edges" in s
        assert "1.2" in s

    def test_density(self):
        G = nx.complete_graph(5)
        result = BuildResult(
            graph=G, builder_name="T", builder_slug="t",
            category="c", params={}, build_time_ms=0,
        )
        assert abs(result.density - 1.0) < 1e-9


class TestParamInfo:
    def test_markdown_row(self):
        p = ParamInfo("k", "Neighbors", "int", 5, "k ≥ 1")
        row = p.as_markdown_row()
        assert "`k`" in row
        assert "Neighbors" in row
        assert "`5`" in row


class TestCategoryConstants:
    def test_directory_map_complete(self):
        for cat in list_categories():
            assert cat in CATEGORY_DIRECTORY_MAP, f"Missing {cat} in DIRECTORY_MAP"

    def test_display_names_complete(self):
        for cat in list_categories():
            assert cat in CATEGORY_DISPLAY_NAMES, f"Missing {cat} in DISPLAY_NAMES"

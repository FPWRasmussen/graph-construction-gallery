"""Visibility graph construction algorithms."""

from graphgallery.visibility.geometric_visibility import GeometricVisibilityGraph
from graphgallery.visibility.natural_visibility import NaturalVisibilityGraph
from graphgallery.visibility.horizontal_visibility import (
    HorizontalVisibilityGraph,
)

__all__ = [
    "GeometricVisibilityGraph",
    "NaturalVisibilityGraph",
    "HorizontalVisibilityGraph",
    "all_visibility_builders",
]


def all_visibility_builders():
    """Return default instances of every visibility builder."""

    return [
        GeometricVisibilityGraph(),
        NaturalVisibilityGraph(),
        HorizontalVisibilityGraph(),
    ]


"""
Triangulation-Based Graph Construction Algorithms.

This subpackage contains 5 graph builders derived from triangulations
and Voronoi structures:

1.  Delaunay Triangulation
2.  Constrained Delaunay Triangulation
3.  Conforming Delaunay Triangulation
4.  Weighted (Regular) Triangulation
5.  Voronoi Dual Graph

Dependencies:
    - ``scipy`` for standard Delaunay and Voronoi
    - ``triangle`` (optional) for constrained/conforming variants
      Install with: ``pip install triangle``
"""

from graphgallery.triangulation.delaunay import DelaunayGraph
from graphgallery.triangulation.constrained_delaunay import ConstrainedDelaunayGraph
from graphgallery.triangulation.conforming_delaunay import ConformingDelaunayGraph
from graphgallery.triangulation.weighted_triangulation import WeightedTriangulationGraph
from graphgallery.triangulation.voronoi_dual import VoronoiDualGraph

__all__ = [
    "DelaunayGraph",
    "ConstrainedDelaunayGraph",
    "ConformingDelaunayGraph",
    "WeightedTriangulationGraph",
    "VoronoiDualGraph",
]


def all_triangulation_builders():
    """Return instances of every triangulation builder with default params."""
    return [
        DelaunayGraph(),
        ConstrainedDelaunayGraph(),
        ConformingDelaunayGraph(),
        WeightedTriangulationGraph(),
        VoronoiDualGraph(),
    ]

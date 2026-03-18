"""
Triangulation-Based Graph Construction Algorithms.

This subpackage contains graph builders derived from Delaunay
triangulations and constraint handling:

1.  Delaunay Triangulation
2.  Constrained Delaunay Triangulation

Dependencies:
    - ``scipy`` for standard Delaunay
    - ``triangle`` (optional) for constrained variants
      Install with: ``pip install triangle``
"""

from graphgallery.triangulation.constrained_delaunay import ConstrainedDelaunayGraph
from graphgallery.triangulation.delaunay import DelaunayGraph

__all__ = [
    "ConstrainedDelaunayGraph",
    "DelaunayGraph",
]


def all_triangulation_builders():
    """Return instances of every triangulation builder with defaults."""
    return [
        ConstrainedDelaunayGraph(),
        DelaunayGraph(),
    ]

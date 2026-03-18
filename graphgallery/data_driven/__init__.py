"""
Data-Driven / Learned Graph Construction Algorithms.

This subpackage contains 5 graph builders that infer graph structure
from statistical relationships between variables:

1.  Correlation Graph
2.  Partial Correlation Graph
3.  Graphical LASSO (Sparse Inverse Covariance)
4.  Mutual Information Graph
5.  Expansion Graph

Unlike proximity or kernel graphs that use point *coordinates*
directly, these methods treat each point as a *random variable*
and infer edges from statistical dependencies observed across
multiple samples.

To create meaningful statistical structure from our 2D point layout,
each builder generates synthetic multivariate observations where
nearby points have correlated values (via a spatial covariance
function).  The resulting statistical graph can then be compared
against the spatial proximity structure.

Dependencies:
    - ``numpy`` for core computation
    - ``scipy`` for matrix operations
    - ``sklearn`` (optional) for Graphical LASSO
"""

from graphgallery.data_driven.correlation import CorrelationGraph
from graphgallery.data_driven.partial_correlation import PartialCorrelationGraph
from graphgallery.data_driven.glasso import GraphicalLassoGraph
from graphgallery.data_driven.mutual_information import MutualInformationGraph
from graphgallery.data_driven.expansion import ExpansionGraph

__all__ = [
    "CorrelationGraph",
    "PartialCorrelationGraph",
    "GraphicalLassoGraph",
    "MutualInformationGraph",
    "ExpansionGraph",
]


def all_data_driven_builders():
    """Return instances of every data-driven builder with default params."""
    return [
        CorrelationGraph(),
        PartialCorrelationGraph(),
        GraphicalLassoGraph(),
        MutualInformationGraph(),
        ExpansionGraph(),
    ]

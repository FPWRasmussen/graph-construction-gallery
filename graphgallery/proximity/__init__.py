"""
Proximity & Distance-Based Graph Construction Algorithms.

This subpackage contains 10 graph builders that connect points
based on spatial distance or neighbor relationships:

1.  Complete Graph
2.  k-Nearest Neighbors (directed)
3.  Symmetric k-NN (undirected, union)
4.  Mutual k-NN (undirected, intersection)
5.  ε-Neighborhood
6.  Gabriel Graph
7.  Relative Neighborhood Graph
8.  β-Skeleton
9.  Urquhart Graph
10. Sphere of Influence Graph
"""

from graphgallery.proximity.complete import CompleteGraph
from graphgallery.proximity.knn import KNNGraph
from graphgallery.proximity.symmetric_knn import SymmetricKNNGraph
from graphgallery.proximity.mutual_knn import MutualKNNGraph
from graphgallery.proximity.epsilon import EpsilonNeighborhoodGraph
from graphgallery.proximity.gabriel import GabrielGraph
from graphgallery.proximity.rng import RelativeNeighborhoodGraph
from graphgallery.proximity.beta_skeleton import BetaSkeletonGraph
from graphgallery.proximity.urquhart import UrquhartGraph
from graphgallery.proximity.influence import SphereOfInfluenceGraph

__all__ = [
    "CompleteGraph",
    "KNNGraph",
    "SymmetricKNNGraph",
    "MutualKNNGraph",
    "EpsilonNeighborhoodGraph",
    "GabrielGraph",
    "RelativeNeighborhoodGraph",
    "BetaSkeletonGraph",
    "UrquhartGraph",
    "SphereOfInfluenceGraph",
]


def all_proximity_builders():
    """Return instances of every proximity builder with default parameters."""
    return [
        CompleteGraph(),
        KNNGraph(),
        SymmetricKNNGraph(),
        MutualKNNGraph(),
        EpsilonNeighborhoodGraph(),
        GabrielGraph(),
        RelativeNeighborhoodGraph(),
        BetaSkeletonGraph(),
        UrquhartGraph(),
        SphereOfInfluenceGraph(),
    ]

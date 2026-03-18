"""
Proximity & Distance-Based Graph Construction Algorithms.

This subpackage contains 9 graph builders that connect points
based on spatial distance or neighbor relationships:

1.  Complete Graph
2.  Symmetric k-NN (undirected, union)
3.  Mutual k-NN (undirected, intersection)
4.  ε-Neighborhood
5.  Gabriel Graph
6.  Relative Neighborhood Graph
7.  β-Skeleton
8.  Urquhart Graph
9.  Sphere of Influence Graph
"""

from graphgallery.proximity.complete import CompleteGraph
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
        SymmetricKNNGraph(),
        MutualKNNGraph(),
        EpsilonNeighborhoodGraph(),
        GabrielGraph(),
        RelativeNeighborhoodGraph(),
        BetaSkeletonGraph(),
        UrquhartGraph(),
        SphereOfInfluenceGraph(),
    ]

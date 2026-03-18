"""
Geometric Spanner Graph Construction Algorithms.

This subpackage contains 5 graph builders that produce sparse
subgraphs approximately preserving pairwise Euclidean distances:

1.  t-Spanner (naïve edge filtering)
2.  Yao Graph
3.  Theta (Θ) Graph
4.  WSPD Spanner (Well-Separated Pair Decomposition)
5.  Greedy Spanner

A geometric t-spanner for a point set S is a graph G such that for
any two points p, q ∈ S, the shortest-path distance in G is at most
t times the Euclidean distance |pq|:

    δ_G(p, q) ≤ t · |pq|     for all p, q ∈ S

The parameter t ≥ 1 is called the "stretch factor" or "dilation".
Lower t means better distance preservation but denser graphs.

Dependencies:
    - ``numpy``, ``scipy`` for core computation
    - ``networkx`` for shortest-path verification
"""

from graphgallery.spanners.t_spanner import TSpannerGraph
from graphgallery.spanners.yao import YaoGraph
from graphgallery.spanners.theta import ThetaGraph
from graphgallery.spanners.wspd_spanner import WSPDSpannerGraph
from graphgallery.spanners.greedy_spanner import GreedySpannerGraph

__all__ = [
    "TSpannerGraph",
    "YaoGraph",
    "ThetaGraph",
    "WSPDSpannerGraph",
    "GreedySpannerGraph",
]


def all_spanner_builders():
    """Return instances of every spanner builder with default params."""
    return [
        TSpannerGraph(),
        YaoGraph(),
        ThetaGraph(),
        WSPDSpannerGraph(),
        GreedySpannerGraph(),
    ]

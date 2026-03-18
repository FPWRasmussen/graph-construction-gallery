"""
Graph Construction Gallery.

A comprehensive visual encyclopedia of 74 graph construction
algorithms, all demonstrated on the same canonical point layout.

Subpackages:
    proximity       10 algorithms  — distance & neighbor graphs
    triangulation    5 algorithms  — Delaunay & Voronoi variants
    spanning         6 algorithms  — MST & random spanning trees
    random_models   13 algorithms  — Erdős-Rényi, BA, WS, etc.
    lattice          9 algorithms  — grids, rings, hypercubes, etc.
    spanners         5 algorithms  — Yao, Theta, greedy spanners
    ann              6 algorithms  — NSW, HNSW, Vamana, etc.
    kernel           5 algorithms  — RBF, cosine, Jaccard, etc.
    visibility       3 algorithms  — geometric & time-series
    data_driven      5 algorithms  — correlation, GLASSO, MI, etc.
    misc             7 algorithms  — KD-tree, De Bruijn, Cayley, etc.

Quick start:
    >>> from graphgallery.points import make_two_cluster_layout
    >>> from graphgallery.proximity import KNNGraph
    >>> from graphgallery.viz import plot_graph
    >>>
    >>> layout = make_two_cluster_layout()
    >>> G = KNNGraph(k=5).build(layout)
    >>> fig = plot_graph(G, layout, title="k-NN (k=5)")
"""

__version__ = "0.1.0"
__author__ = "Graph Construction Gallery Contributors"

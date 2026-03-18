"""
Kernel & Similarity-Based Graph Construction Algorithms.

This subpackage contains 5 graph builders that derive edges from
kernel functions or similarity measures:

1.  Gaussian (RBF) Kernel Graph
2.  Cosine Similarity Graph
3.  Thresholded Similarity Graph
4.  Adaptive Bandwidth Kernel Graph
5.  Jaccard Similarity Graph

Kernel functions provide a principled way to convert distances into
similarities.  They are continuous, symmetric, and positive
everywhere, producing values between 0 and 1 that naturally serve
as edge weights.

The key distinction from proximity graphs is that kernel graphs
emphasize *similarity magnitude* (edge weights) rather than binary
connectivity decisions, though thresholding can convert them to
unweighted graphs.

These graphs are widely used in spectral clustering, kernel PCA,
manifold learning, and graph neural networks.
"""

from graphgallery.kernel.gaussian_rbf import GaussianRBFGraph
from graphgallery.kernel.cosine import CosineSimilarityGraph
from graphgallery.kernel.thresholded import ThresholdedSimilarityGraph
from graphgallery.kernel.adaptive_bandwidth import AdaptiveBandwidthGraph
from graphgallery.kernel.jaccard import JaccardSimilarityGraph

__all__ = [
    "GaussianRBFGraph",
    "CosineSimilarityGraph",
    "ThresholdedSimilarityGraph",
    "AdaptiveBandwidthGraph",
    "JaccardSimilarityGraph",
]


def all_kernel_builders():
    """Return instances of every kernel/similarity builder with defaults."""
    return [
        GaussianRBFGraph(),
        CosineSimilarityGraph(),
        ThresholdedSimilarityGraph(),
        AdaptiveBandwidthGraph(),
        JaccardSimilarityGraph(),
    ]

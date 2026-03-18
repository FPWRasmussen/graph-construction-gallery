"""
Cosine Similarity Graph.

The cosine similarity measures the angular closeness between two
vectors, independent of their magnitude:

    cos(x_i, x_j) = ⟨x_i, x_j⟩ / (‖x_i‖ · ‖x_j‖)

Values range from -1 (opposite directions) through 0 (orthogonal)
to +1 (same direction).  For nonnegative data, values are in [0, 1].

Cosine similarity is widely used in:
    - Text mining / NLP (TF-IDF, embeddings)
    - Recommendation systems (user/item similarity)
    - Information retrieval
    - Any domain where feature scale is less important than direction

The graph connects pairs whose cosine similarity exceeds a threshold.

Note: For 2D point coordinates (our gallery layout), cosine similarity
measures the angular relationship from the origin, which may not be
the most natural metric.  However, it demonstrates the algorithm
clearly and produces visually interesting graphs.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.kernel._kernel_utils import (
    cosine_similarity_matrix,
    similarity_matrix_stats,
)


class CosineSimilarityGraph(GraphBuilder):
    """Connect points whose cosine similarity exceeds a threshold.

    Parameters:
        threshold: Minimum cosine similarity for edge creation.
        weighted: If True, edge weights are the cosine similarities.
            If False, edges are unweighted.
    """

    slug = "cosine"
    category = "kernel"

    def __init__(
        self,
        threshold: float = 0.8,
        weighted: bool = True,
    ):
        self.threshold = threshold
        self.weighted = weighted

    @property
    def name(self) -> str:
        return "Cosine Similarity"

    @property
    def description(self) -> str:
        return (
            f"Connect pairs with cosine similarity > {self.threshold}. "
            f"Measures angular closeness."
        )

    @property
    def complexity(self) -> str:
        return "O(n²d)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "threshold", "Minimum cosine similarity",
                "float", 0.8, "-1 ≤ threshold ≤ 1",
            ),
            ParamInfo(
                "weighted", "Use similarity as edge weight",
                "bool", True,
            ),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points

        # Compute cosine similarity matrix
        S = cosine_similarity_matrix(layout.points)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = S[i, j]
                if sim >= self.threshold:
                    if self.weighted:
                        G.add_edge(i, j, weight=float(sim))
                    else:
                        G.add_edge(i, j)

        stats = similarity_matrix_stats(S)
        G.graph["algorithm"] = "cosine_similarity"
        G.graph["threshold"] = self.threshold
        G.graph["similarity_stats"] = stats

        return G

"""
Correlation Graph.

The correlation graph connects variables whose Pearson correlation
coefficient exceeds a threshold.  The Pearson correlation measures
the *linear* relationship between two variables:

    ρ(X, Y) = Cov(X, Y) / (σ_X · σ_Y)

Values range from -1 (perfect negative correlation) through 0
(no linear relationship) to +1 (perfect positive correlation).

Important caveat: Pearson correlation only captures *linear*
dependencies.  Two variables can have zero correlation but strong
nonlinear dependence (e.g., Y = X²).  For nonlinear relationships,
see mutual information.

Also, correlation does not imply causation, and marginal correlation
does not account for confounding variables.  For conditional
independence, see the partial correlation graph.

The correlation graph is the simplest and most widely used
statistical graph construction method.  It is the starting point
for many network analysis pipelines in neuroscience (functional
connectivity), finance (stock correlation networks), and genomics
(gene co-expression networks).
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.data_driven._data_utils import (
    generate_spatially_correlated_data,
    estimate_covariance,
    covariance_to_correlation,
)


class CorrelationGraph(GraphBuilder):
    """Connect variables with Pearson correlation above a threshold.

    Parameters:
        threshold: Minimum absolute correlation for edge creation.
        n_samples: Number of synthetic observations to generate.
        length_scale: Spatial correlation scale for data generation.
        use_absolute: If True, threshold on |ρ| (ignoring sign).
            If False, threshold on ρ directly (only positive corr).
        seed: Random seed.
    """

    slug = "correlation"
    category = "data_driven"

    def __init__(
        self,
        threshold: float = 0.5,
        n_samples: int = 500,
        length_scale: float = 1.0,
        use_absolute: bool = True,
        seed: int | None = None,
    ):
        self.threshold = threshold
        self.n_samples = n_samples
        self.length_scale = length_scale
        self.use_absolute = use_absolute
        self.seed = seed

    @property
    def name(self) -> str:
        return "Correlation Graph"

    @property
    def description(self) -> str:
        mode = "|ρ|" if self.use_absolute else "ρ"
        return (
            f"Connect variables with Pearson {mode} ≥ {self.threshold}. "
            f"From {self.n_samples} synthetic observations."
        )

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n² · m) where m = n_samples"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "threshold", "Minimum correlation for edges",
                "float", 0.5, "0 ≤ threshold ≤ 1",
            ),
            ParamInfo(
                "n_samples", "Number of synthetic observations",
                "int", 500, "≥ n",
            ),
            ParamInfo(
                "length_scale", "Spatial correlation scale",
                "float", 1.0, "> 0",
            ),
            ParamInfo(
                "use_absolute", "Threshold on |ρ| instead of ρ",
                "bool", True,
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        seed = self.seed if self.seed is not None else (layout.seed + 301)

        # Generate synthetic data
        data = generate_spatially_correlated_data(
            layout,
            n_samples=self.n_samples,
            length_scale=self.length_scale,
            seed=seed,
        )

        # Estimate correlation matrix
        cov = estimate_covariance(data, method="empirical")
        corr = covariance_to_correlation(cov)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                rho = corr[i, j]
                test_value = abs(rho) if self.use_absolute else rho

                if test_value >= self.threshold:
                    G.add_edge(
                        i, j,
                        weight=float(abs(rho)),
                        correlation=float(rho),
                        sign=1 if rho > 0 else -1,
                    )

        G.graph["algorithm"] = "correlation"
        G.graph["threshold"] = self.threshold
        G.graph["n_samples"] = self.n_samples
        G.graph["correlation_matrix"] = corr

        return G

"""
Partial Correlation Graph (Gaussian Graphical Model).

The partial correlation between variables i and j measures their
*direct* linear relationship after removing the linear effects of
all other variables.  It is computed from the precision matrix
(inverse covariance):

    ρ_{ij|rest} = -Θ_{ij} / √(Θ_{ii} · Θ_{jj})

where Θ = Σ⁻¹ is the precision matrix.

Key distinction from marginal correlation:
    - Marginal correlation captures both direct and indirect
      relationships (A↔B may be high because A→C→B).
    - Partial correlation isolates ONLY direct relationships.
    - Zero partial correlation ⟺ conditional independence (for Gaussian).

The partial correlation graph is the undirected Gaussian graphical
model (GGM).  In this model, an absent edge means that the two
variables are conditionally independent given all other variables.

This is a fundamental tool in:
    - Neuroscience (brain connectivity)
    - Genomics (gene regulatory networks)
    - Finance (direct stock relationships)
    - Causal discovery (first step in the PC algorithm)

Reference:
    Lauritzen, S.L. (1996). "Graphical Models." Oxford University Press.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.data_driven._data_utils import (
    generate_spatially_correlated_data,
    estimate_covariance,
    safe_inverse,
    precision_to_partial_correlation,
)


class PartialCorrelationGraph(GraphBuilder):
    """Connect variables with significant partial correlation.

    Infers the structure of the Gaussian graphical model by
    thresholding the partial correlation matrix derived from
    the precision (inverse covariance) matrix.

    Parameters:
        threshold: Minimum |partial correlation| for edge creation.
        n_samples: Number of synthetic observations.
        length_scale: Spatial correlation scale.
        regularization: Ridge regularization for matrix inversion.
        seed: Random seed.
    """

    slug = "partial_correlation"
    category = "data_driven"

    def __init__(
        self,
        threshold: float = 0.15,
        n_samples: int = 500,
        length_scale: float = 1.0,
        regularization: float = 0.01,
        seed: int | None = None,
    ):
        self.threshold = threshold
        self.n_samples = n_samples
        self.length_scale = length_scale
        self.regularization = regularization
        self.seed = seed

    @property
    def name(self) -> str:
        return "Partial Correlation"

    @property
    def description(self) -> str:
        return (
            f"Gaussian graphical model: direct linear relationships "
            f"via precision matrix. |ρ_partial| ≥ {self.threshold}."
        )

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        return "O(n³) for matrix inversion + O(n² · m)"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "threshold", "Minimum |partial correlation|",
                "float", 0.15, "0 ≤ threshold ≤ 1",
            ),
            ParamInfo(
                "n_samples", "Synthetic observations",
                "int", 500, "≥ n",
            ),
            ParamInfo(
                "length_scale", "Spatial correlation scale",
                "float", 1.0, "> 0",
            ),
            ParamInfo(
                "regularization", "Ridge for matrix inversion",
                "float", 0.01, "≥ 0",
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        seed = self.seed if self.seed is not None else (layout.seed + 302)

        # Generate synthetic data
        data = generate_spatially_correlated_data(
            layout,
            n_samples=self.n_samples,
            length_scale=self.length_scale,
            seed=seed,
        )

        # Estimate covariance and invert to get precision
        cov = estimate_covariance(data, method="empirical")
        precision = safe_inverse(cov, regularization=self.regularization)

        # Compute partial correlations
        pcorr = precision_to_partial_correlation(precision)

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                pc = pcorr[i, j]
                if abs(pc) >= self.threshold:
                    G.add_edge(
                        i, j,
                        weight=float(abs(pc)),
                        partial_correlation=float(pc),
                        sign=1 if pc > 0 else -1,
                    )

        G.graph["algorithm"] = "partial_correlation"
        G.graph["threshold"] = self.threshold
        G.graph["n_samples"] = self.n_samples
        G.graph["partial_correlation_matrix"] = pcorr
        G.graph["precision_matrix"] = precision

        return G

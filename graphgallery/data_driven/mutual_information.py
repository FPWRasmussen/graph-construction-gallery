"""
Mutual Information Graph.

Mutual information (MI) measures the total (linear AND nonlinear)
statistical dependence between two random variables:

    MI(X, Y) = Σ p(x,y) · log(p(x,y) / (p(x)·p(y)))

Properties:
    - MI(X, Y) ≥ 0  (non-negative)
    - MI(X, Y) = 0  ⟺  X and Y are independent
    - MI(X, Y) = MI(Y, X)  (symmetric)
    - Captures ALL dependencies, not just linear
    - Invariant under invertible transformations of X or Y

Compared to correlation:
    - Correlation = 0 does NOT imply independence
    - MI = 0 DOES imply independence
    - MI detects nonlinear relationships like Y = X², Y = sin(X)

This builder uses both histogram-based and KSG (Kraskov-Stögbauer-
Grassberger) estimators.  Nonlinear data generation options showcase
MI's ability to detect dependencies invisible to correlation.

Reference:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). "Estimating
    mutual information." Physical Review E, 69(6), 066138.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

from graphgallery.base import GraphBuilder, ParamInfo
from graphgallery.points import PointLayout
from graphgallery.data_driven._data_utils import (
    generate_spatially_correlated_data,
    generate_nonlinear_data,
    estimate_mutual_information,
    estimate_mutual_information_ksg,
)


class MutualInformationGraph(GraphBuilder):
    """Connect variables with significant mutual information.

    Captures both linear and nonlinear statistical dependencies.

    Parameters:
        threshold: Minimum MI for edge creation.
        n_samples: Number of synthetic observations.
        length_scale: Spatial correlation scale.
        estimator: MI estimation method (``"histogram"`` or ``"ksg"``).
        n_bins: Bins for histogram estimator.
        ksg_k: Neighbors for KSG estimator.
        nonlinear: If True, apply nonlinear transforms to create
            dependencies invisible to correlation.
        seed: Random seed.
    """

    slug = "mutual_information"
    category = "data_driven"

    def __init__(
        self,
        threshold: float = 0.1,
        n_samples: int = 500,
        length_scale: float = 1.0,
        estimator: Literal["histogram", "ksg"] = "ksg",
        n_bins: int = 20,
        ksg_k: int = 5,
        nonlinear: bool = False,
        seed: int | None = None,
    ):
        self.threshold = threshold
        self.n_samples = n_samples
        self.length_scale = length_scale
        self.estimator = estimator
        self.n_bins = n_bins
        self.ksg_k = ksg_k
        self.nonlinear = nonlinear
        self.seed = seed

    @property
    def name(self) -> str:
        return "Mutual Information"

    @property
    def description(self) -> str:
        nl = " (nonlinear data)" if self.nonlinear else ""
        return (
            f"MI-based edges ({self.estimator} estimator){nl}. "
            f"Detects all dependencies, not just linear."
        )

    @property
    def is_spatial(self) -> bool:
        return False

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def complexity(self) -> str:
        if self.estimator == "ksg":
            return "O(n² · m · log m) for KSG"
        return "O(n² · m) for histogram"

    def params_info(self) -> list[ParamInfo]:
        return [
            ParamInfo(
                "threshold", "Minimum MI for edges",
                "float", 0.1, "≥ 0",
            ),
            ParamInfo("n_samples", "Synthetic observations", "int", 500, "≥ 30"),
            ParamInfo("length_scale", "Spatial correlation scale", "float", 1.0, "> 0"),
            ParamInfo(
                "estimator", "MI estimation method",
                "str", "ksg", "histogram | ksg",
            ),
            ParamInfo("n_bins", "Histogram bins", "int", 20, "≥ 5"),
            ParamInfo("ksg_k", "KSG neighbor count", "int", 5, "≥ 1"),
            ParamInfo(
                "nonlinear", "Apply nonlinear transforms",
                "bool", False,
            ),
            ParamInfo("seed", "Random seed", "int | None", None),
        ]

    def build(self, layout: PointLayout) -> nx.Graph:
        self.validate_layout(layout)
        n = layout.n_points
        seed = self.seed if self.seed is not None else (layout.seed + 304)

        # Generate data
        if self.nonlinear:
            data = generate_nonlinear_data(
                layout,
                n_samples=self.n_samples,
                length_scale=self.length_scale,
                nonlinearity="mixed",
                seed=seed,
            )
        else:
            data = generate_spatially_correlated_data(
                layout,
                n_samples=self.n_samples,
                length_scale=self.length_scale,
                seed=seed,
            )

        # Compute pairwise MI
        mi_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                if self.estimator == "ksg":
                    mi = estimate_mutual_information_ksg(
                        data[:, i], data[:, j], k=self.ksg_k
                    )
                else:
                    mi = estimate_mutual_information(
                        data[:, i], data[:, j], n_bins=self.n_bins
                    )

                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if mi_matrix[i, j] >= self.threshold:
                    G.add_edge(
                        i, j,
                        weight=float(mi_matrix[i, j]),
                        mutual_information=float(mi_matrix[i, j]),
                    )

        G.graph["algorithm"] = "mutual_information"
        G.graph["estimator"] = self.estimator
        G.graph["threshold"] = self.threshold
        G.graph["n_samples"] = self.n_samples
        G.graph["nonlinear"] = self.nonlinear
        G.graph["mi_matrix"] = mi_matrix

        return G

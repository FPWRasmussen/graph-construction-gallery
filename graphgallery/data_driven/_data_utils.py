"""
Shared utilities for data-driven graph construction.

Provides synthetic data generation from spatial layouts, covariance
estimation, matrix regularization, and statistical analysis helpers.

The core idea: given n points in 2D, we treat them as n random
variables and generate m observations from a multivariate Gaussian
whose covariance structure reflects spatial proximity.  This creates
a principled bridge between our geometric point layout and
statistical graph learning methods.
"""

from __future__ import annotations

import numpy as np

from graphgallery.points import PointLayout, pairwise_distances


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_spatially_correlated_data(
    layout: PointLayout,
    n_samples: int = 500,
    length_scale: float = 1.0,
    noise_std: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Generate synthetic multivariate data with spatial covariance.

    Each of the n points becomes a random variable.  Observations are
    drawn from a multivariate Gaussian whose covariance is determined
    by a squared-exponential (Gaussian) kernel on the point positions:

        Σ_ij = exp(-‖p_i - p_j‖² / (2 · ℓ²)) + noise · δ_ij

    This ensures that nearby points have highly correlated values while
    distant points are nearly independent.

    Args:
        layout: The point layout (n points define n variables).
        n_samples: Number of observations to generate.
        length_scale: Spatial correlation length scale ℓ.
            Smaller → more local correlation.
            Larger → broader correlation.
        noise_std: Per-variable noise standard deviation.
        seed: Random seed.

    Returns:
        (n_samples, n) data matrix where each column is a variable
        corresponding to one point.
    """
    n = layout.n_points
    seed = seed if seed is not None else (layout.seed + 300)
    rng = np.random.default_rng(seed)

    # Build spatial covariance matrix
    dist = pairwise_distances(layout.points)
    cov = np.exp(-dist ** 2 / (2.0 * length_scale ** 2))

    # Add noise on diagonal for numerical stability and to model
    # observation noise
    cov += noise_std ** 2 * np.eye(n)

    # Ensure positive definiteness
    cov = _nearest_positive_definite(cov)

    # Generate samples
    data = rng.multivariate_normal(
        mean=np.zeros(n),
        cov=cov,
        size=n_samples,
    )

    return data


def generate_nonlinear_data(
    layout: PointLayout,
    n_samples: int = 500,
    length_scale: float = 1.0,
    noise_std: float = 0.1,
    nonlinearity: str = "square",
    seed: int | None = None,
) -> np.ndarray:
    """Generate data with nonlinear dependencies between variables.

    First generates spatially correlated Gaussian data, then applies
    nonlinear transformations to specific variables to create
    dependencies that are invisible to linear correlation but
    detectable by mutual information.

    Args:
        layout: Point layout.
        n_samples: Number of observations.
        length_scale: Spatial correlation scale.
        noise_std: Noise level.
        nonlinearity: Type of nonlinear transform:
            - ``"square"``: x → x²
            - ``"abs"``: x → |x|
            - ``"sin"``: x → sin(πx)
            - ``"mixed"``: apply different transforms to different variables
        seed: Random seed.

    Returns:
        (n_samples, n) transformed data matrix.
    """
    data = generate_spatially_correlated_data(
        layout, n_samples, length_scale, noise_std, seed
    )

    rng = np.random.default_rng((seed or 42) + 1)
    n = data.shape[1]

    if nonlinearity == "square":
        # Square every other variable
        for j in range(0, n, 2):
            data[:, j] = data[:, j] ** 2

    elif nonlinearity == "abs":
        for j in range(0, n, 3):
            data[:, j] = np.abs(data[:, j])

    elif nonlinearity == "sin":
        for j in range(0, n, 2):
            data[:, j] = np.sin(np.pi * data[:, j])

    elif nonlinearity == "mixed":
        transforms = [
            lambda x: x ** 2,
            lambda x: np.abs(x),
            lambda x: np.sin(np.pi * x),
            lambda x: x,  # Identity
        ]
        for j in range(n):
            t = transforms[j % len(transforms)]
            data[:, j] = t(data[:, j])

    return data


# ---------------------------------------------------------------------------
# Covariance and correlation estimation
# ---------------------------------------------------------------------------

def estimate_covariance(
    data: np.ndarray,
    method: str = "empirical",
    shrinkage: float = 0.0,
) -> np.ndarray:
    """Estimate the covariance matrix from data.

    Args:
        data: (n_samples, n) data matrix.
        method: Estimation method:
            - ``"empirical"``: Standard sample covariance.
            - ``"shrinkage"``: Ledoit-Wolf-style shrinkage toward identity.
        shrinkage: Shrinkage intensity (0 = pure empirical, 1 = identity).

    Returns:
        (n, n) covariance matrix estimate.
    """
    n_samples, n = data.shape

    # Center the data
    data_centered = data - data.mean(axis=0)

    if method == "empirical":
        cov = (data_centered.T @ data_centered) / (n_samples - 1)
    elif method == "shrinkage":
        empirical = (data_centered.T @ data_centered) / (n_samples - 1)
        target = np.eye(n) * np.trace(empirical) / n
        cov = (1.0 - shrinkage) * empirical + shrinkage * target
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    return cov


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Args:
        cov: (n, n) covariance matrix.

    Returns:
        (n, n) correlation matrix with diagonal = 1.
    """
    std = np.sqrt(np.diag(cov))
    std = np.maximum(std, 1e-12)
    corr = cov / np.outer(std, std)
    np.clip(corr, -1.0, 1.0, out=corr)
    np.fill_diagonal(corr, 1.0)
    return corr


def precision_to_partial_correlation(precision: np.ndarray) -> np.ndarray:
    """Convert a precision (inverse covariance) matrix to partial correlations.

    The partial correlation between variables i and j (controlling for
    all other variables) is:

        ρ_ij|rest = -Θ_ij / √(Θ_ii · Θ_jj)

    where Θ is the precision matrix.

    Args:
        precision: (n, n) precision matrix.

    Returns:
        (n, n) partial correlation matrix with values in [-1, 1].
    """
    diag = np.sqrt(np.diag(precision))
    diag = np.maximum(diag, 1e-12)
    pcorr = -precision / np.outer(diag, diag)
    np.fill_diagonal(pcorr, 1.0)
    np.clip(pcorr, -1.0, 1.0, out=pcorr)
    return pcorr


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

def _nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """Find the nearest positive definite matrix to A.

    Uses the method of Higham (1988): symmetrize, then clamp
    eigenvalues to a small positive minimum.

    Args:
        A: (n, n) symmetric matrix.

    Returns:
        (n, n) positive definite matrix.
    """
    B = (A + A.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def safe_inverse(
    matrix: np.ndarray,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Compute matrix inverse with regularization for stability.

    Args:
        matrix: (n, n) square matrix.
        regularization: Ridge added to diagonal.

    Returns:
        (n, n) inverse matrix.
    """
    n = matrix.shape[0]
    regularized = matrix + regularization * np.eye(n)
    return np.linalg.inv(regularized)


# ---------------------------------------------------------------------------
# Mutual information estimation
# ---------------------------------------------------------------------------

def estimate_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Estimate mutual information between two 1D arrays using binning.

    MI(X, Y) = Σ_x,y p(x,y) · log(p(x,y) / (p(x) · p(y)))

    Uses histogram-based density estimation.

    Args:
        x: (n_samples,) first variable.
        y: (n_samples,) second variable.
        n_bins: Number of bins per axis.

    Returns:
        Estimated mutual information in nats (≥ 0).
    """
    # 2D histogram
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)

    # Convert to probability
    p_xy = hist_2d / hist_2d.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    # Compute MI, avoiding log(0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

    return max(0.0, mi)


def estimate_mutual_information_ksg(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
) -> float:
    """Estimate MI using the KSG (Kraskov-Stögbauer-Grassberger) estimator.

    A non-parametric, bias-corrected estimator based on k-nearest
    neighbor distances.  More accurate than histogram-based methods,
    especially for small samples.

    Args:
        x: (n_samples,) first variable.
        y: (n_samples,) second variable.
        k: Number of nearest neighbors.

    Returns:
        Estimated mutual information in nats (can be slightly negative
        due to estimation bias, clipped to 0).
    """
    from scipy.special import digamma

    n = len(x)
    if n < k + 1:
        return 0.0

    # Reshape for distance computation
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xy = np.hstack([x, y])

    # For each point, find the k-th nearest neighbor in joint space (Chebyshev)
    from scipy.spatial import KDTree

    tree_xy = KDTree(xy, leafsize=16)
    tree_x = KDTree(x, leafsize=16)
    tree_y = KDTree(y, leafsize=16)

    mi_sum = 0.0
    for i in range(n):
        # k+1 because the point itself is included
        dists, _ = tree_xy.query(xy[i], k=k + 1, p=np.inf)
        eps = dists[-1]  # Distance to k-th neighbor in joint space

        if eps < 1e-15:
            eps = 1e-15

        # Count points within eps in marginal spaces
        n_x = tree_x.query_ball_point(x[i], r=eps, p=np.inf)
        n_y = tree_y.query_ball_point(y[i], r=eps, p=np.inf)

        n_x_count = len(n_x) - 1  # Exclude self
        n_y_count = len(n_y) - 1

        if n_x_count > 0 and n_y_count > 0:
            mi_sum += digamma(n_x_count) + digamma(n_y_count)

    mi = digamma(k) - mi_sum / n + digamma(n)
    return max(0.0, float(mi))

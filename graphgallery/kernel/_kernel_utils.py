"""
Shared utilities for kernel and similarity graph construction.

Provides common kernel functions, similarity matrices, sparsification
strategies, and analysis helpers used across multiple builders.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

def gaussian_kernel_matrix(
    points: np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    """Compute the Gaussian (RBF) kernel matrix.

    K(i, j) = exp(-‖x_i - x_j‖² / (2σ²))

    When sigma is a scalar, all pairs use the same bandwidth.  When
    sigma is a (n,) array, pair (i, j) uses σ_i · σ_j (geometric
    mean bandwidth) following the self-tuning convention.

    Args:
        points: (n, d) array of point coordinates.
        sigma: Bandwidth — scalar or (n,) per-point array.

    Returns:
        (n, n) symmetric kernel matrix with diagonal = 1.
    """
    # Compute squared distance matrix efficiently
    sq_dist = _squared_distance_matrix(points)

    if np.isscalar(sigma) or (isinstance(sigma, np.ndarray) and sigma.ndim == 0):
        s = float(sigma)
        if s <= 0:
            raise ValueError(f"sigma must be > 0, got {s}")
        K = np.exp(-sq_dist / (2.0 * s * s))
    else:
        # Per-point bandwidth: σ_ij = σ_i · σ_j
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.shape != (points.shape[0],):
            raise ValueError(
                f"Per-point sigma shape {sigma.shape} != ({points.shape[0]},)"
            )
        sigma = np.maximum(sigma, 1e-12)  # Prevent division by zero
        sigma_prod = np.outer(sigma, sigma)
        K = np.exp(-sq_dist / (2.0 * sigma_prod))

    return K


def cosine_similarity_matrix(points: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine similarity matrix.

    cos(x_i, x_j) = ⟨x_i, x_j⟩ / (‖x_i‖ · ‖x_j‖)

    Args:
        points: (n, d) array of point coordinates.

    Returns:
        (n, n) symmetric matrix with values in [-1, 1].
        Diagonal entries are 1.0 (or 0 for zero vectors).
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-12)
    normalized = points / norms

    similarity = normalized @ normalized.T

    # Clamp to [-1, 1] due to floating-point errors
    np.clip(similarity, -1.0, 1.0, out=similarity)

    return similarity


def jaccard_binary_similarity_matrix(
    features: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Compute pairwise Jaccard similarity from continuous features.

    Features are first binarized: feature > threshold → 1, else → 0.
    Then Jaccard is computed as:

        J(A, B) = |A ∩ B| / |A ∪ B|

    For continuous features where binarization is unnatural, we
    generate binary features from multiple random thresholds.

    Args:
        features: (n, d) array.
        threshold: Binarization cutoff.

    Returns:
        (n, n) symmetric matrix with values in [0, 1].
    """
    binary = (features > threshold).astype(np.float64)

    n = binary.shape[0]
    similarity = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            intersection = np.sum(np.minimum(binary[i], binary[j]))
            union = np.sum(np.maximum(binary[i], binary[j]))

            if union > 0:
                sim = intersection / union
            else:
                sim = 0.0

            similarity[i, j] = sim
            similarity[j, i] = sim

    return similarity


def jaccard_multithreshold_similarity(
    features: np.ndarray,
    n_thresholds: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Compute Jaccard similarity using multiple random thresholds.

    For each threshold, the features are binarized and Jaccard is
    computed.  The final similarity is the average across all thresholds.
    This produces more robust similarity estimates for continuous data.

    Args:
        features: (n, d) array.
        n_thresholds: Number of random thresholds to use.
        seed: Random seed.

    Returns:
        (n, n) symmetric matrix with values in [0, 1].
    """
    rng = np.random.default_rng(seed)
    n = features.shape[0]

    # Generate thresholds from the feature value range
    f_min, f_max = features.min(), features.max()
    thresholds = rng.uniform(f_min, f_max, size=n_thresholds)

    cumulative = np.zeros((n, n), dtype=np.float64)

    for t in thresholds:
        cumulative += jaccard_binary_similarity_matrix(features, threshold=t)

    return cumulative / n_thresholds


# ---------------------------------------------------------------------------
# Sparsification strategies
# ---------------------------------------------------------------------------

def threshold_sparsify(
    similarity: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Zero out entries below a threshold.

    Args:
        similarity: (n, n) similarity matrix.
        threshold: Minimum value to keep.

    Returns:
        Sparsified (n, n) matrix (modified copy).
    """
    sparse = similarity.copy()
    sparse[sparse < threshold] = 0.0
    np.fill_diagonal(sparse, 0.0)
    return sparse


def knn_sparsify(
    similarity: np.ndarray,
    k: int,
    symmetric: bool = True,
) -> np.ndarray:
    """Keep only the k largest similarities per row.

    Args:
        similarity: (n, n) similarity matrix.
        k: Number of neighbors to keep.
        symmetric: If True, symmetrize (union of directed kNN).

    Returns:
        Sparsified (n, n) matrix.
    """
    n = similarity.shape[0]
    sparse = np.zeros_like(similarity)

    for i in range(n):
        row = similarity[i].copy()
        row[i] = -np.inf  # Exclude self
        top_k = np.argpartition(row, -k)[-k:]
        for j in top_k:
            if row[j] > 0:
                sparse[i, j] = similarity[i, j]

    if symmetric:
        sparse = np.maximum(sparse, sparse.T)

    return sparse


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _squared_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances.

    Uses the expansion ‖x-y‖² = ‖x‖² + ‖y‖² - 2⟨x,y⟩ for efficiency.
    """
    sq_norms = np.sum(points ** 2, axis=1)
    sq_dist = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (points @ points.T)
    # Clamp negative values due to floating-point errors
    np.maximum(sq_dist, 0.0, out=sq_dist)
    return sq_dist


def similarity_matrix_stats(S: np.ndarray) -> dict[str, float]:
    """Compute summary statistics of a similarity matrix.

    Args:
        S: (n, n) symmetric similarity matrix.

    Returns:
        Dict with keys: mean, median, std, min, max, sparsity, nnz.
    """
    n = S.shape[0]
    # Upper triangle (excluding diagonal)
    upper = S[np.triu_indices(n, k=1)]
    total_pairs = len(upper)
    nnz = int(np.count_nonzero(upper))

    return {
        "mean": float(upper.mean()) if total_pairs > 0 else 0.0,
        "median": float(np.median(upper)) if total_pairs > 0 else 0.0,
        "std": float(upper.std()) if total_pairs > 0 else 0.0,
        "min": float(upper.min()) if total_pairs > 0 else 0.0,
        "max": float(upper.max()) if total_pairs > 0 else 0.0,
        "sparsity": 1.0 - nnz / total_pairs if total_pairs > 0 else 1.0,
        "nnz": nnz,
    }

"""Joint (multivariate) fidelity metrics.

Measures the distance between the full joint distributions of real and
synthetic data — going beyond marginals and pairwise correlations.

Metrics
-------
MMD
    Maximum Mean Discrepancy with an RBF kernel.  Uses the median-heuristic
    bandwidth.  Lower is better (0 = identical distributions).

Precision / Recall  (Kynkäänniemi et al. 2019)
    Precision  — fraction of synthetic samples within the support of real data
                 (high precision ⟹ no hallucinated modes).
    Recall     — fraction of real data modes covered by synthetic data
                 (high recall ⟹ no mode dropping).

Density / Coverage  (Naeem et al. 2020)
    Coverage   — fraction of real data k-NN balls that contain ≥1 synthetic
                 point (more robust alternative to Recall).

All methods use the feature space produced by FeatureProcessor (scaled +
one-hot encoded).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler


@dataclass
class JointFidelityResults:
    mmd: float
    precision: float
    recall: float
    coverage: float

    def summary(self) -> Dict[str, float]:
        return {
            "fidelity/mmd": self.mmd,
            "fidelity/precision": self.precision,
            "fidelity/recall": self.recall,
            "fidelity/coverage": self.coverage,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian (RBF) kernel matrix K(X, Y)."""
    # ||x - y||^2 via broadcasting — keep memory under control
    diff = X[:, None, :] - Y[None, :, :]          # (n, m, d)
    sq_dist = (diff ** 2).sum(axis=-1)              # (n, m)
    return np.exp(-sq_dist / (2 * sigma ** 2))


def _mmd(real: np.ndarray, synth: np.ndarray) -> float:
    """Unbiased MMD² estimate with RBF kernel, median-heuristic bandwidth."""
    # Subsample for speed if necessary
    max_samples = 2000
    if len(real) > max_samples:
        idx = np.random.choice(len(real), max_samples, replace=False)
        real = real[idx]
    if len(synth) > max_samples:
        idx = np.random.choice(len(synth), max_samples, replace=False)
        synth = synth[idx]

    # Median heuristic on pairwise distances in the combined set
    combined = np.concatenate([real, synth], axis=0)
    n_combined = len(combined)
    sample_idx = np.random.choice(n_combined, min(500, n_combined), replace=False)
    sample = combined[sample_idx]
    pairwise_sq = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
    sigma = float(np.sqrt(np.median(pairwise_sq[pairwise_sq > 0]) / 2)) + 1e-8

    K_rr = _rbf_kernel(real, real, sigma)
    K_ss = _rbf_kernel(synth, synth, sigma)
    K_rs = _rbf_kernel(real, synth, sigma)

    n, m = len(real), len(synth)
    # Unbiased estimator
    mmd2 = (
        (K_rr.sum() - np.trace(K_rr)) / (n * (n - 1))
        + (K_ss.sum() - np.trace(K_ss)) / (m * (m - 1))
        - 2 * K_rs.mean()
    )
    return float(max(0.0, mmd2))


def _knn_radii(data: np.ndarray, k: int) -> np.ndarray:
    """Return the distance to the k-th nearest neighbour for each point."""
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    knn.fit(data)
    distances, _ = knn.kneighbors(data)
    return distances[:, k]  # distance to k-th neighbour (0-indexed, skip self)


def _precision_recall(
    real: np.ndarray, synth: np.ndarray, k: int = 5
) -> tuple[float, float]:
    """Precision and Recall for distributions (Kynkäänniemi et al. 2019)."""
    real_radii = _knn_radii(real, k)
    synth_radii = _knn_radii(synth, k)

    knn_real = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    knn_real.fit(real)
    knn_synth = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    knn_synth.fit(synth)

    # Precision: fraction of synthetic points inside the support of real data
    dist_synth_to_real, idx_real = knn_real.kneighbors(synth)
    precision = float((dist_synth_to_real[:, 0] <= real_radii[idx_real[:, 0]]).mean())

    # Recall: fraction of real points inside the support of synthetic data
    dist_real_to_synth, idx_synth = knn_synth.kneighbors(real)
    recall = float((dist_real_to_synth[:, 0] <= synth_radii[idx_synth[:, 0]]).mean())

    return precision, recall


def _coverage(real: np.ndarray, synth: np.ndarray, k: int = 5) -> float:
    """Coverage (Naeem et al. 2020) — fraction of real k-NN balls containing ≥1 synthetic point."""
    real_radii = _knn_radii(real, k)
    knn_real = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    knn_real.fit(real)
    dist_synth_to_real, idx_real = knn_real.kneighbors(synth)
    # For each real point, check if any synthetic point falls within its radius
    covered = np.zeros(len(real), dtype=bool)
    for i in range(len(synth)):
        nearest_real = idx_real[i, 0]
        if dist_synth_to_real[i, 0] <= real_radii[nearest_real]:
            covered[nearest_real] = True
    return float(covered.mean())


# ---------------------------------------------------------------------------
# Estimator class
# ---------------------------------------------------------------------------

class JointFidelityEstimator:
    """Compute joint distribution fidelity metrics.

    Parameters
    ----------
    scaler :
        Scaler applied before distance computations.
    categorical_columns :
        Columns to one-hot encode.
    k :
        Number of neighbours used for Precision/Recall/Coverage.
    max_features :
        Cap on number of features passed to the distance computations.
    """

    def __init__(
        self,
        scaler: Optional[Scaler] = None,
        categorical_columns: Optional[List[str]] = None,
        k: int = 5,
        max_features: int = 100,
    ):
        self.feature_processor = FeatureProcessor(
            scaler=scaler if scaler is not None else RobustScaler(),
            categorical_columns=categorical_columns,
        )
        self.k = k
        self.max_features = max_features

    def estimate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
    ) -> JointFidelityResults:
        logger.info("Computing joint fidelity metrics (MMD, Precision, Recall, Coverage)...")

        real_arr = self.feature_processor.fit_transform(real_df)
        synth_arr = self.feature_processor.transform(synth_df)

        # Cap features to avoid memory issues with very high-dimensional data
        if real_arr.shape[1] > self.max_features:
            real_arr = real_arr[:, : self.max_features]
            synth_arr = synth_arr[:, : self.max_features]

        mmd_val = _mmd(real_arr, synth_arr)
        precision, recall = _precision_recall(real_arr, synth_arr, k=self.k)
        cov = _coverage(real_arr, synth_arr, k=self.k)

        logger.info(
            f"JointFidelity — MMD: {mmd_val:.6f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, Coverage: {cov:.4f}"
        )

        return JointFidelityResults(
            mmd=mmd_val,
            precision=precision,
            recall=recall,
            coverage=cov,
        )

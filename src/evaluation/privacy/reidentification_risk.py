"""Re-identification Risk privacy metric.

For each synthetic record, re-identification risk measures how confidently an
adversary could link that record back to a *specific* real individual.

Approach
--------
For every synthetic record we find its two nearest neighbours in the real
dataset.  If the nearest neighbour is much closer than the second nearest, the
synthetic record effectively "points to" a unique real individual — an
adversary could re-identify that person with high confidence.

We quantify this via the **distance ratio**:

    ratio = d(synth, 1st-NN) / d(synth, 2nd-NN)

A ratio close to 0 means the closest real record is *much* closer than the
runner-up — high re-identification risk.  A ratio close to 1 means the two
nearest real records are roughly equidistant — low risk.

Key metrics
-----------
- ``mean_distance_ratio``   : mean of the per-record distance ratios
- ``median_distance_ratio`` : median of the per-record distance ratios
- ``reidentification_rate`` : fraction of synthetic records whose distance
                              ratio falls below a configurable threshold
                              (default 0.5), indicating high re-identification
                              risk
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
from loguru import logger

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler
from src.evaluation.privacy.gower_distance import GowerDistanceCalculator


class ReidentificationResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Per-record arrays
    distance_ratios: np.ndarray         # d(1st-NN) / d(2nd-NN) per synthetic record
    nearest_distances: np.ndarray       # d(synth, 1st-NN)
    second_nearest_distances: np.ndarray  # d(synth, 2nd-NN)

    # Scalar summaries
    mean_distance_ratio: float
    median_distance_ratio: float
    reidentification_rate: float        # fraction below threshold
    num_at_risk: int                    # count below threshold

    def summary(self) -> dict:
        return {
            "mean_distance_ratio": self.mean_distance_ratio,
            "median_distance_ratio": self.median_distance_ratio,
            "reidentification_rate": self.reidentification_rate,
            "num_at_risk": self.num_at_risk,
        }


class ReidentificationRiskEstimator:
    """Estimate re-identification risk of synthetic records.

    Parameters
    ----------
    scaler :
        Any sklearn scaler.  Defaults to RobustScaler.
    categorical_columns :
        Column names that should be one-hot encoded rather than scaled.
    risk_threshold :
        Distance-ratio threshold below which a synthetic record is considered
        at risk of re-identification.  Default 0.5 (the nearest real record is
        at least twice as close as the second nearest).
    algorithm :
        NearestNeighbors algorithm.  ``"ball_tree"`` is the default.
        Ignored when ``distance_metric="gower"``.
    distance_metric :
        ``"euclidean"`` (default) uses FeatureProcessor + Euclidean NN search.
        ``"gower"`` uses Gower's distance on raw features.
    fitted_feature_processor :
        A pre-fitted FeatureProcessor to share with other privacy estimators.
    """

    def __init__(
        self,
        scaler: Optional[Scaler] = None,
        categorical_columns: Optional[List[str]] = None,
        risk_threshold: float = 0.5,
        algorithm: str = "ball_tree",
        distance_metric: str = "euclidean",
        fitted_feature_processor: Optional["FeatureProcessor"] = None,
    ):
        self.distance_metric = distance_metric
        self.categorical_columns = categorical_columns or []
        self.risk_threshold = risk_threshold
        self.algorithm = algorithm

        if distance_metric == "gower":
            self._gower = GowerDistanceCalculator(
                categorical_columns=self.categorical_columns,
            )
            self.feature_processor = None
        else:
            self._gower = None
            if fitted_feature_processor is not None:
                self.feature_processor = fitted_feature_processor
            else:
                self.feature_processor = FeatureProcessor(
                    scaler=scaler if scaler is not None else RobustScaler(),
                    categorical_columns=categorical_columns,
                )

        self._real_data: Optional[np.ndarray] = None
        self._real_df: Optional[pl.DataFrame] = None
        self._knn: Optional[NearestNeighbors] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, real_dataframe: pl.DataFrame) -> "ReidentificationRiskEstimator":
        """Fit on the full real dataset (reference for re-identification)."""
        self._real_df = real_dataframe

        if self.distance_metric == "gower":
            self._gower.fit(real_dataframe)
        else:
            if self.feature_processor.fitted:
                self._real_data = self.feature_processor.transform(real_dataframe)
            else:
                self._real_data = self.feature_processor.fit_transform(real_dataframe)

            self._knn = NearestNeighbors(n_neighbors=2, algorithm=self.algorithm)
            self._knn.fit(self._real_data)

        logger.info(
            f"ReidentificationRiskEstimator fitted ({self.distance_metric}): "
            f"{len(real_dataframe)} real records."
        )
        return self

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(self, synthetic_dataframe: pl.DataFrame) -> ReidentificationResults:
        """Compute re-identification risk for each synthetic record."""
        if self.distance_metric == "gower":
            if not self._gower.fitted:
                raise RuntimeError("Call fit() before estimate().")

            distances, _ = self._gower.nearest_neighbor_distances(
                synthetic_dataframe, self._real_df, n_neighbors=2,
            )
        else:
            if self._knn is None:
                raise RuntimeError("Call fit() before estimate().")

            synth_array = self.feature_processor.transform(synthetic_dataframe)
            distances, _ = self._knn.kneighbors(synth_array, n_neighbors=2)

        nearest = distances[:, 0]
        second_nearest = distances[:, 1]

        # Avoid division by zero: if 2nd-NN distance is 0, ratio = 1.0
        # (both records are equidistant at 0 — no unique linkage)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                second_nearest > 0,
                nearest / second_nearest,
                np.where(nearest == 0, 1.0, 0.0),
            )

        at_risk_mask = ratios < self.risk_threshold
        reid_rate = float(at_risk_mask.mean())

        mean_ratio = float(np.mean(ratios))
        median_ratio = float(np.median(ratios))

        logger.info(
            f"Re-identification risk — mean ratio: {mean_ratio:.4f}, "
            f"median ratio: {median_ratio:.4f}, "
            f"re-identification rate (threshold={self.risk_threshold}): {reid_rate:.2%}"
        )

        return ReidentificationResults(
            distance_ratios=ratios,
            nearest_distances=nearest,
            second_nearest_distances=second_nearest,
            mean_distance_ratio=mean_ratio,
            median_distance_ratio=median_ratio,
            reidentification_rate=reid_rate,
            num_at_risk=int(at_risk_mask.sum()),
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        results: ReidentificationResults,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Distance-ratio histogram and nearest-vs-second-nearest scatter."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # --- Ratio histogram ---
        ax = axes[0]
        ax.hist(results.distance_ratios, bins=50, color="steelblue", alpha=0.7,
                edgecolor="white")
        ax.axvline(self.risk_threshold, color="red", linestyle="--",
                   label=f"Threshold ({self.risk_threshold})")
        ax.set_xlabel("Distance ratio (1st-NN / 2nd-NN)")
        ax.set_ylabel("Count")
        ax.set_title("Re-identification Risk — Distance Ratios")
        ax.legend(fontsize=8)

        # --- Scatter: nearest vs second-nearest ---
        ax = axes[1]
        at_risk = results.distance_ratios < self.risk_threshold
        ax.scatter(
            results.nearest_distances[~at_risk],
            results.second_nearest_distances[~at_risk],
            alpha=0.3, s=8, color="steelblue", label="Low risk",
        )
        ax.scatter(
            results.nearest_distances[at_risk],
            results.second_nearest_distances[at_risk],
            alpha=0.5, s=12, color="red", label="At risk",
        )
        # Plot the threshold line (ratio = threshold → y = x / threshold)
        max_val = max(
            results.nearest_distances.max(),
            results.second_nearest_distances.max(),
        )
        xs = np.linspace(0, max_val, 100)
        ax.plot(xs, xs / self.risk_threshold, "r--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Distance to 1st-NN (real)")
        ax.set_ylabel("Distance to 2nd-NN (real)")
        ax.set_title("Nearest vs. Second-Nearest Distance")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"Re-identification Risk  |  rate={results.reidentification_rate:.2%}  |  "
            f"median ratio={results.median_distance_ratio:.3f}",
            fontsize=11,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

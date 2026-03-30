"""Distance to Closest Record (DCR) privacy metric.

For each synthetic record, DCR is the distance to its nearest neighbour in the
real dataset.  A holdout baseline (real-to-real distance) is computed by
splitting the real data in half, so that the two distributions can be compared.

Key outputs
-----------
- median_dcr_ratio      : median(DCR_synth) / median(DCR_holdout)  — near 1.0 is good
- privacy_at_risk       : fraction of synthetic records whose DCR falls below the
                          `par_percentile`-th percentile of the real holdout DCR
- KDE + CDF plots comparing the two DCR distributions
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


class DCRResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Raw DCR arrays
    dcr_synthetic: np.ndarray       # one value per synthetic record
    dcr_real_holdout: np.ndarray    # one value per holdout real record

    # Scalar summaries
    median_dcr_synthetic: float
    median_dcr_holdout: float
    median_dcr_ratio: float         # synth / holdout — near 1.0 is ideal
    privacy_at_risk: float          # fraction of synthetic records below PAR threshold

    # Indices of synthetic records at risk
    at_risk_indices: np.ndarray

    def summary(self) -> dict:
        return {
            "median_dcr_synthetic": self.median_dcr_synthetic,
            "median_dcr_holdout": self.median_dcr_holdout,
            "median_dcr_ratio": self.median_dcr_ratio,
            "privacy_at_risk": self.privacy_at_risk,
            "num_at_risk": int(len(self.at_risk_indices)),
        }


class DCREstimator:
    """Compute Distance to Closest Record privacy metrics.

    Parameters
    ----------
    scaler :
        Any sklearn scaler.  RobustScaler is the default because peptide data
        contains heavy-tailed distributions.
    categorical_columns :
        Column names that should be one-hot encoded rather than scaled.
    holdout_fraction :
        Fraction of real data held out as the reference set for the real-to-real
        DCR baseline.  Default 0.5.
    par_percentile :
        Percentile of the real holdout DCR distribution used as the
        Privacy-at-Risk threshold.  Default 5 (5th percentile).
    algorithm :
        NearestNeighbors algorithm.  "ball_tree" is fast for medium-dimensional
        data.
    """

    def __init__(
        self,
        scaler: Optional[Scaler] = None,
        categorical_columns: Optional[List[str]] = None,
        holdout_fraction: float = 0.5,
        par_percentile: float = 5.0,
        algorithm: str = "ball_tree",
    ):
        self.feature_processor = FeatureProcessor(
            scaler=scaler if scaler is not None else RobustScaler(),
            categorical_columns=categorical_columns,
        )
        self.holdout_fraction = holdout_fraction
        self.par_percentile = par_percentile
        self.algorithm = algorithm

        self._train_data: Optional[np.ndarray] = None
        self._holdout_data: Optional[np.ndarray] = None
        self._knn_train: Optional[NearestNeighbors] = None
        self._knn_holdout: Optional[NearestNeighbors] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, real_dataframe: pl.DataFrame) -> "DCREstimator":
        """Fit on real data — split into training and holdout halves."""
        n = len(real_dataframe)
        n_holdout = max(1, int(n * self.holdout_fraction))
        idx = np.random.permutation(n)
        train_idx = idx[n_holdout:]
        holdout_idx = idx[:n_holdout]

        train_df = real_dataframe[train_idx.tolist()]
        holdout_df = real_dataframe[holdout_idx.tolist()]

        self._train_data = self.feature_processor.fit_transform(train_df)
        self._holdout_data = self.feature_processor.transform(holdout_df)

        self._knn_train = NearestNeighbors(n_neighbors=1, algorithm=self.algorithm)
        self._knn_train.fit(self._train_data)

        # For the holdout baseline we fit a separate index on train as well —
        # the holdout records query this index so they are never their own
        # nearest neighbour.
        self._knn_holdout = self._knn_train

        logger.info(
            f"DCREstimator fitted: {len(train_df)} training / {len(holdout_df)} holdout real records."
        )
        return self

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(self, synthetic_dataframe: pl.DataFrame) -> DCRResults:
        """Compute DCR for synthetic data and the real holdout baseline."""
        if self._knn_train is None:
            raise RuntimeError("Call fit() before estimate().")

        synth_array = self.feature_processor.transform(synthetic_dataframe)

        # Synthetic → real (training half)
        dcr_synth, _ = self._knn_train.kneighbors(synth_array, n_neighbors=1)
        dcr_synth = dcr_synth.flatten()

        # Holdout real → real (training half)
        dcr_holdout, _ = self._knn_holdout.kneighbors(
            self._holdout_data, n_neighbors=1
        )
        dcr_holdout = dcr_holdout.flatten()

        median_synth = float(np.median(dcr_synth))
        median_holdout = float(np.median(dcr_holdout))
        ratio = median_synth / median_holdout if median_holdout > 0 else np.inf

        threshold = float(np.percentile(dcr_holdout, self.par_percentile))
        at_risk_mask = dcr_synth < threshold
        par = float(at_risk_mask.mean())
        at_risk_indices = np.where(at_risk_mask)[0]

        logger.info(
            f"DCR — median synth: {median_synth:.4f}, median holdout: {median_holdout:.4f}, "
            f"ratio: {ratio:.3f}, privacy-at-risk: {par:.2%}"
        )

        return DCRResults(
            dcr_synthetic=dcr_synth,
            dcr_real_holdout=dcr_holdout,
            median_dcr_synthetic=median_synth,
            median_dcr_holdout=median_holdout,
            median_dcr_ratio=ratio,
            privacy_at_risk=par,
            at_risk_indices=at_risk_indices,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        results: DCRResults,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """KDE and CDF comparison of synthetic vs real holdout DCR distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # --- KDE ---
        ax = axes[0]
        clip_val = float(np.percentile(results.dcr_real_holdout, 99))
        synth_clipped = np.clip(results.dcr_synthetic, 0, clip_val)
        holdout_clipped = np.clip(results.dcr_real_holdout, 0, clip_val)

        from scipy.stats import gaussian_kde

        xs = np.linspace(0, clip_val, 300)
        if synth_clipped.std() > 0:
            ax.plot(xs, gaussian_kde(synth_clipped)(xs), label="Synthetic DCR", color="steelblue")
        if holdout_clipped.std() > 0:
            ax.plot(xs, gaussian_kde(holdout_clipped)(xs), label="Real holdout DCR", color="darkorange")
        threshold = float(np.percentile(results.dcr_real_holdout, self.par_percentile))
        ax.axvline(threshold, color="red", linestyle="--", label=f"PAR threshold ({self.par_percentile}th pct)")
        ax.set_title("DCR Distribution (KDE)")
        ax.set_xlabel("Distance to closest real record")
        ax.legend(fontsize=8)

        # --- CDF ---
        ax = axes[1]
        for arr, label, color in [
            (results.dcr_synthetic, "Synthetic DCR", "steelblue"),
            (results.dcr_real_holdout, "Real holdout DCR", "darkorange"),
        ]:
            sorted_arr = np.sort(arr)
            cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
            ax.plot(sorted_arr, cdf, label=label, color=color)
        ax.set_title("DCR Cumulative Distribution")
        ax.set_xlabel("DCR value")
        ax.set_ylabel("Cumulative probability")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"Distance to Closest Record  |  ratio={results.median_dcr_ratio:.3f}  |  "
            f"PAR={results.privacy_at_risk:.2%}",
            fontsize=11,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

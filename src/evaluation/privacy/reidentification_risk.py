"""Re-identification Risk — linkage-based attack on synthetic tabular data.

Attack framing
--------------
Given D_real (real records used for training) and D_syn (synthetic data
released as a privacy-safe substitute), an attacker who knows a real
individual's quasi-identifiers searches D_syn for the nearest matching
synthetic record.  If the generator memorised that individual, a near-exact
copy will appear in D_syn.

This is the correct privacy threat model for a synthetic-data release:
membership is 100% by definition (every record in D_real was used for
training), so the question is not *whether* a person was in the training set
but *whether an attacker can find their record in the synthetic output and
read off their sensitive attributes*.

Attack mechanics (composition of linkage + attribute inference)
---------------------------------------------------------------
1. **Linkage** — for each real record, find its 1st and 2nd nearest
   neighbours in D_syn (real → synthetic search direction).
2. **Gap scoring** — ``gap = d(2nd-NN_synth) / d(1st-NN_synth)``.
   A large gap means one synthetic record unambiguously matches this real
   individual and nothing else is close — the attacker can confidently link
   the person to a synthetic row and read off their sensitive columns.
   A gap near 1 means many synthetic records are equidistant — no confident
   link is possible.
3. **Attribution** — with a high-confidence link, the attacker reads
   sensitive columns from the matched synthetic row.  This step is implicit
   here; the per-record gap scores flag which individuals are most exposed.

The per-record gap scores are the primary output of a privacy audit: they
identify real records that the generator memorised too faithfully — outliers,
people with rare attribute combinations, and edge cases the model latched
onto.

Key metrics
-----------
- ``mean_gap_ratio``        : mean gap across all real records
- ``median_gap_ratio``      : median gap
- ``reidentification_rate`` : fraction of real records with gap > ``risk_threshold``
                              (default 2.0 — the 2nd nearest synthetic record is
                              at least twice as far as the nearest one, indicating
                              an unambiguous match)
- ``num_at_risk``           : count of high-risk real records
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict
from loguru import logger

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler
from src.evaluation.privacy.gower_distance import GowerDistanceCalculator


class ReidentificationResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Per-record arrays (indexed over real records)
    gap_ratios: np.ndarray                      # d(2nd-NN_synth) / d(1st-NN_synth) per real record
    nearest_synth_distances: np.ndarray         # d(real, 1st-NN_synth)
    second_nearest_synth_distances: np.ndarray  # d(real, 2nd-NN_synth)

    # Scalar summaries
    mean_gap_ratio: float
    median_gap_ratio: float
    reidentification_rate: float                # fraction with gap > risk_threshold
    num_at_risk: int

    def summary(self) -> dict:
        return {
            "mean_gap_ratio": self.mean_gap_ratio,
            "median_gap_ratio": self.median_gap_ratio,
            "reidentification_rate": self.reidentification_rate,
            "num_at_risk": self.num_at_risk,
        }


class ReidentificationRiskEstimator:
    """Estimate per-record re-identification risk via D_syn linkage.

    For each real record, the attacker searches D_syn for the nearest synthetic
    match.  The gap ratio ``d(2nd-NN_synth) / d(1st-NN_synth)`` measures how
    unambiguous that match is: a large gap means a single synthetic record
    stands out as a clear copy of the real individual.

    Parameters
    ----------
    scaler :
        Any sklearn scaler.  Defaults to RobustScaler.
    categorical_columns :
        Column names that should be one-hot encoded rather than scaled.
    risk_threshold :
        Gap-ratio threshold above which a real record is considered at risk.
        ``gap = d(2nd-NN_synth) / d(1st-NN_synth)``.  Default 2.0 — the 2nd
        nearest synthetic record is at least twice as far as the closest one,
        indicating an unambiguous synthetic match.
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
        risk_threshold: float = 2.0,
        algorithm: str = "ball_tree",
        distance_metric: str = "euclidean",
        fitted_feature_processor: Optional["FeatureProcessor"] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        self.distance_metric = distance_metric
        self.categorical_columns = categorical_columns or []
        self.risk_threshold = risk_threshold
        self.algorithm = algorithm

        if distance_metric == "gower":
            self._gower = GowerDistanceCalculator(
                categorical_columns=self.categorical_columns,
                feature_groups=feature_groups,
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

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, real_dataframe: pl.DataFrame) -> "ReidentificationRiskEstimator":
        """Store and preprocess the real dataset (query set for the attack)."""
        self._real_df = real_dataframe

        if self.distance_metric == "gower":
            self._gower.fit(real_dataframe)
        else:
            if self.feature_processor.fitted:
                self._real_data = self.feature_processor.transform(real_dataframe)
            else:
                self._real_data = self.feature_processor.fit_transform(real_dataframe)

        logger.info(
            f"ReidentificationRiskEstimator fitted ({self.distance_metric}): "
            f"{len(real_dataframe)} real records."
        )
        return self

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(self, synthetic_dataframe: pl.DataFrame) -> ReidentificationResults:
        """Compute per-record re-identification risk for every real record.

        For each real record, finds its two nearest neighbours in D_syn
        (real → synth direction) and computes
        ``gap = d(2nd-NN_synth) / d(1st-NN_synth)``.  A large gap signals an
        unambiguous synthetic match — high re-identification risk.

        Edge cases:
        - ``d1 = 0`` (perfect copy in D_syn) → gap capped at 100 → maximum risk
        - ``d1 = 0`` and ``d2 = 0`` → gap set to 1.0 → treated as low risk
          (the whole synthetic distribution collapsed here)
        """
        if self.distance_metric == "gower":
            if not self._gower.fitted:
                raise RuntimeError("Call fit() before estimate().")

            distances, _ = self._gower.nearest_neighbor_distances(
                df_query=self._real_df,
                df_reference=synthetic_dataframe,
                n_neighbors=2,
            )
        else:
            if self._real_data is None:
                raise RuntimeError("Call fit() before estimate().")

            synth_array = self.feature_processor.transform(synthetic_dataframe)
            knn = NearestNeighbors(n_neighbors=2, algorithm=self.algorithm)
            knn.fit(synth_array)
            distances, _ = knn.kneighbors(self._real_data, n_neighbors=2)

        nearest = distances[:, 0]
        second_nearest = distances[:, 1]

        # gap = d2 / d1 — large gap means unambiguous match (high risk)
        with np.errstate(divide="ignore", invalid="ignore"):
            gaps = np.where(
                nearest > 0,
                second_nearest / nearest,
                np.where(second_nearest > 0, 100.0, 1.0),
            )

        at_risk_mask = gaps > self.risk_threshold
        reid_rate = float(at_risk_mask.mean())
        mean_gap = float(np.mean(gaps))
        median_gap = float(np.median(gaps))

        logger.info(
            f"Re-identification risk — mean gap: {mean_gap:.4f}, "
            f"median gap: {median_gap:.4f}, "
            f"re-identification rate (gap > {self.risk_threshold}): {reid_rate:.2%}, "
            f"num at risk: {int(at_risk_mask.sum())}"
        )

        return ReidentificationResults(
            gap_ratios=gaps,
            nearest_synth_distances=nearest,
            second_nearest_synth_distances=second_nearest,
            mean_gap_ratio=mean_gap,
            median_gap_ratio=median_gap,
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
        """Gap-ratio histogram and nearest-vs-second-nearest synthetic distance scatter."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # --- Gap ratio histogram ---
        ax = axes[0]
        ax.hist(results.gap_ratios, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(
            self.risk_threshold, color="red", linestyle="--",
            label=f"Threshold ({self.risk_threshold})",
        )
        ax.set_xlabel("Gap ratio  d(2nd-NN_synth) / d(1st-NN_synth)")
        ax.set_ylabel("Count (real records)")
        ax.set_title("Re-identification Risk — Gap Ratios")
        ax.legend(fontsize=8)

        # --- Scatter: nearest vs second-nearest synth distance ---
        ax = axes[1]
        at_risk = results.gap_ratios > self.risk_threshold
        ax.scatter(
            results.nearest_synth_distances[~at_risk],
            results.second_nearest_synth_distances[~at_risk],
            alpha=0.3, s=8, color="steelblue", label="Low risk",
        )
        ax.scatter(
            results.nearest_synth_distances[at_risk],
            results.second_nearest_synth_distances[at_risk],
            alpha=0.5, s=12, color="red", label="At risk",
        )
        # Threshold line: gap = threshold → d2 = threshold * d1
        max_val = max(
            float(results.nearest_synth_distances.max()),
            float(results.second_nearest_synth_distances.max()),
        )
        xs = np.linspace(0, max_val, 100)
        ax.plot(xs, self.risk_threshold * xs, "r--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Distance to 1st-NN synthetic")
        ax.set_ylabel("Distance to 2nd-NN synthetic")
        ax.set_title("Real → Synthetic: Nearest vs. Second-Nearest Distance")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"Re-identification Risk  |  rate={results.reidentification_rate:.2%}  |  "
            f"median gap={results.median_gap_ratio:.3f}  |  "
            f"at-risk={results.num_at_risk}",
            fontsize=11,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

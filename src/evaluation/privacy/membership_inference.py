"""Membership Inference Attack (MIA) privacy metric.

Approach
--------
For each candidate record (member = in training set, non-member = held-out real),
we compute an *attack signal* and train a logistic regression classifier to
distinguish the two groups.

Attack signals
~~~~~~~~~~~~~~
``dcr``
    Distance from the candidate record to its nearest synthetic neighbour.
    Intuition: the model tends to generate synthetic points close to records it
    memorised, so members have a *smaller* distance to their nearest synthetic
    neighbour than non-members.

``likelihood``
    Not applicable to all model types.  When the generative model exposes a
    ``log_prob`` method (e.g. Gaussian copula), we use it.  Otherwise this
    signal is skipped automatically.

``both``
    Concatenate both signals (when both are available).

Key metrics
-----------
- ``auc``          : AUC-ROC of the attack classifier  (0.5 = perfect privacy)
- ``advantage``    : 2 * (AUC - 0.5), range [0, 1]
- ``tpr_at_fpr``   : TPR at FPR = 0.01 (worst-case attacker precision)
- ROC curve data for plotting
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import RobustScaler
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict
from loguru import logger

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler


class MIAResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    auc: float
    advantage: float                    # 2 * (AUC - 0.5)
    tpr_at_low_fpr: float               # TPR at FPR = 0.01
    fpr_curve: np.ndarray
    tpr_curve: np.ndarray
    thresholds: np.ndarray
    attack_scores_members: np.ndarray   # per-record risk scores for members
    attack_scores_nonmembers: np.ndarray

    def summary(self) -> dict:
        return {
            "mia_auc": self.auc,
            "mia_advantage": self.advantage,
            "mia_tpr_at_fpr_001": self.tpr_at_low_fpr,
        }


class MembershipInferenceAttack:
    """Assess whether a generative model leaks membership information.

    Parameters
    ----------
    scaler :
        Scaler used by the internal FeatureProcessor.
    categorical_columns :
        Columns to one-hot encode.
    holdout_fraction :
        Fraction of real data withheld as *non-members*.  Default 0.2.
    attack_signal :
        ``"dcr"`` | ``"likelihood"`` | ``"both"``.  When the generative model
        does not expose ``log_prob``, ``"likelihood"`` falls back to ``"dcr"``.
    algorithm :
        NearestNeighbors algorithm for DCR signal computation.
    fitted_feature_processor :
        A pre-fitted FeatureProcessor to reuse.  When provided, ``scaler`` and
        ``categorical_columns`` are ignored.  Pass this from PrivacyReport to
        share a single processor across all privacy estimators.
    """

    def __init__(
        self,
        scaler: Optional[Scaler] = None,
        categorical_columns: Optional[List[str]] = None,
        holdout_fraction: float = 0.2,
        attack_signal: Literal["dcr", "likelihood", "both"] = "dcr",
        algorithm: str = "ball_tree",
        fitted_feature_processor: Optional["FeatureProcessor"] = None,
    ):
        if fitted_feature_processor is not None:
            self.feature_processor = fitted_feature_processor
        else:
            self.feature_processor = FeatureProcessor(
                scaler=scaler if scaler is not None else RobustScaler(),
                categorical_columns=categorical_columns,
            )
        self.holdout_fraction = holdout_fraction
        self.attack_signal = attack_signal
        self.algorithm = algorithm

        self._members_array: Optional[np.ndarray] = None
        self._nonmembers_array: Optional[np.ndarray] = None
        self._members_df: Optional[pl.DataFrame] = None
        self._nonmembers_df: Optional[pl.DataFrame] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, real_dataframe: pl.DataFrame) -> "MembershipInferenceAttack":
        """Split real data into members (training) and non-members (holdout)."""
        n = len(real_dataframe)
        n_holdout = max(1, int(n * self.holdout_fraction))
        idx = np.random.permutation(n)
        member_idx = idx[n_holdout:]
        nonmember_idx = idx[:n_holdout]

        self._members_df = real_dataframe[member_idx.tolist()]
        self._nonmembers_df = real_dataframe[nonmember_idx.tolist()]

        self._members_array = self.feature_processor.fit_transform(self._members_df)
        self._nonmembers_array = self.feature_processor.transform(self._nonmembers_df)

        logger.info(
            f"MIA fitted: {len(self._members_df)} members, "
            f"{len(self._nonmembers_df)} non-members."
        )
        return self

    # ------------------------------------------------------------------
    # Attack signal computation
    # ------------------------------------------------------------------

    def _dcr_signal(self, synthetic_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (member_scores, nonmember_scores) based on DCR to synthetic."""
        knn = NearestNeighbors(n_neighbors=1, algorithm=self.algorithm)
        knn.fit(synthetic_array)

        member_dcr, _ = knn.kneighbors(self._members_array, n_neighbors=1)
        nonmember_dcr, _ = knn.kneighbors(self._nonmembers_array, n_neighbors=1)

        # Negate: lower DCR = more likely to be a member = higher attack score
        return -member_dcr.flatten(), -nonmember_dcr.flatten()

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        synthetic_dataframe: pl.DataFrame,
        generative_model=None,
    ) -> MIAResults:
        """Run the attack.

        Parameters
        ----------
        synthetic_dataframe :
            Synthetic data generated by the model under evaluation.
        generative_model :
            Optional — if supplied and has a ``log_prob(df) -> np.ndarray``
            method, it is used for the ``likelihood`` signal.
        """
        if self._members_array is None:
            raise RuntimeError("Call fit() before estimate().")

        synth_array = self.feature_processor.transform(synthetic_dataframe)

        # --- Compute attack signals ---
        use_likelihood = (
            self.attack_signal in ("likelihood", "both")
            and generative_model is not None
            and hasattr(generative_model, "log_prob")
        )

        member_signals_list = []
        nonmember_signals_list = []

        if self.attack_signal in ("dcr", "both") or not use_likelihood:
            m_dcr, nm_dcr = self._dcr_signal(synth_array)
            member_signals_list.append(m_dcr.reshape(-1, 1))
            nonmember_signals_list.append(nm_dcr.reshape(-1, 1))

        if use_likelihood:
            m_ll = generative_model.log_prob(self._members_df).reshape(-1, 1)
            nm_ll = generative_model.log_prob(self._nonmembers_df).reshape(-1, 1)
            member_signals_list.append(m_ll)
            nonmember_signals_list.append(nm_ll)

        member_features = np.concatenate(member_signals_list, axis=1)
        nonmember_features = np.concatenate(nonmember_signals_list, axis=1)

        X = np.concatenate([member_features, nonmember_features], axis=0)
        y = np.concatenate(
            [np.ones(len(member_features)), np.zeros(len(nonmember_features))]
        )

        # --- Train attack classifier ---
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        clf.fit(X, y)
        scores = clf.predict_proba(X)[:, 1]

        attack_scores_members = scores[: len(member_features)]
        attack_scores_nonmembers = scores[len(member_features):]

        auc = float(roc_auc_score(y, scores))
        advantage = float(2 * (auc - 0.5))
        fpr_curve, tpr_curve, thresholds = roc_curve(y, scores)

        # TPR at FPR ≤ 0.01
        tpr_at_low_fpr = float(
            tpr_curve[np.searchsorted(fpr_curve, 0.01, side="right") - 1]
        )

        logger.info(
            f"MIA — AUC: {auc:.4f}, advantage: {advantage:.4f}, "
            f"TPR@FPR=0.01: {tpr_at_low_fpr:.4f}"
        )

        return MIAResults(
            auc=auc,
            advantage=advantage,
            tpr_at_low_fpr=tpr_at_low_fpr,
            fpr_curve=fpr_curve,
            tpr_curve=tpr_curve,
            thresholds=thresholds,
            attack_scores_members=attack_scores_members,
            attack_scores_nonmembers=attack_scores_nonmembers,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        results: MIAResults,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """ROC curve + attack score histogram."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # --- ROC ---
        ax = axes[0]
        ax.plot(results.fpr_curve, results.tpr_curve, color="steelblue",
                label=f"ROC (AUC={results.auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("MIA — ROC Curve")
        ax.legend(fontsize=8)

        # --- Score histogram ---
        ax = axes[1]
        ax.hist(results.attack_scores_members, bins=30, alpha=0.6,
                label="Members", color="steelblue")
        ax.hist(results.attack_scores_nonmembers, bins=30, alpha=0.6,
                label="Non-members", color="darkorange")
        ax.set_xlabel("Attack score (P(member))")
        ax.set_title("MIA — Attack Score Distribution")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"Membership Inference Attack  |  AUC={results.auc:.3f}  |  "
            f"Advantage={results.advantage:.3f}",
            fontsize=11,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

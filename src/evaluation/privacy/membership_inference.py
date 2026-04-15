"""Membership Inference Attack (MIA) privacy metric.

Approach
--------
For each candidate record (member = in training set, non-member = held-out real),
we compute attack signals and train a logistic regression classifier to
distinguish the two groups.  We then also run two simpler threshold-based
attacks and report the **worst-case AUC** across all strategies.

Attack signals
~~~~~~~~~~~~~~
``dcr``
    Multi-k DCR features: distances from the candidate record to its 1st, 3rd,
    and 5th nearest synthetic neighbours.  Normalised by the median
    synthetic-to-synthetic distance so the signal is density-invariant.
    Intuition: the model tends to generate synthetic points close to records it
    memorised, so members have a *smaller* normalised DCR than non-members.

``likelihood``
    Not applicable to all model types.  When the generative model exposes a
    ``log_prob`` method (e.g. Gaussian copula), we use it.  Otherwise this
    signal is skipped automatically.

``both``
    Concatenate both signals (when both are available).

Key metrics
-----------
- ``auc``                : worst-case AUC-ROC across all attack strategies (0.5 = perfect privacy)
- ``advantage``          : 2 * (AUC - 0.5), range [0, 1]
- ``tpr_at_fpr``         : TPR at FPR = 0.01 (worst-case attacker precision)
- ``fraction_high_risk`` : fraction of members with attack score > 0.9
- ROC curve data for plotting
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict
from loguru import logger

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler


class MIAResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    auc: float
    advantage: float  # 2 * (AUC - 0.5)
    tpr_at_low_fpr: float  # TPR at FPR = 0.01
    fraction_high_risk: float  # fraction of members with score > 0.9
    fpr_curve: np.ndarray
    tpr_curve: np.ndarray
    thresholds: np.ndarray
    attack_scores_members: np.ndarray  # per-record risk scores for members
    attack_scores_nonmembers: np.ndarray

    def summary(self) -> dict:
        return {
            "mia_auc": self.auc,
            "mia_advantage": self.advantage,
            "mia_tpr_at_fpr_001": self.tpr_at_low_fpr,
            "mia_fraction_high_risk": self.fraction_high_risk,
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
    classifier :
        Attack classifier: ``"logistic_regression"`` | ``"gradient_boosting"`` |
        ``"random_forest"``.  Default ``"logistic_regression"``.
        All classifiers are probability-calibrated with isotonic regression via
        3-fold ``CalibratedClassifierCV`` and use balanced class weights where
        supported.
    algorithm :
        NearestNeighbors algorithm for DCR signal computation.
    n_folds :
        Number of folds for stratified k-fold cross-validation of the attack
        classifier.  Default 5.  Using CV prevents the classifier from
        overfitting to the attack dataset and gives a more honest AUC estimate.
    fitted_feature_processor :
        A pre-fitted FeatureProcessor to reuse.  When provided, ``scaler`` and
        ``categorical_columns`` are ignored.  Pass this from PrivacyReport to
        share a single processor across all privacy estimators.
    high_risk_threshold :
        Attack score above which a member is considered high-risk.  Default 0.9.
        Used to compute ``fraction_high_risk`` in the summary.
    """

    # k values used for multi-k DCR features
    _KS = (1, 3, 5)

    def __init__(
        self,
        scaler: Optional[Scaler] = None,
        categorical_columns: Optional[List[str]] = None,
        holdout_fraction: float = 0.2,
        attack_signal: Literal["dcr", "likelihood", "both"] = "dcr",
        classifier: Literal[
            "logistic_regression", "gradient_boosting", "random_forest"
        ] = "logistic_regression",
        algorithm: str = "ball_tree",
        n_folds: int = 5,
        fitted_feature_processor: Optional["FeatureProcessor"] = None,
        high_risk_threshold: float = 0.9,
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
        self.classifier = classifier
        self.algorithm = algorithm
        self.n_folds = n_folds
        self.high_risk_threshold = high_risk_threshold

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
        """Return (member_features, nonmember_features) as multi-k normalised DCR.

        Features per record: [dcr_k1, dcr_k3, dcr_k5] each normalised by the
        median synthetic-to-synthetic distance, making the signal invariant to
        the overall density of the synthetic distribution.
        """
        max_k = max(self._KS)
        # Need enough synthetic points for k-NN
        k_query = min(max_k, len(synthetic_array) - 1)
        ks_used = [k for k in self._KS if k <= k_query]

        knn = NearestNeighbors(n_neighbors=k_query, algorithm=self.algorithm)
        knn.fit(synthetic_array)

        member_dists, _ = knn.kneighbors(self._members_array, n_neighbors=k_query)
        nonmember_dists, _ = knn.kneighbors(self._nonmembers_array, n_neighbors=k_query)

        # Compute synthetic self-distances for normalisation
        synth_self_dists, _ = knn.kneighbors(
            synthetic_array, n_neighbors=min(2, k_query)
        )
        # Use 1st neighbour (index 0 is itself when querying the same array, so take index 1 if available)
        synth_nn_col = 1 if synth_self_dists.shape[1] > 1 else 0
        synth_median = float(np.median(synth_self_dists[:, synth_nn_col]))
        # Avoid division by zero
        norm = synth_median if synth_median > 0 else 1.0

        # Extract features at each k (k-th neighbour is at index k-1)
        k_indices = [k - 1 for k in ks_used]
        member_feats = (
            -member_dists[:, k_indices] / norm
        )  # negate: lower dist = higher risk
        nonmember_feats = -nonmember_dists[:, k_indices] / norm

        return member_feats, nonmember_feats

    # ------------------------------------------------------------------
    # Attack strategies
    # ------------------------------------------------------------------

    def _build_base_classifier(self):
        """Instantiate the configured attack classifier."""
        if self.classifier == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        if self.classifier == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            )
        # default: logistic_regression
        return LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
        )

    def _run_classifier_attack(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Attack classifier (configurable) with probability calibration and k-fold CV."""
        scores = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in skf.split(X, y):
            base_clf = self._build_base_classifier()
            clf = CalibratedClassifierCV(base_clf, cv=3, method="isotonic")
            clf.fit(X[train_idx], y[train_idx])
            scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

        auc = float(roc_auc_score(y, scores))
        return scores, auc

    def _run_threshold_attack(
        self,
        member_signal: np.ndarray,
        nonmember_signal: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Direct threshold on a 1-D signal (e.g. raw DCR_k1). No classifier."""
        y = np.concatenate(
            [np.ones(len(member_signal)), np.zeros(len(nonmember_signal))]
        )
        scores = np.concatenate([member_signal, nonmember_signal])
        auc = float(roc_auc_score(y, scores))
        return scores, auc

    def _run_ratio_attack(
        self,
        synthetic_array: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Ratio signal: DCR(real→synth) / DCR(synth→synth).

        This is already baked into the normalised DCR features but we also
        run it as a standalone threshold attack on k=1 only for robustness.
        """
        knn = NearestNeighbors(n_neighbors=1, algorithm=self.algorithm)
        knn.fit(synthetic_array)

        member_dcr, _ = knn.kneighbors(self._members_array, n_neighbors=1)
        nonmember_dcr, _ = knn.kneighbors(self._nonmembers_array, n_neighbors=1)

        synth_self, _ = knn.kneighbors(
            synthetic_array, n_neighbors=min(2, len(synthetic_array) - 1)
        )
        synth_nn_col = 1 if synth_self.shape[1] > 1 else 0
        synth_median = float(np.median(synth_self[:, synth_nn_col]))
        norm = synth_median if synth_median > 0 else 1.0

        member_ratio = -(member_dcr.flatten() / norm)
        nonmember_ratio = -(nonmember_dcr.flatten() / norm)

        return self._run_threshold_attack(member_ratio, nonmember_ratio)

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        synthetic_dataframe: pl.DataFrame,
        generative_model=None,
    ) -> MIAResults:
        """Run all attack strategies and return the worst-case result.

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

        # --- Compute DCR features ---
        member_dcr_feats, nonmember_dcr_feats = self._dcr_signal(synth_array)

        use_likelihood = (
            self.attack_signal in ("likelihood", "both")
            and generative_model is not None
            and hasattr(generative_model, "log_prob")
        )

        member_signals_list = [member_dcr_feats]
        nonmember_signals_list = [nonmember_dcr_feats]

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

        # --- Run all attack strategies ---
        results_by_strategy: dict[str, tuple[np.ndarray, float]] = {}

        # 1. Classifier attack (multi-k normalised DCR + optional likelihood)
        clf_scores, clf_auc = self._run_classifier_attack(X, y)
        results_by_strategy["classifier"] = (clf_scores, clf_auc)

        # 2. Threshold attack on raw k=1 DCR signal
        m_k1 = member_dcr_feats[:, 0]
        nm_k1 = nonmember_dcr_feats[:, 0]
        thr_scores, thr_auc = self._run_threshold_attack(m_k1, nm_k1)
        results_by_strategy["threshold_k1"] = (
            np.concatenate([thr_scores[: len(m_k1)], thr_scores[len(m_k1) :]]),
            thr_auc,
        )

        # 3. Ratio attack
        ratio_scores, ratio_auc = self._run_ratio_attack(synth_array)
        results_by_strategy["ratio"] = (ratio_scores, ratio_auc)

        # --- Worst-case strategy ---
        worst_name = max(results_by_strategy, key=lambda k: results_by_strategy[k][1])
        best_scores, auc = results_by_strategy[worst_name]

        logger.info(
            f"MIA strategy AUCs [{self.classifier}] — "
            + ", ".join(f"{k}: {v[1]:.4f}" for k, v in results_by_strategy.items())
            + f" → worst-case: {worst_name} ({auc:.4f})"
        )

        # Re-derive member/nonmember split from the worst-case scores
        # Classifier scores are already split correctly; threshold/ratio scores
        # are concatenated [members | nonmembers] in the same order as y
        n_members = len(member_features)
        attack_scores_members = best_scores[:n_members]
        attack_scores_nonmembers = best_scores[n_members:]

        advantage = float(2 * (auc - 0.5))

        y_worst = np.concatenate(
            [np.ones(n_members), np.zeros(len(nonmember_features))]
        )
        fpr_curve, tpr_curve, thresholds = roc_curve(y_worst, best_scores)

        tpr_at_low_fpr = float(
            tpr_curve[np.searchsorted(fpr_curve, 0.01, side="right") - 1]
        )

        fraction_high_risk = float((attack_scores_members > self.high_risk_threshold).mean())

        logger.info(
            f"MIA worst-case — AUC: {auc:.4f}, advantage: {advantage:.4f}, "
            f"TPR@FPR=0.01: {tpr_at_low_fpr:.4f}, "
            f"high-risk fraction: {fraction_high_risk:.4f}"
        )

        return MIAResults(
            auc=auc,
            advantage=advantage,
            tpr_at_low_fpr=tpr_at_low_fpr,
            fraction_high_risk=fraction_high_risk,
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
        ax.plot(
            results.fpr_curve,
            results.tpr_curve,
            color="steelblue",
            label=f"ROC (AUC={results.auc:.3f})",
        )
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("MIA — ROC Curve (worst-case)")
        ax.legend(fontsize=8)

        # --- Score histogram ---
        ax = axes[1]
        ax.hist(
            results.attack_scores_members,
            bins=30,
            alpha=0.6,
            label="Members",
            color="steelblue",
        )
        ax.hist(
            results.attack_scores_nonmembers,
            bins=30,
            alpha=0.6,
            label="Non-members",
            color="darkorange",
        )
        ax.set_xlabel("Attack score (P(member))")
        ax.set_title("MIA — Attack Score Distribution")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"Membership Inference Attack  |  AUC={results.auc:.3f}  |  "
            f"Advantage={results.advantage:.3f}  |  "
            f"High-risk={results.fraction_high_risk:.1%}",
            fontsize=11,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

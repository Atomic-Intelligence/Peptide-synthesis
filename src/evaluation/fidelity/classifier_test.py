"""Two-sample classifier test for evaluating synthetic data fidelity.

Trains a binary classifier to distinguish real from synthetic samples using
stratified k-fold cross-validation and reports the mean AUC-ROC.

Interpretation
--------------
AUC ≈ 0.5  →  classifier cannot distinguish real from synthetic  →  high fidelity
AUC ≈ 1.0  →  classifier easily separates the distributions      →  low fidelity
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler
from src.evaluation.utils.eval_utils import sparse_peptide_columns


@dataclass
class TwoSampleClassifierResults:
    """Results from the two-sample classifier test.

    Attributes
    ----------
    auc_mean :
        Mean AUC-ROC across cross-validation folds.
    auc_std :
        Standard deviation of AUC-ROC across folds.
    auc_per_fold :
        Per-fold AUC-ROC scores.
    feature_importances :
        Mean feature importances across folds (random forest only; None otherwise).
    feature_names :
        Feature names corresponding to ``feature_importances``.
    classifier_type :
        Which classifier was used.
    n_folds :
        Number of cross-validation folds.
    """

    auc_mean: float = 0.0
    auc_std: float = 0.0
    auc_per_fold: List[float] = field(default_factory=list)
    feature_importances: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    classifier_type: str = "random_forest"
    n_folds: int = 5

    def summary(self) -> Dict[str, float]:
        """Flat dict of headline metrics for mlflow.log_metrics()."""
        return {
            "fidelity/two_sample_auc": self.auc_mean,
            "fidelity/two_sample_auc_std": self.auc_std,
        }


def _make_classifier(classifier_type: str):
    """Instantiate the requested classifier."""
    if classifier_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            n_jobs=-1,
            random_state=42,
        )
    elif classifier_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unknown classifier type '{classifier_type}'. "
            "Choose 'random_forest' or 'logistic_regression'."
        )


class TwoSampleClassifierTest:
    """Two-sample classifier test for evaluating synthetic data fidelity.

    Labels real samples as 0 and synthetic samples as 1, then trains a
    classifier via stratified k-fold cross-validation and reports the mean
    AUC-ROC.

    Parameters
    ----------
    scaler :
        Scaler applied to numerical features before classification.
        Defaults to RobustScaler.
    categorical_columns :
        Columns to one-hot encode instead of scale.
    classifier :
        Which classifier to use — ``"random_forest"`` (default) or
        ``"logistic_regression"``.
    n_folds :
        Number of stratified k-fold cross-validation folds.  Default 5.
    peptide_zero_threshold :
        If provided, peptide columns with a zero fraction above this threshold
        in the real data are dropped before the test (mirrors the joint fidelity
        module behaviour).
    """

    def __init__(
        self,
        scaler: Optional[Scaler] = None,
        categorical_columns: Optional[List[str]] = None,
        classifier: str = "random_forest",
        n_folds: int = 5,
        peptide_zero_threshold: Optional[float] = None,
    ):
        self.classifier_type = classifier
        self.n_folds = n_folds
        self.peptide_zero_threshold = peptide_zero_threshold
        self.feature_processor = FeatureProcessor(
            scaler=scaler if scaler is not None else RobustScaler(),
            categorical_columns=categorical_columns,
        )

    def estimate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
    ) -> TwoSampleClassifierResults:
        """Run the two-sample classifier test.

        Parameters
        ----------
        real_df :
            Real (reference) dataset.
        synth_df :
            Synthetic dataset to evaluate.

        Returns
        -------
        TwoSampleClassifierResults
        """
        logger.info(
            f"TwoSampleClassifierTest — classifier={self.classifier_type}, "
            f"n_folds={self.n_folds}"
        )

        # Optionally drop sparse peptide columns (same logic as JointFidelity)
        if self.peptide_zero_threshold is not None:
            drop_cols = sparse_peptide_columns(real_df, self.peptide_zero_threshold)
            if drop_cols:
                real_df = real_df.drop(drop_cols)
                synth_df = synth_df.drop(drop_cols)

        # Only keep columns present in both frames
        shared_cols = [c for c in real_df.columns if c in synth_df.columns]
        real_df = real_df.select(shared_cols)
        synth_df = synth_df.select(shared_cols)

        # Encode features: fit on real data, transform both
        real_arr = self.feature_processor.fit_transform(real_df)
        synth_arr = self.feature_processor.transform(synth_df)
        feature_names = list(self.feature_processor.feature_names)

        # Stack real (label=0) and synthetic (label=1)
        X = np.concatenate([real_arr, synth_arr], axis=0)
        y = np.concatenate([
            np.zeros(len(real_arr), dtype=int),
            np.ones(len(synth_arr), dtype=int),
        ])

        # Stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        auc_scores: List[float] = []
        fold_importances: List[np.ndarray] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = _make_classifier(self.classifier_type)
            clf.fit(X[train_idx], y[train_idx])

            proba = clf.predict_proba(X[val_idx])[:, 1]
            auc = float(roc_auc_score(y[val_idx], proba))
            auc_scores.append(auc)
            logger.debug(f"  Fold {fold_idx + 1}/{self.n_folds} — AUC: {auc:.4f}")

            if hasattr(clf, "feature_importances_"):
                fold_importances.append(clf.feature_importances_)

        auc_mean = float(np.mean(auc_scores))
        auc_std = float(np.std(auc_scores))

        feature_importances = (
            np.mean(fold_importances, axis=0) if fold_importances else None
        )

        logger.info(
            f"TwoSampleClassifier — AUC: {auc_mean:.4f} ± {auc_std:.4f}  "
            f"(per fold: {[f'{s:.4f}' for s in auc_scores]})"
        )

        return TwoSampleClassifierResults(
            auc_mean=auc_mean,
            auc_std=auc_std,
            auc_per_fold=auc_scores,
            feature_importances=feature_importances,
            feature_names=feature_names,
            classifier_type=self.classifier_type,
            n_folds=self.n_folds,
        )

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(self, results: TwoSampleClassifierResults) -> plt.Figure:
        """Produce a summary figure for the two-sample classifier test.

        The figure has up to two panels:

        1. **AUC per fold** — bar chart of per-fold AUC scores with the mean
           ± std shown.  A dashed reference line at 0.5 marks chance performance.
        2. **Top-20 feature importances** — horizontal bar chart of the most
           discriminative features (random forest only).
        """
        show_importances = (
            results.feature_importances is not None
            and results.feature_names is not None
            and len(results.feature_importances) > 0
        )

        n_panels = 2 if show_importances else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]

        # ── panel 1: per-fold AUC bars ────────────────────────────────────
        ax = axes[0]
        folds = list(range(1, len(results.auc_per_fold) + 1))
        bar_colors = [
            "#d62728" if auc > 0.7 else "#2ca02c" if auc < 0.55 else "#1f77b4"
            for auc in results.auc_per_fold
        ]
        ax.bar(folds, results.auc_per_fold, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, ls="--", lw=1.2, color="grey", label="Chance (AUC = 0.5)")
        ax.axhline(
            results.auc_mean, ls="-", lw=1.5, color="black",
            label=f"Mean AUC = {results.auc_mean:.4f}",
        )
        ax.fill_between(
            [0.5, len(folds) + 0.5],
            results.auc_mean - results.auc_std,
            results.auc_mean + results.auc_std,
            alpha=0.15, color="black",
            label=f"±1 std ({results.auc_std:.4f})",
        )
        ax.set_xlabel("Fold", fontsize=9)
        ax.set_ylabel("AUC-ROC", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(folds)
        ax.set_title(
            f"Two-Sample Classifier Test\n"
            f"({results.classifier_type.replace('_', ' ').title()}, "
            f"{results.n_folds}-fold CV)",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=7)

        # ── panel 2: feature importances ─────────────────────────────────
        if show_importances:
            ax2 = axes[1]
            importances = results.feature_importances
            names = np.array(results.feature_names)
            top_n = min(20, len(importances))
            top_idx = np.argsort(importances)[-top_n:][::-1]
            top_imp = importances[top_idx]
            top_names = names[top_idx]

            y_pos = np.arange(top_n)
            ax2.barh(y_pos, top_imp[::-1], color="#aec7e8", edgecolor="white", linewidth=0.4)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_names[::-1], fontsize=6)
            ax2.set_xlabel("Mean importance", fontsize=9)
            ax2.set_title(
                f"Top-{top_n} Discriminative Features",
                fontsize=10, fontweight="bold",
            )

        plt.tight_layout()
        return fig

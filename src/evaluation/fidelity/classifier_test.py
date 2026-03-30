"""Two-sample classifier test (Train-on-Real+Synth, Classify-Real-vs-Synthetic).

A binary classifier is trained to distinguish real samples (label 0) from
synthetic samples (label 1) using stratified k-fold cross-validation.

If the classifier cannot do better than chance (AUC ≈ 0.5), the synthetic data
is statistically indistinguishable from the real data — a sign of high fidelity.

AUC interpretation guide
-------------------------
0.50 – 0.55  Excellent fidelity
0.55 – 0.65  Good fidelity
0.65 – 0.80  Moderate — inspect feature importances
> 0.80       Poor fidelity — synthetic data is clearly distinguishable

Supported classifiers
----------------------
``"random_forest"``       : sklearn RandomForestClassifier (default)
``"logistic_regression"`` : sklearn LogisticRegression
``"gradient_boosted"``    : sklearn GradientBoostingClassifier
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass, field
from loguru import logger

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler


@dataclass
class TwoSampleClassifierResults:
    fold_aucs: List[float]
    mean_auc: float
    std_auc: float
    mean_accuracy: float
    mean_f1: float
    feature_importances: Dict[str, float]   # sorted descending
    fpr_mean: np.ndarray
    tpr_mean: np.ndarray

    def summary(self) -> Dict[str, float]:
        return {
            "fidelity/two_sample_auc": self.mean_auc,
            "fidelity/two_sample_auc_std": self.std_auc,
            "fidelity/two_sample_accuracy": self.mean_accuracy,
            "fidelity/two_sample_f1": self.mean_f1,
        }

    @property
    def fidelity_grade(self) -> str:
        if self.mean_auc <= 0.55:
            return "Excellent"
        elif self.mean_auc <= 0.65:
            return "Good"
        elif self.mean_auc <= 0.80:
            return "Moderate"
        return "Poor"


def _build_classifier(
    clf_type: str,
    n_estimators: int,
    random_state: int,
):
    if clf_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=1, random_state=random_state
        )
    elif clf_type == "gradient_boosted":
        return GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif clf_type == "logistic_regression":
        return LogisticRegression(max_iter=500, solver="lbfgs", random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier type: {clf_type!r}")


class TwoSampleClassifierTest:
    """Cross-validated two-sample classifier test.

    Parameters
    ----------
    scaler :
        Scaler used by the internal FeatureProcessor.
    categorical_columns :
        Columns to one-hot encode.
    classifier :
        ``"random_forest"`` | ``"logistic_regression"`` | ``"gradient_boosted"``.
    n_folds :
        Number of stratified CV folds.  Default 5.
    n_estimators :
        Number of trees (for tree-based classifiers).
    top_n_features :
        How many features to include in the importance dict / plot.
    random_state :
        Reproducibility seed.
    """

    def __init__(
        self,
        scaler: Optional[Scaler] = None,
        categorical_columns: Optional[List[str]] = None,
        classifier: Literal[
            "random_forest", "logistic_regression", "gradient_boosted"
        ] = "random_forest",
        n_folds: int = 5,
        n_estimators: int = 200,
        top_n_features: int = 20,
        random_state: int = 42,
    ):
        self.feature_processor = FeatureProcessor(
            scaler=scaler if scaler is not None else RobustScaler(),
            categorical_columns=categorical_columns,
        )
        self.classifier_type = classifier
        self.n_folds = n_folds
        self.n_estimators = n_estimators
        self.top_n_features = top_n_features
        self.random_state = random_state

    def estimate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
    ) -> TwoSampleClassifierResults:
        """Run the cross-validated two-sample test.

        Parameters
        ----------
        real_df :
            Real data (label = 0).
        synth_df :
            Synthetic data (label = 1).
        """
        # Align columns
        shared = list(set(real_df.columns) & set(synth_df.columns))
        real_df = real_df.select(shared)
        synth_df = synth_df.select(shared)

        # Fit processor on real data, transform both
        real_arr = self.feature_processor.fit_transform(real_df)
        synth_arr = self.feature_processor.transform(synth_df)

        X = np.concatenate([real_arr, synth_arr], axis=0)
        y = np.concatenate([np.zeros(len(real_arr)), np.ones(len(synth_arr))])

        logger.info(
            f"TwoSampleClassifier: {len(real_arr)} real + {len(synth_arr)} synthetic samples, "
            f"{X.shape[1]} features, {self.n_folds}-fold CV, classifier={self.classifier_type}"
        )

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fold_aucs: List[float] = []
        fold_accs: List[float] = []
        fold_f1s: List[float] = []
        all_importances: List[np.ndarray] = []
        all_fpr: List[np.ndarray] = []
        all_tpr: List[np.ndarray] = []
        common_fpr = np.linspace(0, 1, 200)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = _build_classifier(self.classifier_type, self.n_estimators, self.random_state)
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[val_idx])[:, 1]
            preds = (proba >= 0.5).astype(int)

            auc = float(roc_auc_score(y[val_idx], proba))
            acc = float(accuracy_score(y[val_idx], preds))
            f1 = float(f1_score(y[val_idx], preds, zero_division=0))

            fold_aucs.append(auc)
            fold_accs.append(acc)
            fold_f1s.append(f1)

            fpr, tpr, _ = roc_curve(y[val_idx], proba)
            interp_tpr = np.interp(common_fpr, fpr, tpr)
            all_fpr.append(common_fpr)
            all_tpr.append(interp_tpr)

            if hasattr(clf, "feature_importances_"):
                all_importances.append(clf.feature_importances_)
            elif hasattr(clf, "coef_"):
                all_importances.append(np.abs(clf.coef_[0]))

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        mean_accuracy = float(np.mean(fold_accs))
        mean_f1 = float(np.mean(fold_f1s))
        tpr_mean = np.mean(all_tpr, axis=0)

        # Feature importances
        feat_names = self.feature_processor.feature_names
        if all_importances and feat_names is not None:
            mean_imp = np.mean(all_importances, axis=0)
            top_n = min(self.top_n_features, len(feat_names))
            top_idx = np.argsort(mean_imp)[::-1][:top_n]
            feature_importances = {
                str(feat_names[i]): float(mean_imp[i]) for i in top_idx
            }
        else:
            feature_importances = {}

        logger.info(
            f"TwoSampleClassifier — mean AUC: {mean_auc:.4f} ± {std_auc:.4f}, "
            f"grade: {TwoSampleClassifierResults(fold_aucs=fold_aucs, mean_auc=mean_auc, std_auc=std_auc, mean_accuracy=mean_accuracy, mean_f1=mean_f1, feature_importances=feature_importances, fpr_mean=common_fpr, tpr_mean=tpr_mean).fidelity_grade}"
        )

        return TwoSampleClassifierResults(
            fold_aucs=fold_aucs,
            mean_auc=mean_auc,
            std_auc=std_auc,
            mean_accuracy=mean_accuracy,
            mean_f1=mean_f1,
            feature_importances=feature_importances,
            fpr_mean=common_fpr,
            tpr_mean=tpr_mean,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        results: TwoSampleClassifierResults,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """ROC curve with fold-level shading + top-N feature importance bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # --- ROC ---
        ax = axes[0]
        ax.plot(results.fpr_mean, results.tpr_mean, color="steelblue", linewidth=2,
                label=f"Mean ROC (AUC={results.mean_auc:.3f}±{results.std_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Two-Sample Classifier ROC\nFidelity: {results.fidelity_grade}")
        ax.legend(fontsize=8)

        # --- Feature importances ---
        ax = axes[1]
        if results.feature_importances:
            names = list(results.feature_importances.keys())
            vals = list(results.feature_importances.values())
            y_pos = np.arange(len(names))
            ax.barh(y_pos, vals, color="steelblue", alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Mean importance")
            ax.set_title("Top Feature Importances\n(features most separating real vs synth)")
        else:
            ax.text(0.5, 0.5, "No feature importances available", ha="center", va="center")

        fig.suptitle("Two-Sample Classifier Test", fontsize=11)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

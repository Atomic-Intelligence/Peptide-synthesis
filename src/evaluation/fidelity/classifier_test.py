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
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler
from src.evaluation.utils.eval_utils import (
    get_peptide_columns,
    peptide_columns_in_zero_range,
    sparse_peptide_columns,
)


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


@dataclass
class TwoSampleClassifierResultsByRange:
    """Results from multiple two-sample classifier tests, one per zero-fraction range.

    Attributes
    ----------
    results_by_range :
        Dict mapping range label (e.g. ``"0.0-0.4"``) to its
        ``TwoSampleClassifierResults``.
    """

    results_by_range: Dict[str, TwoSampleClassifierResults] = field(
        default_factory=dict
    )

    def summary(self) -> Dict[str, float]:
        """Flat dict of per-range headline metrics for mlflow.log_metrics()."""
        metrics: Dict[str, float] = {}
        for label, r in self.results_by_range.items():
            safe = label.replace(".", "p").replace("-", "_to_")
            for k, v in r.summary().items():
                metrics[f"{k}_{safe}"] = v
        return metrics


@dataclass
class TwoSampleMultiClassifierResults:
    """Results from running the two-sample test with several discriminators.

    Attributes
    ----------
    results_by_classifier :
        Ordered dict mapping classifier type (e.g. ``"random_forest"``) to its
        result.  Each value is a ``TwoSampleClassifierResults`` (single test) or
        a ``TwoSampleClassifierResultsByRange`` (when zero-fraction ranges are
        configured).
    primary :
        The classifier whose metrics are also emitted under the original,
        unsuffixed keys (``fidelity/two_sample_auc``) for backward compatibility.
        Defaults to the first classifier in ``results_by_classifier``.
    """

    results_by_classifier: Dict[
        str, Union[TwoSampleClassifierResults, TwoSampleClassifierResultsByRange]
    ] = field(default_factory=dict)
    primary: Optional[str] = None

    def summary(self) -> Dict[str, float]:
        """Flat metric dict for mlflow.log_metrics().

        The primary discriminator keeps the original unsuffixed keys.  When more
        than one discriminator ran, every discriminator additionally contributes
        keys suffixed with ``__<classifier_type>``.
        """
        metrics: Dict[str, float] = {}
        if not self.results_by_classifier:
            return metrics

        primary = self.primary or next(iter(self.results_by_classifier))
        if primary in self.results_by_classifier:
            metrics.update(self.results_by_classifier[primary].summary())

        if len(self.results_by_classifier) > 1:
            for clf, res in self.results_by_classifier.items():
                for k, v in res.summary().items():
                    metrics[f"{k}__{clf}"] = v
        return metrics


def _make_classifier(classifier_type: str):
    """Instantiate the requested classifier.

    All classifiers use class_weight='balanced' (or equivalent) so that
    class imbalance between real and synthetic is handled within each fold.
    StratifiedKFold preserves the class ratio across folds.
    """
    if classifier_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    elif classifier_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif classifier_type == "svc":
        # probability=True enables predict_proba via Platt scaling.
        # class_weight='balanced' corrects for real/synth size differences.
        return SVC(
            kernel="linear",  # 'rbf', 'poly'
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    elif classifier_type == "svc_rbf":
        # Non-linear (RBF-kernel) SVM.  probability=False avoids the expensive
        # internal Platt-scaling CV — AUC is computed from decision_function
        # scores, which only need to rank samples, not be calibrated.
        return SVC(
            kernel="rbf",
            gamma="scale",
            probability=False,
            class_weight="balanced",
            random_state=42,
        )
    elif classifier_type == "gradient_boosted":
        # Histogram-based gradient boosting: a strong non-linear discriminator.
        # class_weight='balanced' (added in sklearn 1.5) mirrors the other
        # classifiers' handling of real/synth size differences.
        return HistGradientBoostingClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(
            f"Unknown classifier type '{classifier_type}'. Choose 'random_forest', "
            "'logistic_regression', 'svc', 'svc_rbf', or 'gradient_boosted'."
        )


def _predict_scores(clf, X: np.ndarray) -> np.ndarray:
    """Return per-sample scores for the positive (synthetic) class.

    Prefers calibrated probabilities (``predict_proba``); falls back to the
    uncalibrated ``decision_function`` for estimators built without probability
    support (e.g. an RBF SVM with ``probability=False``).  AUC-ROC is invariant
    to monotonic transforms, so either scoring is valid for ranking.
    """
    if hasattr(clf, "predict_proba") and getattr(clf, "probability", True):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


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
        Which classifier to use — ``"random_forest"`` (default),
        ``"logistic_regression"``, or ``"svc"``.
    n_folds :
        Number of stratified k-fold cross-validation folds.  Default 5.
    peptide_zero_threshold :
        If provided, peptide columns with a zero fraction above this threshold
        in the real data are dropped before the test (mirrors the joint fidelity
        module behaviour).  Ignored when ``peptide_zero_ranges`` is set.
    peptide_zero_ranges :
        If provided, run a separate classifier test for each ``(lo, hi)`` range.
        Only peptide columns whose zero-value fraction in the real data falls in
        ``[lo, hi)`` are included in each sub-test; clinical (non-peptide) columns
        are always included.  When set, ``estimate()`` returns a
        ``TwoSampleClassifierResultsByRange`` instead of
        ``TwoSampleClassifierResults``.
    """

    def __init__(
        self,
        scaler: Optional[Scaler] = None,
        categorical_columns: Optional[List[str]] = None,
        classifier: str = "random_forest",
        n_folds: int = 5,
        peptide_zero_threshold: Optional[float] = None,
        peptide_zero_ranges: Optional[List[Tuple[float, float]]] = None,
    ):
        self.classifier_type = classifier
        self.n_folds = n_folds
        self.peptide_zero_threshold = peptide_zero_threshold
        self.peptide_zero_ranges = peptide_zero_ranges
        self._scaler = scaler if scaler is not None else RobustScaler()
        self._categorical_columns = categorical_columns
        self.feature_processor = FeatureProcessor(
            scaler=self._scaler,
            categorical_columns=categorical_columns,
        )

    def estimate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
    ) -> Union[TwoSampleClassifierResults, TwoSampleClassifierResultsByRange]:
        """Run the two-sample classifier test.

        When ``peptide_zero_ranges`` is configured, runs a separate test for
        each range and returns a ``TwoSampleClassifierResultsByRange``.
        Otherwise returns a single ``TwoSampleClassifierResults``.

        Parameters
        ----------
        real_df :
            Real (reference) dataset.
        synth_df :
            Synthetic dataset to evaluate.
        """
        if self.peptide_zero_ranges is not None:
            return self._estimate_ranged(real_df, synth_df)
        return self._estimate_single(real_df, synth_df)

    def _estimate_single(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        label: Optional[str] = None,
        apply_zero_threshold: bool = True,
    ) -> TwoSampleClassifierResults:
        tag = f" [{label}]" if label else ""
        logger.info(
            f"TwoSampleClassifierTest{tag} — classifier={self.classifier_type}, "
            f"n_folds={self.n_folds}"
        )

        # Optionally drop sparse peptide columns (same logic as JointFidelity).
        # Skipped in ranged mode — the caller already selected columns by range.
        if apply_zero_threshold and self.peptide_zero_threshold is not None:
            drop_cols = sparse_peptide_columns(real_df, self.peptide_zero_threshold)
            if drop_cols:
                real_df = real_df.drop(drop_cols)
                synth_df = synth_df.drop(drop_cols)

        # Only keep columns present in both frames
        shared_cols = [c for c in real_df.columns if c in synth_df.columns]
        real_df = real_df.select(shared_cols)
        synth_df = synth_df.select(shared_cols)

        # Encode features: fit on real data, transform both
        self.feature_processor = FeatureProcessor(
            scaler=self._scaler,
            categorical_columns=self._categorical_columns,
        )
        real_arr = self.feature_processor.fit_transform(real_df)
        synth_arr = self.feature_processor.transform(synth_df)
        feature_names = list(self.feature_processor.feature_names)

        # Stack real (label=0) and synthetic (label=1)
        X = np.concatenate([real_arr, synth_arr], axis=0)
        y = np.concatenate(
            [
                np.zeros(len(real_arr), dtype=int),
                np.ones(len(synth_arr), dtype=int),
            ]
        )

        # Stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        auc_scores: List[float] = []
        fold_importances: List[np.ndarray] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = _make_classifier(self.classifier_type)
            clf.fit(X[train_idx], y[train_idx])

            scores = _predict_scores(clf, X[val_idx])
            auc = float(roc_auc_score(y[val_idx], scores))
            auc_scores.append(auc)
            logger.debug(f"  Fold {fold_idx + 1}/{self.n_folds}{tag} — AUC: {auc:.4f}")

            if hasattr(clf, "feature_importances_"):
                fold_importances.append(clf.feature_importances_)

        auc_mean = float(np.mean(auc_scores))
        auc_std = float(np.std(auc_scores))

        feature_importances = (
            np.mean(fold_importances, axis=0) if fold_importances else None
        )

        logger.info(
            f"TwoSampleClassifier{tag} — AUC: {auc_mean:.4f} ± {auc_std:.4f}  "
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

    def _estimate_ranged(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
    ) -> TwoSampleClassifierResultsByRange:
        """Run a separate classifier test for each configured zero-fraction range."""
        peptide_cols = set(get_peptide_columns(real_df))
        clinical_cols = [c for c in real_df.columns if c not in peptide_cols]

        results_by_range: Dict[str, TwoSampleClassifierResults] = {}
        for lo, hi in self.peptide_zero_ranges:  # type: ignore[union-attr]
            label = f"{lo}-{hi}"
            range_peptides = peptide_columns_in_zero_range(real_df, lo, hi)
            if not range_peptides:
                logger.warning(
                    f"TwoSampleClassifierTest: no peptide columns in zero range "
                    f"[{lo}, {hi}), skipping."
                )
                continue

            selected = [
                c for c in clinical_cols + range_peptides if c in synth_df.columns
            ]
            results_by_range[label] = self._estimate_single(
                real_df.select(selected),
                synth_df.select(selected),
                label=label,
                apply_zero_threshold=False,
            )

        return TwoSampleClassifierResultsByRange(results_by_range=results_by_range)

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(
        self,
        results: Union[TwoSampleClassifierResults, TwoSampleClassifierResultsByRange],
    ) -> plt.Figure:
        """Produce a summary figure for the two-sample classifier test.

        The figure has up to two panels for a single-result test:

        1. **AUC per fold** — bar chart of per-fold AUC scores with the mean
           ± std shown.  A dashed reference line at 0.5 marks chance performance.
        2. **Top-20 feature importances** — horizontal bar chart of the most
           discriminative features (random forest only).

        For ``TwoSampleClassifierResultsByRange``, one AUC-per-fold panel is
        produced per range.
        """
        if isinstance(results, TwoSampleClassifierResultsByRange):
            return self._plot_ranged(results)
        return self._plot_single(results)

    def _plot_single(
        self, results: TwoSampleClassifierResults, title_suffix: str = ""
    ) -> plt.Figure:
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
        ax.bar(
            folds,
            results.auc_per_fold,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(0.5, ls="--", lw=1.2, color="grey", label="Chance (AUC = 0.5)")
        ax.axhline(
            results.auc_mean,
            ls="-",
            lw=1.5,
            color="black",
            label=f"Mean AUC = {results.auc_mean:.4f}",
        )
        ax.fill_between(
            [0.5, len(folds) + 0.5],
            results.auc_mean - results.auc_std,
            results.auc_mean + results.auc_std,
            alpha=0.15,
            color="black",
            label=f"±1 std ({results.auc_std:.4f})",
        )
        ax.set_xlabel("Fold", fontsize=9)
        ax.set_ylabel("AUC-ROC", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(folds)
        clf_label = results.classifier_type.replace("_", " ").title()
        ax.set_title(
            f"Two-Sample Classifier Test{title_suffix}\n"
            f"({clf_label}, {results.n_folds}-fold CV)",
            fontsize=10,
            fontweight="bold",
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
            ax2.barh(
                y_pos, top_imp[::-1], color="#aec7e8", edgecolor="white", linewidth=0.4
            )
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_names[::-1], fontsize=6)
            ax2.set_xlabel("Mean importance", fontsize=9)
            ax2.set_title(
                f"Top-{top_n} Discriminative Features",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        return fig

    def _plot_ranged(self, results: TwoSampleClassifierResultsByRange) -> plt.Figure:
        """One AUC-per-fold panel per zero-fraction range."""
        items = list(results.results_by_range.items())
        n = len(items)
        if n == 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No results", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
        for ax, (label, r) in zip(axes[0], items):
            folds = list(range(1, len(r.auc_per_fold) + 1))
            bar_colors = [
                "#d62728" if auc > 0.7 else "#2ca02c" if auc < 0.55 else "#1f77b4"
                for auc in r.auc_per_fold
            ]
            ax.bar(
                folds,
                r.auc_per_fold,
                color=bar_colors,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.axhline(0.5, ls="--", lw=1.2, color="grey", label="Chance")
            ax.axhline(
                r.auc_mean,
                ls="-",
                lw=1.5,
                color="black",
                label=f"Mean = {r.auc_mean:.4f}",
            )
            ax.fill_between(
                [0.5, len(folds) + 0.5],
                r.auc_mean - r.auc_std,
                r.auc_mean + r.auc_std,
                alpha=0.15,
                color="black",
            )
            ax.set_xlabel("Fold", fontsize=9)
            ax.set_ylabel("AUC-ROC", fontsize=9)
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(folds)
            ax.set_title(
                f"Zero-fraction range [{label})\n" f"(±std: {r.auc_std:.4f})",
                fontsize=10,
                fontweight="bold",
            )
            ax.legend(fontsize=7)

        fig.suptitle(
            f"Two-Sample Classifier Test by Zero-Fraction Range\n"
            f"({items[0][1].classifier_type.replace('_', ' ').title()}, "
            f"{items[0][1].n_folds}-fold CV)",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig


class MultiTwoSampleClassifierTest:
    """Run the two-sample classifier test with several discriminators.

    Instantiates one ``TwoSampleClassifierTest`` per requested classifier type,
    runs them against the same real/synthetic data, and collects the results
    into a ``TwoSampleMultiClassifierResults``.  Sharing the same interface as
    ``TwoSampleClassifierTest`` (``estimate`` / ``plot``) keeps the orchestrating
    ``FidelityReport`` agnostic to whether one or many discriminators run.

    Parameters
    ----------
    classifier_types :
        A single classifier name or a list of them.  Order is preserved and
        duplicates are dropped; the first entry becomes the "primary"
        discriminator.  Valid names: ``"random_forest"``,
        ``"logistic_regression"``, ``"svc"``, ``"gradient_boosted"``.
    scaler, categorical_columns, n_folds, peptide_zero_threshold,
    peptide_zero_ranges :
        Forwarded unchanged to each underlying ``TwoSampleClassifierTest``.
    """

    def __init__(
        self,
        classifier_types: Union[str, List[str]],
        scaler: Optional[Scaler] = None,
        categorical_columns: Optional[List[str]] = None,
        n_folds: int = 5,
        peptide_zero_threshold: Optional[float] = None,
        peptide_zero_ranges: Optional[List[Tuple[float, float]]] = None,
    ):
        if isinstance(classifier_types, str):
            classifier_types = [classifier_types]
        # De-duplicate while preserving order (handles OmegaConf ListConfig too).
        seen: set = set()
        ordered: List[str] = []
        for c in classifier_types:
            c = str(c)
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        if not ordered:
            raise ValueError("MultiTwoSampleClassifierTest: no classifier types given.")

        self.classifier_types = ordered
        self._tests: Dict[str, TwoSampleClassifierTest] = {
            c: TwoSampleClassifierTest(
                scaler=scaler,
                categorical_columns=categorical_columns,
                classifier=c,
                n_folds=n_folds,
                peptide_zero_threshold=peptide_zero_threshold,
                peptide_zero_ranges=peptide_zero_ranges,
            )
            for c in ordered
        }

    def estimate(
        self, real_df: pl.DataFrame, synth_df: pl.DataFrame
    ) -> TwoSampleMultiClassifierResults:
        """Run every configured discriminator and collect their results."""
        logger.info(
            "MultiTwoSampleClassifierTest — discriminators: "
            f"{', '.join(self.classifier_types)}"
        )
        results: Dict[
            str, Union[TwoSampleClassifierResults, TwoSampleClassifierResultsByRange]
        ] = {}
        for clf, test in self._tests.items():
            results[clf] = test.estimate(real_df, synth_df)
        return TwoSampleMultiClassifierResults(
            results_by_classifier=results,
            primary=self.classifier_types[0],
        )

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(self, results: TwoSampleMultiClassifierResults) -> plt.Figure:
        """Comparison figure across discriminators.

        Non-ranged: a bar of mean AUC ± std per discriminator, plus a top
        feature-importances panel from the first tree-based discriminator that
        exposes them.  Ranged: grouped bars with zero-fraction ranges on the
        x-axis and one bar per discriminator in each group.
        """
        items = list(results.results_by_classifier.items())
        if not items:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No results", ha="center", va="center")
            return fig

        if isinstance(items[0][1], TwoSampleClassifierResultsByRange):
            return self._plot_ranged(results)
        return self._plot_single(results)

    @staticmethod
    def _auc_color(auc: float) -> str:
        return "#d62728" if auc > 0.7 else "#2ca02c" if auc < 0.55 else "#1f77b4"

    def _plot_single(self, results: TwoSampleMultiClassifierResults) -> plt.Figure:
        items: List[Tuple[str, TwoSampleClassifierResults]] = list(
            results.results_by_classifier.items()
        )
        n_folds = items[0][1].n_folds

        # Pick a discriminator with feature importances (primary first, then any).
        primary = results.primary or items[0][0]
        order = [primary] + [c for c, _ in items if c != primary]
        imp_clf = next(
            (
                c
                for c in order
                if results.results_by_classifier[c].feature_importances is not None
                and results.results_by_classifier[c].feature_names is not None
            ),
            None,
        )

        n_panels = 2 if imp_clf else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
        if n_panels == 1:
            axes = [axes]

        # ── panel 1: mean AUC ± std per discriminator ─────────────────────
        ax = axes[0]
        labels = [c.replace("_", " ").title() for c, _ in items]
        means = [r.auc_mean for _, r in items]
        stds = [r.auc_std for _, r in items]
        x = np.arange(len(items))
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=[self._auc_color(m) for m in means],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(0.5, ls="--", lw=1.2, color="grey", label="Chance (AUC = 0.5)")
        for xi, m in zip(x, means):
            ax.text(xi, m + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("AUC-ROC", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"Two-Sample Classifier Comparison\n({n_folds}-fold CV, mean ± std)",
            fontsize=10,
            fontweight="bold",
        )
        ax.legend(fontsize=7)

        # ── panel 2: feature importances from a tree-based discriminator ──
        if imp_clf:
            ax2 = axes[1]
            r = results.results_by_classifier[imp_clf]
            importances = r.feature_importances
            names = np.array(r.feature_names)
            top_n = min(20, len(importances))
            top_idx = np.argsort(importances)[-top_n:][::-1]
            y_pos = np.arange(top_n)
            ax2.barh(
                y_pos,
                importances[top_idx][::-1],
                color="#aec7e8",
                edgecolor="white",
                linewidth=0.4,
            )
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names[top_idx][::-1], fontsize=6)
            ax2.set_xlabel("Mean importance", fontsize=9)
            ax2.set_title(
                f"Top-{top_n} Features ({imp_clf.replace('_', ' ').title()})",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        return fig

    def _plot_ranged(self, results: TwoSampleMultiClassifierResults) -> plt.Figure:
        items: List[Tuple[str, TwoSampleClassifierResultsByRange]] = list(
            results.results_by_classifier.items()
        )
        # Union of range labels, ordered by first appearance.
        range_labels: List[str] = []
        for _, r in items:
            for label in r.results_by_range:
                if label not in range_labels:
                    range_labels.append(label)

        if not range_labels:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No ranged results", ha="center", va="center")
            return fig

        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(range_labels)), 4.5))
        x = np.arange(len(range_labels))
        n_clf = len(items)
        width = 0.8 / n_clf
        cmap = plt.get_cmap("tab10")

        for i, (clf, r) in enumerate(items):
            means = [
                r.results_by_range[label].auc_mean if label in r.results_by_range else 0.0
                for label in range_labels
            ]
            stds = [
                r.results_by_range[label].auc_std if label in r.results_by_range else 0.0
                for label in range_labels
            ]
            ax.bar(
                x + (i - (n_clf - 1) / 2) * width,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=cmap(i),
                edgecolor="white",
                linewidth=0.4,
                label=clf.replace("_", " ").title(),
            )

        ax.axhline(0.5, ls="--", lw=1.2, color="grey", label="Chance")
        ax.set_xticks(x)
        ax.set_xticklabels([f"[{lab})" for lab in range_labels], fontsize=8)
        ax.set_xlabel("Zero-fraction range", fontsize=9)
        ax.set_ylabel("AUC-ROC", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            "Two-Sample Classifier Comparison by Zero-Fraction Range",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        return fig

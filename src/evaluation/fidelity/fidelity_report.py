"""FidelityReport — orchestrator for all fidelity metrics.

Runs each sub-module (marginal, correlation, joint, two-sample classifier,
correlation uncertainty, effect size) and collects all results into a single
``FidelityResults`` dataclass.

Usage
-----
    from src.evaluation.fidelity.fidelity_report import FidelityReport
    from sklearn.preprocessing import RobustScaler

    report = FidelityReport(
        categorical_columns=[...],
        scaler=RobustScaler(),
    )
    results = report.run(real_df, synth_df)
    metrics = results.summary()          # flat dict → mlflow.log_metrics(metrics)
    figures = results.figures            # dict of matplotlib figures

MLflow headline metrics logged
-------------------------------
fidelity/mean_ks_statistic
fidelity/mean_wasserstein
fidelity/frac_ks_significant
fidelity/mean_tvd
fidelity/correlation_frobenius
fidelity/correlation_mace
fidelity/mmd
fidelity/precision
fidelity/recall
fidelity/coverage
fidelity/two_sample_auc
fidelity/two_sample_auc_std
correlation_uncertainty/frac_ci_overlap
fidelity/effect_size/cohen_d
fidelity/effect_size/rank_biserial
fidelity/effect_size/median_abs_shift
fidelity/effect_size/cles
fidelity/effect_size/overlap_coef
fidelity/effect_size/normalized_wasserstein
fidelity/effect_size/hellinger
fidelity/effect_size/js_divergence
"""

from __future__ import annotations

import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger
from sklearn.preprocessing import RobustScaler

from src.evaluation.fidelity.marginal_fidelity import (
    MarginalFidelityEstimator,
    MarginalFidelityResults,
)
from src.evaluation.fidelity.correlation_fidelity import (
    CorrelationFidelityEstimator,
    CorrelationFidelityResults,
)
from src.evaluation.fidelity.joint_fidelity import (
    JointFidelityEstimator,
    JointFidelityResults,
)
from src.evaluation.fidelity.classifier_test import (
    TwoSampleClassifierTest,
    TwoSampleClassifierResults,
)
from src.evaluation.analysis.correlation_uncertainty import (
    CorrelationUncertaintyEstimator,
    CorrelationUncertaintyResults,
)
from src.evaluation.fidelity.effect_size import (
    EffectSizeEstimator,
    EffectSizeResults,
    DEFAULT_CONTINUOUS,
    DEFAULT_CATEGORICAL,
)
from src.evaluation.privacy.preprocessing import Scaler


@dataclass
class FidelityResults:
    marginal: Optional[MarginalFidelityResults] = None
    correlation: Optional[CorrelationFidelityResults] = None
    joint: Optional[JointFidelityResults] = None
    classifier: Optional[TwoSampleClassifierResults] = None
    corr_uncertainty: Optional[CorrelationUncertaintyResults] = None
    effect_size: Optional[EffectSizeResults] = None
    figures: Dict[str, plt.Figure] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        """Flat dict of headline metrics — suitable for mlflow.log_metrics()."""
        metrics: Dict[str, float] = {}
        if self.marginal:
            metrics.update(self.marginal.summary())
        if self.correlation:
            metrics.update(self.correlation.summary())
        if self.joint:
            metrics.update(self.joint.summary())
        if self.classifier:
            metrics.update(self.classifier.summary())
        if self.corr_uncertainty:
            metrics.update(self.corr_uncertainty.summary())
        if self.effect_size:
            metrics.update(self.effect_size.summary())
        return metrics

    def to_json(self) -> str:
        """Serialise scalar metrics to JSON for artifact logging."""
        return json.dumps(self.summary(), indent=2)


class FidelityReport:
    """Orchestrator that runs all fidelity sub-modules.

    Parameters
    ----------
    categorical_columns :
        Columns to treat as categorical in distance-based and encoding steps.
    scaler :
        Scaler applied in distance-based modules.  Defaults to RobustScaler.
    run_marginal, run_correlation, run_joint, run_classifier,
    run_corr_uncertainty, run_effect_size :
        Feature flags to enable / disable individual sub-modules.
    corr_method :
        ``"spearman"`` or ``"pearson"`` for correlation sub-modules.
    n_bootstrap :
        Bootstrap iterations for correlation uncertainty.
    n_classifier_folds :
        Cross-validation folds for the two-sample classifier test.
    classifier_type :
        Classifier for the two-sample test.
    max_correlation_cols :
        Column cap for correlation sub-modules.
    max_corr_uncertainty_cols :
        Column cap specifically for the (slower) bootstrap uncertainty module.
    effect_size_continuous_metrics :
        List of continuous effect size metrics to compute.  Must be a subset of
        ``effect_size.CONTINUOUS_METRICS``.  Defaults to all available metrics.
    effect_size_categorical_metrics :
        List of categorical effect size metrics to compute.  Must be a subset of
        ``effect_size.CATEGORICAL_METRICS``.  Defaults to all available metrics.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        scaler: Optional[Scaler] = None,
        run_marginal: bool = True,
        run_correlation: bool = True,
        run_joint: bool = True,
        run_classifier: bool = True,
        run_corr_uncertainty: bool = True,
        run_effect_size: bool = True,
        corr_method: str = "spearman",
        n_bootstrap: int = 1000,
        n_classifier_folds: int = 5,
        classifier_type: str = "random_forest",
        max_correlation_cols: int = 60,
        max_corr_uncertainty_cols: int = 50,
        effect_size_continuous_metrics: Optional[List[str]] = None,
        effect_size_categorical_metrics: Optional[List[str]] = None,
    ):
        self.categorical_columns = categorical_columns or []
        _scaler = scaler if scaler is not None else RobustScaler()

        self._marginal_est = (
            MarginalFidelityEstimator(categorical_columns=self.categorical_columns)
            if run_marginal else None
        )
        self._corr_est = (
            CorrelationFidelityEstimator(method=corr_method, max_columns=max_correlation_cols)
            if run_correlation else None
        )
        self._joint_est = (
            JointFidelityEstimator(scaler=_scaler, categorical_columns=self.categorical_columns)
            if run_joint else None
        )
        self._clf_test = (
            TwoSampleClassifierTest(
                scaler=_scaler,
                categorical_columns=self.categorical_columns,
                classifier=classifier_type,
                n_folds=n_classifier_folds,
            )
            if run_classifier else None
        )
        self._corr_unc_est = (
            CorrelationUncertaintyEstimator(
                method=corr_method,
                n_bootstrap=n_bootstrap,
                max_columns=max_corr_uncertainty_cols,
            )
            if run_corr_uncertainty else None
        )
        self._effect_size_est = (
            EffectSizeEstimator(
                continuous_metrics=effect_size_continuous_metrics,
                categorical_metrics=effect_size_categorical_metrics,
                categorical_columns=self.categorical_columns,
            )
            if run_effect_size else None
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> FidelityResults:
        """Run all enabled fidelity modules and return a FidelityResults object.

        Parameters
        ----------
        real_df :
            Real (reference) dataset.
        synth_df :
            Synthetic dataset to evaluate.
        columns :
            Optional column subset to pass to marginal and correlation modules.
            If None, all shared columns are used.
        """
        results = FidelityResults()

        # Run independent modules in parallel via threads (safe for numpy/scipy)
        tasks: Dict[str, callable] = {}

        if self._marginal_est:
            tasks["marginal"] = lambda: self._marginal_est.estimate(real_df, synth_df, columns)
        if self._corr_est:
            tasks["correlation"] = lambda: self._corr_est.estimate(real_df, synth_df, columns)
        if self._joint_est:
            tasks["joint"] = lambda: self._joint_est.estimate(real_df, synth_df)
        if self._clf_test:
            tasks["classifier"] = lambda: self._clf_test.estimate(real_df, synth_df)
        if self._corr_unc_est:
            tasks["corr_uncertainty"] = lambda: self._corr_unc_est.estimate(real_df, synth_df, columns)
        if self._effect_size_est:
            tasks["effect_size"] = lambda: self._effect_size_est.estimate(real_df, synth_df, columns)

        with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as executor:
            future_map = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    result = future.result()
                    setattr(results, name if name != "corr_uncertainty" else "corr_uncertainty", result)
                    logger.info(f"FidelityReport: '{name}' completed.")
                except Exception as exc:
                    logger.error(f"FidelityReport: '{name}' failed — {exc}")

        # Generate figures from completed results
        results.figures = self._generate_figures(results, real_df, synth_df)

        # Log a compact summary
        summary = results.summary()
        logger.success(
            "FidelityReport complete. Summary:\n"
            + "\n".join(f"  {k}: {v:.4f}" for k, v in summary.items())
        )

        return results

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def _generate_figures(
        self,
        results: FidelityResults,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
    ) -> Dict[str, plt.Figure]:
        figures: Dict[str, plt.Figure] = {}

        if results.marginal and self._marginal_est:
            try:
                figures["marginal_top_divergent"] = self._marginal_est.plot_top_divergent(
                    results.marginal, real_df, synth_df
                )
            except Exception as e:
                logger.warning(f"Could not generate marginal plot: {e}")

        if results.correlation and self._corr_est:
            try:
                figures["correlation_heatmaps"] = self._corr_est.plot(results.correlation)
            except Exception as e:
                logger.warning(f"Could not generate correlation heatmap: {e}")

        if results.classifier and self._clf_test:
            try:
                figures["two_sample_classifier"] = self._clf_test.plot(results.classifier)
            except Exception as e:
                logger.warning(f"Could not generate classifier plot: {e}")

        if results.corr_uncertainty and self._corr_unc_est:
            try:
                figures["correlation_uncertainty"] = self._corr_unc_est.plot(
                    results.corr_uncertainty
                )
            except Exception as e:
                logger.warning(f"Could not generate correlation uncertainty plot: {e}")

        return figures

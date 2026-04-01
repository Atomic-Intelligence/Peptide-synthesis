"""Unified evaluation pipeline for real vs. synthetic data.

Usage
-----
    from evaluation.evaluate import SyntheticDataEvaluator

    evaluator = SyntheticDataEvaluator(categorical_columns=["sex", "cohort"])
    results = evaluator.evaluate(real_df, synth_df)

    # Flat dict of all headline metrics
    print(results.summary())

    # Save all figures to disk
    results.save_figures("./evaluation_output")

    # JSON-serialisable scalar metrics
    print(results.to_json())

The evaluator runs **fidelity** and **privacy** sub-reports in parallel,
each of which internally parallelises their own sub-metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from sklearn.preprocessing import RobustScaler

from evaluation.fidelity.fidelity_report import FidelityReport, FidelityResults
from evaluation.privacy.privacy_report import PrivacyReport, PrivacyResults


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResults:
    """Aggregated results from the full evaluation pipeline."""

    fidelity: Optional[FidelityResults] = None
    privacy: Optional[PrivacyResults] = None
    figures: Dict[str, plt.Figure] = field(default_factory=dict)

    # ── summary ───────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a flat dict of all headline metrics (suitable for logging)."""
        out: Dict[str, Any] = {}
        if self.fidelity is not None:
            out.update(self.fidelity.summary())
        if self.privacy is not None:
            out.update(self.privacy.summary())
        return out

    def to_json(self, indent: int = 2) -> str:
        """JSON string of scalar metrics."""
        raw = self.summary()
        clean: Dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, float) and (v != v):  # NaN
                clean[k] = None
            elif isinstance(v, (int, float, str, bool, type(None))):
                clean[k] = v
            else:
                clean[k] = str(v)
        return json.dumps(clean, indent=indent)

    # ── figures ───────────────────────────────────────────────────────────

    def save_figures(self, output_dir: str | Path, fmt: str = "png", dpi: int = 150) -> List[Path]:
        """Save all collected figures to *output_dir*. Returns list of paths."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for name, fig in self.figures.items():
            path = out_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            logger.info(f"Saved figure: {path}")
        return saved


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class SyntheticDataEvaluator:
    """One-call evaluator: fidelity + privacy for real vs. synthetic data.

    Parameters
    ----------
    categorical_columns : list[str] | None
        Columns to treat as categorical across all sub-metrics.
    scaler
        Scaler for numerical features (default: ``RobustScaler``).

    Fidelity flags
    ~~~~~~~~~~~~~~
    run_marginal, run_correlation, run_joint, run_classifier,
    run_corr_uncertainty, run_effect_size : bool
        Toggle individual fidelity sub-metrics.
    corr_method : "spearman" | "pearson"
    max_correlation_cols : int
    n_bootstrap : int
        Bootstrap iterations for correlation uncertainty.
    max_corr_uncertainty_cols : int
    n_classifier_folds : int
    classifier_type : "random_forest" | "logistic_regression" | "gradient_boosted"
    effect_size_continuous_metrics / effect_size_categorical_metrics : list[str] | None

    Privacy flags
    ~~~~~~~~~~~~~
    run_dcr, run_authenticity, run_mia, run_reidentification : bool
        Toggle individual privacy sub-metrics.
    holdout_fraction : float
        Fraction of real data held out for DCR baseline / MIA non-members.
    par_percentile : float
        Percentile threshold for privacy-at-risk (DCR).
    authenticity_threshold : float
    mia_holdout_fraction : float
    mia_attack_signal : "dcr" | "likelihood" | "both"
    mia_n_folds : int
    reid_risk_threshold : float
    dcr_distance_metric / reid_distance_metric : "euclidean" | "gower"
    """

    def __init__(
        self,
        # ── shared ────────────────────────────────────────────────────────
        categorical_columns: Optional[List[str]] = None,
        scaler=None,
        # ── fidelity toggles ──────────────────────────────────────────────
        run_fidelity: bool = True,
        run_marginal: bool = True,
        run_correlation: bool = True,
        run_joint: bool = True,
        run_classifier: bool = True,
        run_corr_uncertainty: bool = True,
        run_effect_size: bool = True,
        corr_method: Literal["spearman", "pearson"] = "spearman",
        max_correlation_cols: int = 60,
        n_bootstrap: int = 1000,
        max_corr_uncertainty_cols: int = 50,
        n_classifier_folds: int = 5,
        classifier_type: Literal[
            "random_forest", "logistic_regression", "gradient_boosted"
        ] = "random_forest",
        effect_size_continuous_metrics: Optional[List[str]] = None,
        effect_size_categorical_metrics: Optional[List[str]] = None,
        # ── privacy toggles ───────────────────────────────────────────────
        run_privacy: bool = True,
        run_dcr: bool = True,
        run_authenticity: bool = True,
        run_mia: bool = True,
        run_reidentification: bool = True,
        holdout_fraction: float = 0.5,
        par_percentile: float = 5.0,
        authenticity_threshold: float = 1.0,
        mia_holdout_fraction: float = 0.2,
        mia_attack_signal: Literal["dcr", "likelihood", "both"] = "dcr",
        mia_n_folds: int = 5,
        reid_risk_threshold: float = 0.5,
        dcr_distance_metric: Literal["euclidean", "gower"] = "euclidean",
        reid_distance_metric: Literal["euclidean", "gower"] = "euclidean",
    ):
        self.categorical_columns = categorical_columns
        self.scaler = scaler or RobustScaler()

        # Fidelity
        self.run_fidelity = run_fidelity
        self._fidelity_report = FidelityReport(
            categorical_columns=categorical_columns,
            scaler=self.scaler,
            run_marginal=run_marginal,
            run_correlation=run_correlation,
            run_joint=run_joint,
            run_classifier=run_classifier,
            run_corr_uncertainty=run_corr_uncertainty,
            run_effect_size=run_effect_size,
            corr_method=corr_method,
            n_bootstrap=n_bootstrap,
            n_classifier_folds=n_classifier_folds,
            classifier_type=classifier_type,
            max_correlation_cols=max_correlation_cols,
            max_corr_uncertainty_cols=max_corr_uncertainty_cols,
            effect_size_continuous_metrics=effect_size_continuous_metrics,
            effect_size_categorical_metrics=effect_size_categorical_metrics,
        )

        # Privacy
        self.run_privacy = run_privacy
        self._privacy_report = PrivacyReport(
            categorical_columns=categorical_columns,
            scaler=self.scaler,
            holdout_fraction=holdout_fraction,
            par_percentile=par_percentile,
            run_dcr=run_dcr,
            run_authenticity=run_authenticity,
            authenticity_threshold=authenticity_threshold,
            run_mia=run_mia,
            mia_holdout_fraction=mia_holdout_fraction,
            mia_attack_signal=mia_attack_signal,
            mia_n_folds=mia_n_folds,
            run_reidentification=run_reidentification,
            reid_risk_threshold=reid_risk_threshold,
            dcr_distance_metric=dcr_distance_metric,
            reid_distance_metric=reid_distance_metric,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> EvaluationResults:
        """Run the full evaluation pipeline.

        Parameters
        ----------
        real_df : pl.DataFrame
            The real (training) dataset.
        synth_df : pl.DataFrame
            The synthetic dataset to evaluate.
        columns : list[str] | None
            Restrict evaluation to these columns. If ``None``, use all
            columns common to both dataframes.

        Returns
        -------
        EvaluationResults
            Aggregated fidelity + privacy results, figures, and a
            ``.summary()`` method for flat metric export.
        """
        # Align columns
        if columns is not None:
            common = [c for c in columns if c in real_df.columns and c in synth_df.columns]
        else:
            common = [c for c in real_df.columns if c in synth_df.columns]

        if not common:
            raise ValueError("No overlapping columns between real and synthetic dataframes.")

        real_df = real_df.select(common)
        synth_df = synth_df.select(common)

        logger.info(
            f"Evaluation pipeline — {len(common)} columns, "
            f"real={len(real_df)} rows, synth={len(synth_df)} rows"
        )

        fidelity_results: Optional[FidelityResults] = None
        privacy_results: Optional[PrivacyResults] = None
        all_figures: Dict[str, plt.Figure] = {}

        # ── Fidelity ──────────────────────────────────────────────────────
        if self.run_fidelity:
            logger.info("Running fidelity report ...")
            fidelity_results = self._fidelity_report.run(real_df, synth_df, columns=common)
            if fidelity_results.figures:
                all_figures.update(
                    {f"fidelity_{k}": v for k, v in fidelity_results.figures.items()}
                )
            logger.info("Fidelity report complete.")

        # ── Privacy ───────────────────────────────────────────────────────
        if self.run_privacy:
            logger.info("Running privacy report ...")
            privacy_results = self._privacy_report.run(real_df, synth_df)
            logger.info("Privacy report complete.")

        results = EvaluationResults(
            fidelity=fidelity_results,
            privacy=privacy_results,
            figures=all_figures,
        )

        logger.info("Evaluation pipeline finished.")
        return results

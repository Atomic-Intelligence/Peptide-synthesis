"""PrivacyReport — orchestrator for DCR and Authenticity privacy metrics.

Fits a single FeatureProcessor on the real dataset and shares it between
DCREstimator and AuthenticityEstimator, eliminating the redundant fit step
that would otherwise occur when running both metrics independently.

DCR and Authenticity are complementary, not redundant:
  - DCR answers "are any synthetic records dangerously close to a real record?"
    (privacy risk, compared to a real-to-real holdout baseline)
  - Authenticity answers "do synthetic records look like plausible members of
    the real distribution?" (data quality, per-record ratio vs. nearest real
    neighbour's own nearest neighbour)

The overlap is entirely in preprocessing (FeatureProcessor) and in the
underlying synth→real kNN lookup, which this class computes once.

Usage
-----
    report = PrivacyReport(categorical_columns=[...])
    results = report.run(real_df, synth_df)
    print(results.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import polars as pl
from loguru import logger
from sklearn.preprocessing import RobustScaler

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler
from src.evaluation.privacy.dcr import DCREstimator, DCRResults
from src.evaluation.privacy.AuthenticityEstimator import (
    AuthenticityEstimator,
    AuthenticityResults,
)


@dataclass
class PrivacyResults:
    dcr: Optional[DCRResults] = None
    authenticity: Optional[AuthenticityResults] = None

    def summary(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if self.dcr is not None:
            metrics.update({f"privacy/dcr_{k}": v for k, v in self.dcr.summary().items()})
        if self.authenticity is not None:
            metrics.update(
                {f"privacy/auth_{k}": v for k, v in self.authenticity.summary().items()}
            )
        return metrics


class PrivacyReport:
    """Run DCR and Authenticity with a shared FeatureProcessor.

    Parameters
    ----------
    categorical_columns :
        Columns to one-hot encode rather than scale.
    scaler :
        Sklearn scaler for numerical features.  Defaults to RobustScaler.
    holdout_fraction :
        Fraction of real data held out for the DCR baseline.
    par_percentile :
        Privacy-at-Risk percentile threshold for DCR.
    run_dcr :
        Whether to run the DCR estimator.
    run_authenticity :
        Whether to run the Authenticity estimator.
    authenticity_threshold :
        Ratio threshold used by AuthenticityEstimator.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        scaler: Optional[Scaler] = None,
        holdout_fraction: float = 0.5,
        par_percentile: float = 5.0,
        run_dcr: bool = True,
        run_authenticity: bool = True,
        authenticity_threshold: float = 1.0,
    ):
        self.categorical_columns = categorical_columns or []
        self._scaler = scaler if scaler is not None else RobustScaler()
        self.holdout_fraction = holdout_fraction
        self.par_percentile = par_percentile
        self.run_dcr = run_dcr
        self.run_authenticity = run_authenticity
        self.authenticity_threshold = authenticity_threshold

    def run(self, real_df: pl.DataFrame, synth_df: pl.DataFrame) -> PrivacyResults:
        """Run enabled privacy metrics with a shared FeatureProcessor.

        The processor is fitted once on the full real dataset, then passed to
        both estimators so they call only transform() rather than fit_transform().

        Parameters
        ----------
        real_df :
            Real (reference) dataset.
        synth_df :
            Synthetic dataset to evaluate.
        """
        results = PrivacyResults()

        # Fit the shared processor on all real data once.
        shared_fp = FeatureProcessor(
            scaler=self._scaler,
            categorical_columns=self.categorical_columns,
        )
        shared_fp.fit(real_df)
        logger.info("PrivacyReport: shared FeatureProcessor fitted on real data.")

        if self.run_dcr:
            try:
                dcr = DCREstimator(
                    holdout_fraction=self.holdout_fraction,
                    par_percentile=self.par_percentile,
                    fitted_feature_processor=shared_fp,
                )
                dcr.fit(real_df)
                results.dcr = dcr.estimate(synth_df)
                logger.success(f"PrivacyReport DCR: {results.dcr.summary()}")
            except Exception as exc:
                logger.error(f"PrivacyReport: DCR failed — {exc}")

        if self.run_authenticity:
            try:
                auth = AuthenticityEstimator(
                    scaler=self._scaler,
                    categorical_columns=self.categorical_columns,
                    authenticity_threshold=self.authenticity_threshold,
                    fitted_feature_processor=shared_fp,
                )
                auth.fit(real_df)
                results.authenticity = auth.estimate_authenticity(
                    synth_df, return_suspicious=True
                )
                logger.success(f"PrivacyReport Authenticity: {results.authenticity.summary()}")
            except Exception as exc:
                logger.error(f"PrivacyReport: Authenticity failed — {exc}")

        return results

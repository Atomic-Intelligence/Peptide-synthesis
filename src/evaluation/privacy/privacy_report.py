"""PrivacyReport — orchestrator for all privacy metrics.

Fits a single FeatureProcessor on the real dataset and shares it between
all estimators, eliminating the redundant fit step that would otherwise occur
when running metrics independently.

Metrics
-------
  - DCR: "are any synthetic records dangerously close to a real record?"
    (compared to a real-to-real holdout baseline)
  - Authenticity: "do synthetic records look like plausible members of the
    real distribution?" (per-record ratio vs. nearest real neighbour's own NN)
  - Re-identification: "can an attacker link a real person to their synthetic
    copy?" (gap ratio d(2nd-NN_synth) / d(1st-NN_synth))
  - Singling out (anonymeter): can an adversary craft queries from the synthetic
    data to uniquely identify a real individual? (univariate + multivariate)
  - Linkability (anonymeter): can the synthetic data bridge two attribute
    views of the same individual across datasets?
  - Attribute inference (anonymeter): can the adversary infer unknown attribute
    values from the nearest synthetic neighbour?

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
from src.evaluation.utils.eval_utils import sparse_peptide_columns
from src.evaluation.privacy.AuthenticityEstimator import (
    AuthenticityEstimator,
    AuthenticityResults,
)
from src.evaluation.privacy.reidentification_risk import (
    ReidentificationRiskEstimator,
    ReidentificationResults,
)
from src.evaluation.privacy.anonymeter_attacks import (
    AnonymeterAttacksEstimator,
    AnonymeterResults,
)
from src.evaluation.privacy.membership_inference import (
    MembershipInferenceAttack,
    MIAResults,
)


@dataclass
class PrivacyResults:
    dcr: Optional[DCRResults] = None
    authenticity: Optional[AuthenticityResults] = None
    reidentification: Optional[ReidentificationResults] = None
    anonymeter: Optional[AnonymeterResults] = None
    mia: Optional[MIAResults] = None

    def summary(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if self.dcr is not None:
            metrics.update({f"privacy/dcr_{k}": v for k, v in self.dcr.summary().items()})
        if self.authenticity is not None:
            metrics.update(
                {f"privacy/auth_{k}": v for k, v in self.authenticity.summary().items()}
            )
        if self.reidentification is not None:
            metrics.update(
                {f"privacy/reid_{k}": v for k, v in self.reidentification.summary().items()}
            )
        if self.anonymeter is not None:
            metrics.update(
                {f"privacy/{k}": v for k, v in self.anonymeter.summary().items()}
            )
        if self.mia is not None:
            metrics.update(
                {f"privacy/{k}": v for k, v in self.mia.summary().items()}
            )
        return metrics


class PrivacyReport:
    """Run DCR, Authenticity, Re-identification, and anonymeter attacks.

    Parameters
    ----------
    categorical_columns :
        Columns to one-hot encode rather than scale.
    scaler :
        Sklearn scaler for numerical features.  Defaults to RobustScaler.
    holdout_fraction :
        Fraction of real data held out for the DCR baseline and anonymeter
        control set.
    par_percentile :
        Privacy-at-Risk percentile threshold for DCR.
    run_dcr :
        Whether to run the DCR estimator.
    run_authenticity :
        Whether to run the Authenticity estimator.
    authenticity_threshold :
        Ratio threshold used by AuthenticityEstimator.
    run_reidentification :
        Whether to run the Re-identification Risk estimator.  Default True.
    reid_risk_threshold :
        Gap-ratio threshold for re-identification risk.
        ``gap = d(2nd-NN_synth) / d(1st-NN_synth)``; records with
        gap > threshold are flagged as at risk.  Default 2.0.
    dcr_distance_metric :
        Distance metric for DCR: ``"euclidean"`` or ``"gower"``.
    reid_distance_metric :
        Distance metric for re-identification risk: ``"euclidean"`` or ``"gower"``.
    peptide_zero_threshold :
        Peptide columns whose real-data zero fraction exceeds this value are
        dropped before any privacy metric is computed.  ``None`` disables
        filtering (default).
    run_singling_out :
        Whether to run the anonymeter singling out attack (univariate +
        multivariate).  Default True.
    run_linkability :
        Whether to run the anonymeter linkability attack.  Default True.
    run_attribute_inference :
        Whether to run the anonymeter attribute inference attack.  Default True.
    n_anonymeter_attacks :
        Number of attack queries for each anonymeter evaluator.  Default 2000.
    linkability_aux_cols :
        Tuple ``(columns_A, columns_B)`` for the linkability attack.
        When ``None`` (default), columns are split 50 / 50 randomly.
    inference_target_cols :
        Columns the adversary tries to infer.  Defaults to
        ``categorical_columns``.
    anonymeter_n_jobs :
        Parallelism for anonymeter evaluators.  Default -1 (all cores).
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
        run_reidentification: bool = True,
        reid_risk_threshold: float = 2.0,
        dcr_distance_metric: str = "euclidean",
        reid_distance_metric: str = "euclidean",
        peptide_zero_threshold: Optional[float] = None,
        run_singling_out: bool = True,
        run_linkability: bool = True,
        run_attribute_inference: bool = True,
        n_anonymeter_attacks: int = 2000,
        linkability_aux_cols: Optional[tuple] = None,
        inference_target_cols: Optional[List[str]] = None,
        anonymeter_n_jobs: int = -1,
        gap_ratio_threshold: float = 2.0,
        inference_tolerance: float = 0.1,
        run_membership_inference: bool = True,
        mia_attack_signal: str = "dcr",
        mia_classifier: str = "logistic_regression",
        mia_n_folds: int = 5,
        mia_high_risk_threshold: float = 0.9,
    ):
        self.categorical_columns = categorical_columns or []
        self._scaler = scaler if scaler is not None else RobustScaler()
        self.holdout_fraction = holdout_fraction
        self.par_percentile = par_percentile
        self.run_dcr = run_dcr
        self.run_authenticity = run_authenticity
        self.authenticity_threshold = authenticity_threshold
        self.run_reidentification = run_reidentification
        self.reid_risk_threshold = reid_risk_threshold
        self.dcr_distance_metric = dcr_distance_metric
        self.reid_distance_metric = reid_distance_metric
        self.peptide_zero_threshold = peptide_zero_threshold
        self.run_singling_out = run_singling_out
        self.run_linkability = run_linkability
        self.run_attribute_inference = run_attribute_inference
        self.n_anonymeter_attacks = n_anonymeter_attacks
        self.linkability_aux_cols = linkability_aux_cols
        self.inference_target_cols = inference_target_cols
        self.anonymeter_n_jobs = anonymeter_n_jobs
        self.gap_ratio_threshold = gap_ratio_threshold
        self.inference_tolerance = inference_tolerance
        self.run_membership_inference = run_membership_inference
        self.mia_attack_signal = mia_attack_signal
        self.mia_classifier = mia_classifier
        self.mia_n_folds = mia_n_folds
        self.mia_high_risk_threshold = mia_high_risk_threshold

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

        # Drop sparse peptide columns before any distance/encoding step.
        if self.peptide_zero_threshold is not None:
            sparse_cols = sparse_peptide_columns(real_df, self.peptide_zero_threshold)
            if sparse_cols:
                real_df = real_df.drop(sparse_cols)
                synth_df = synth_df.drop([c for c in sparse_cols if c in synth_df.columns])
                logger.info(
                    f"PrivacyReport: dropped {len(sparse_cols)} sparse peptide columns "
                    f"(zero fraction > {self.peptide_zero_threshold:.0%}) before privacy estimation."
                )

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
                    distance_metric=self.dcr_distance_metric,
                    categorical_columns=self.categorical_columns,
                    fitted_feature_processor=shared_fp if self.dcr_distance_metric != "gower" else None,
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

        if self.run_reidentification:
            try:
                reid = ReidentificationRiskEstimator(
                    risk_threshold=self.reid_risk_threshold,
                    distance_metric=self.reid_distance_metric,
                    categorical_columns=self.categorical_columns,
                    fitted_feature_processor=shared_fp if self.reid_distance_metric != "gower" else None,
                )
                reid.fit(real_df)
                results.reidentification = reid.estimate(synth_df)
                logger.success(
                    f"PrivacyReport Re-identification: {results.reidentification.summary()}"
                )
            except Exception as exc:
                logger.error(f"PrivacyReport: Re-identification failed — {exc}")

        run_any_anonymeter = (
            self.run_singling_out or self.run_linkability or self.run_attribute_inference
        )
        if run_any_anonymeter:
            try:
                anon = AnonymeterAttacksEstimator(
                    categorical_columns=self.categorical_columns,
                    holdout_fraction=self.holdout_fraction,
                    n_attacks=self.n_anonymeter_attacks,
                    run_singling_out=self.run_singling_out,
                    run_linkability=self.run_linkability,
                    run_attribute_inference=self.run_attribute_inference,
                    linkability_aux_cols=self.linkability_aux_cols,
                    inference_target_cols=self.inference_target_cols,
                    n_jobs=self.anonymeter_n_jobs,
                    gap_ratio_threshold=self.gap_ratio_threshold,
                    inference_tolerance=self.inference_tolerance,
                )
                anon.fit(real_df)
                results.anonymeter = anon.estimate(synth_df)
                logger.success(
                    f"PrivacyReport Anonymeter: {results.anonymeter.summary()}"
                )
            except Exception as exc:
                logger.error(f"PrivacyReport: Anonymeter attacks failed — {exc}")

        if self.run_membership_inference:
            try:
                mia = MembershipInferenceAttack(
                    scaler=self._scaler,
                    categorical_columns=self.categorical_columns,
                    holdout_fraction=self.holdout_fraction,
                    attack_signal=self.mia_attack_signal,
                    classifier=self.mia_classifier,
                    n_folds=self.mia_n_folds,
                    high_risk_threshold=self.mia_high_risk_threshold,
                )
                mia.fit(real_df)
                results.mia = mia.estimate(synth_df)
                logger.success(f"PrivacyReport MIA: {results.mia.summary()}")
            except Exception as exc:
                logger.error(f"PrivacyReport: Membership Inference Attack failed — {exc}")

        return results

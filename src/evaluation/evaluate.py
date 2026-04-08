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

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import RobustScaler

from src.evaluation.fidelity.fidelity_report import FidelityReport, FidelityResults
from src.evaluation.privacy.privacy_report import PrivacyReport, PrivacyResults
from src.evaluation.utils.eval_utils import sparse_peptide_columns


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResults:
    """Aggregated results from the full evaluation pipeline."""

    fidelity: Optional[FidelityResults] = None
    privacy: Optional[PrivacyResults] = None
    figures: Dict[str, plt.Figure] = field(default_factory=dict)
    per_event: Dict[str, "EvaluationResults"] = field(default_factory=dict)

    # ── summary ───────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a flat dict of all headline metrics (suitable for logging).

        Per-event metrics are included with keys prefixed by ``<event>/``.
        """
        out: Dict[str, Any] = {}
        if self.fidelity is not None:
            out.update(self.fidelity.summary())
        if self.privacy is not None:
            out.update(self.privacy.summary())
        for event, event_results in self.per_event.items():
            for k, v in event_results.summary().items():
                out[f"{event}/{k}"] = v
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

    def save_figures(
        self, output_dir: str | Path, fmt: str = "png", dpi: int = 150
    ) -> List[Path]:
        """Save all collected figures to *output_dir*. Returns list of paths.

        Per-event figures are saved to ``<output_dir>/<event>/``.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for name, fig in self.figures.items():
            path = out_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            logger.info(f"Saved figure: {path}")
        for event, event_results in self.per_event.items():
            event_dir = out_dir / event
            event_dir.mkdir(parents=True, exist_ok=True)
            for name, fig in event_results.figures.items():
                path = event_dir / f"{name}.{fmt}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
                logger.info(f"Saved figure [{event}]: {path}")
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
        run_sparse_peptide: bool = True,
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
        peptide_zero_threshold: Optional[float] = None,
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
        self.peptide_zero_threshold = peptide_zero_threshold

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
            run_sparse_peptide=run_sparse_peptide,
            corr_method=corr_method,
            n_bootstrap=n_bootstrap,
            n_classifier_folds=n_classifier_folds,
            classifier_type=classifier_type,
            max_correlation_cols=max_correlation_cols,
            max_corr_uncertainty_cols=max_corr_uncertainty_cols,
            effect_size_continuous_metrics=effect_size_continuous_metrics,
            effect_size_categorical_metrics=effect_size_categorical_metrics,
            peptide_zero_threshold=peptide_zero_threshold,
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
            peptide_zero_threshold=peptide_zero_threshold,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        drop_nan: bool = False,
        per_event_column: Optional[str] = None,
        event_types: Optional[List[str]] = None,
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
        drop_nan : bool
            If ``True``, drop rows containing any null/NaN values from
            both dataframes before evaluation and log the row counts dropped.
        per_event_column : str | None
            Column name used to stratify per-event sub-evaluations (e.g.
            ``"event_type"``). The column is used only for filtering and is
            excluded from the feature set passed to each sub-report.
        event_types : list[str] | None
            Event values to evaluate individually. Defaults to
            ``["hf", "ckf", "no_event"]`` when *per_event_column* is set.

        Returns
        -------
        EvaluationResults
            Aggregated fidelity + privacy results, figures, and a
            ``.summary()`` method for flat metric export. Per-event reports
            are available under ``results.per_event[<event>]``.
        """
        # Align columns
        if columns is not None:
            common = [
                c for c in columns if c in real_df.columns and c in synth_df.columns
            ]
        else:
            common = [c for c in real_df.columns if c in synth_df.columns]

        if not common:
            raise ValueError(
                "No overlapping columns between real and synthetic dataframes."
            )

        # Keep the event column in the working frames for per-event filtering,
        # but exclude it from the feature columns passed to sub-reports.
        event_col_present = per_event_column is not None and per_event_column in common
        eval_common = (
            [c for c in common if c != per_event_column]
            if event_col_present
            else common
        )

        real_df = real_df.select(common)
        synth_df = synth_df.select(common)

        # Optionally drop rows with any null/NaN values
        if drop_nan:
            real_before = len(real_df)
            synth_before = len(synth_df)
            real_df = real_df.drop_nulls()
            synth_df = synth_df.drop_nulls()
            real_dropped = real_before - len(real_df)
            synth_dropped = synth_before - len(synth_df)
            if real_dropped:
                logger.warning(
                    f"Dropped {real_dropped} rows with nulls from real_df "
                    f"({real_before} → {len(real_df)} rows)"
                )
            if synth_dropped:
                logger.warning(
                    f"Dropped {synth_dropped} rows with nulls from synth_df "
                    f"({synth_before} → {len(synth_df)} rows)"
                )

        logger.info(
            f"Evaluation pipeline — {len(eval_common)} feature columns, "
            f"real={len(real_df)} rows, synth={len(synth_df)} rows"
        )

        fidelity_results: Optional[FidelityResults] = None
        privacy_results: Optional[PrivacyResults] = None
        all_figures: Dict[str, plt.Figure] = {}

        # Working frames without the stratification column
        real_feat = real_df.select(eval_common)
        synth_feat = synth_df.select(eval_common)

        # Preserve full frames (with sparse peptide columns) so the
        # SparsePeptideFidelityEstimator can receive them intact.
        real_feat_full = real_feat
        synth_feat_full = synth_feat
        eval_common_full = list(eval_common)

        # Drop sparse peptide columns from the frames passed to classifier,
        # joint fidelity, and other modules that are distorted by zero-inflation.
        # NOTE: classifier_test and joint_fidelity also do this internally; the
        # top-level drop ensures the remaining modules (marginal, correlation,
        # effect_size) are also unaffected.
        if self.peptide_zero_threshold is not None:
            sparse_cols = sparse_peptide_columns(real_feat, self.peptide_zero_threshold)
            if sparse_cols:
                real_feat = real_feat.drop(sparse_cols)
                synth_feat = synth_feat.drop(sparse_cols)
                eval_common = [c for c in eval_common if c not in sparse_cols]
                logger.info(
                    f"Dropped {len(sparse_cols)} sparse peptide columns from full dataset "
                    f"before event splitting (threshold={self.peptide_zero_threshold})"
                )

        # ── Fidelity ──────────────────────────────────────────────────────
        if self.run_fidelity:
            logger.info("Running fidelity report ...")
            fidelity_results = self._fidelity_report.run(
                real_feat, synth_feat, columns=eval_common,
                real_df_full=real_feat_full, synth_df_full=synth_feat_full,
            )
            if fidelity_results.figures:
                all_figures.update(
                    {f"fidelity_{k}": v for k, v in fidelity_results.figures.items()}
                )
            logger.info("Fidelity report complete.")

        # ── Privacy ───────────────────────────────────────────────────────
        if self.run_privacy:
            logger.info("Running privacy report ...")
            privacy_results = self._privacy_report.run(real_feat, synth_feat)
            logger.info("Privacy report complete.")

        # ── Per-event sub-reports ─────────────────────────────────────────
        per_event: Dict[str, EvaluationResults] = {}
        if event_col_present:
            _event_types = event_types or ["hf", "ckf", "no_event"]
            logger.info(
                f"Running per-event evaluation on column '{per_event_column}' "
                f"for events: {_event_types}"
            )
            for event in _event_types:
                real_ev = real_df.filter(pl.col(per_event_column) == event).select(
                    eval_common
                )
                synth_ev = synth_df.filter(pl.col(per_event_column) == event).select(
                    eval_common
                )
                real_ev_full = real_df.filter(pl.col(per_event_column) == event).select(
                    eval_common_full
                )
                synth_ev_full = synth_df.filter(pl.col(per_event_column) == event).select(
                    eval_common_full
                )

                if len(real_ev) == 0 or len(synth_ev) == 0:
                    logger.warning(
                        f"Skipping event '{event}': "
                        f"real={len(real_ev)} rows, synth={len(synth_ev)} rows"
                    )
                    continue

                logger.info(
                    f"[{event}] real={len(real_ev)} rows, synth={len(synth_ev)} rows"
                )

                ev_fidelity: Optional[FidelityResults] = None
                ev_privacy: Optional[PrivacyResults] = None
                ev_figures: Dict[str, plt.Figure] = {}

                if self.run_fidelity:
                    logger.info(f"[{event}] Running fidelity report ...")
                    ev_fidelity = self._fidelity_report.run(
                        real_ev, synth_ev, columns=eval_common,
                        real_df_full=real_ev_full, synth_df_full=synth_ev_full,
                    )
                    if ev_fidelity.figures:
                        ev_figures.update(
                            {f"fidelity_{k}": v for k, v in ev_fidelity.figures.items()}
                        )
                    logger.info(f"[{event}] Fidelity report complete.")

                if self.run_privacy:
                    logger.info(f"[{event}] Running privacy report ...")
                    ev_privacy = self._privacy_report.run(real_ev, synth_ev)
                    logger.info(f"[{event}] Privacy report complete.")

                per_event[event] = EvaluationResults(
                    fidelity=ev_fidelity,
                    privacy=ev_privacy,
                    figures=ev_figures,
                )

            logger.info(f"Per-event reports completed for: {list(per_event.keys())}")

        results = EvaluationResults(
            fidelity=fidelity_results,
            privacy=privacy_results,
            figures=all_figures,
            per_event=per_event,
        )

        logger.info("Evaluation pipeline finished.")
        return results


# ---------------------------------------------------------------------------
# CLI entry point (Hydra)
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.1",
    config_path="../../configs/evaluation",
    config_name="evaluate",
)
def main(cfg: DictConfig) -> None:
    """Run the unified evaluation pipeline from the command line.

    Example
    -------
    python -m src.evaluation.evaluate \\
        paths.real_data_path=/data/real.csv \\
        paths.synthetic_data_path=/data/synth.csv
    """
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    # ── load data ─────────────────────────────────────────────────────────
    real_path = Path(cfg.paths.real_data_path)
    synth_path = Path(cfg.paths.synthetic_data_path)

    logger.info(f"Loading real data from {real_path}")
    real_df = pl.read_csv(real_path)

    logger.debug(f"Loaded real shape {real_df.shape}")
    logger.info(f"Loading synthetic data from {synth_path}")
    synth_df = pl.read_csv(synth_path)
    logger.info(f"Loading synthetic data from {synth_path}")
    logger.debug(f"Loaded synthetic shape {synth_df.shape}")

    # ── build evaluator ───────────────────────────────────────────────────
    fid = cfg.fidelity
    priv = cfg.privacy
    cat_cols = (
        OmegaConf.to_container(cfg.categorical_columns, resolve=True)
        if cfg.categorical_columns
        else None
    )
    columns = OmegaConf.to_container(cfg.columns, resolve=True) if cfg.columns else None
    id_cols = (
        OmegaConf.to_container(cfg.id_columns, resolve=True)
        if cfg.get("id_columns")
        else []
    )
    if id_cols:
        real_df = real_df.drop([c for c in id_cols if c in real_df.columns])
        synth_df = synth_df.drop([c for c in id_cols if c in synth_df.columns])
        logger.info(f"Dropped ID columns: {id_cols}")

    eff_cont = (
        OmegaConf.to_container(fid.effect_size_metrics.continuous, resolve=True)
        if fid.get("effect_size_metrics")
        else None
    )
    eff_cat = (
        OmegaConf.to_container(fid.effect_size_metrics.categorical, resolve=True)
        if fid.get("effect_size_metrics")
        else None
    )

    evaluator = SyntheticDataEvaluator(
        categorical_columns=cat_cols,
        # fidelity
        run_fidelity=cfg.run_fidelity,
        run_marginal=fid.run_marginal,
        run_correlation=fid.run_correlation,
        run_joint=fid.run_joint,
        run_classifier=fid.run_classifier,
        run_corr_uncertainty=fid.run_corr_uncertainty,
        run_effect_size=fid.run_effect_size,
        run_sparse_peptide=fid.get("run_sparse_peptide", True),
        corr_method=fid.corr_method,
        max_correlation_cols=fid.max_correlation_cols,
        n_bootstrap=fid.n_bootstrap,
        max_corr_uncertainty_cols=fid.max_corr_uncertainty_cols,
        n_classifier_folds=fid.n_classifier_folds,
        classifier_type=fid.classifier_type,
        effect_size_continuous_metrics=eff_cont,
        effect_size_categorical_metrics=eff_cat,
        peptide_zero_threshold=fid.get("peptide_zero_threshold", None),
        # privacy
        run_privacy=cfg.run_privacy,
        run_dcr=priv.run_dcr,
        run_authenticity=priv.run_authenticity,
        run_mia=priv.run_mia,
        run_reidentification=priv.run_reidentification,
        holdout_fraction=priv.holdout_fraction,
        par_percentile=priv.par_percentile,
        authenticity_threshold=priv.authenticity_threshold,
        mia_holdout_fraction=priv.mia_holdout_fraction,
        mia_attack_signal=priv.mia_attack_signal,
        mia_n_folds=priv.mia_n_folds,
        reid_risk_threshold=priv.reid_risk_threshold,
        dcr_distance_metric=priv.dcr_distance_metric,
        reid_distance_metric=priv.reid_distance_metric,
    )

    # ── per-event config ──────────────────────────────────────────────────
    per_event_cfg = cfg.get("per_event_evaluation", None)
    per_event_col: Optional[str] = None
    per_event_types: Optional[List[str]] = None
    if per_event_cfg and per_event_cfg.get("enabled", False):
        per_event_col = per_event_cfg.column
        per_event_types = (
            OmegaConf.to_container(per_event_cfg.event_types, resolve=True)
            if per_event_cfg.get("event_types")
            else None
        )

    # ── run ───────────────────────────────────────────────────────────────
    results = evaluator.evaluate(
        real_df,
        synth_df,
        columns=columns,
        drop_nan=cfg.get("drop_nan", False),
        per_event_column=per_event_col,
        event_types=per_event_types,
    )

    # ── save outputs ──────────────────────────────────────────────────────
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.save_figures:
        saved = results.save_figures(out_dir, fmt=cfg.figures_fmt, dpi=cfg.figures_dpi)
        if saved:
            logger.info(f"Saved {len(saved)} figure(s) to {out_dir}")

    if cfg.save_json:
        json_path = out_dir / "metrics.json"
        json_path.write_text(results.to_json())
        logger.info(f"Metrics written to {json_path}")

        # Per-column effect size CSV
        if results.fidelity and results.fidelity.effect_size:
            es = results.fidelity.effect_size
            csv_path = out_dir / "effect_size_per_column.csv"
            es.to_dataframe().write_csv(csv_path)
            logger.info(f"Per-column effect sizes written to {csv_path}")
            if es.most_divergent:
                logger.info(
                    "Most divergent distributions: "
                    + ", ".join(
                        f"{r.column} ({r.divergence_score:.4f})"
                        for r in es.most_divergent
                    )
                )
            if es.least_divergent:
                logger.info(
                    "Least divergent distributions: "
                    + ", ".join(
                        f"{r.column} ({r.divergence_score:.4f})"
                        for r in es.least_divergent
                    )
                )

        # Sparse-peptide (histogram-imputed) fidelity CSV
        if results.fidelity and results.fidelity.sparse_peptide:
            sp = results.fidelity.sparse_peptide
            sp_csv_path = out_dir / "sparse_peptide_fidelity.csv"
            sp.to_dataframe().write_csv(sp_csv_path)
            logger.info(
                f"Sparse-peptide fidelity ({len(sp.sparse_columns)} columns, "
                f"zero fraction > {sp.zero_threshold:.0%}) written to {sp_csv_path}"
            )
            if sp.most_divergent:
                logger.info(
                    "Sparse peptides most divergent from real: "
                    + ", ".join(
                        f"{r.column} (div={r.divergence_score:.4f})"
                        for r in sp.most_divergent
                    )
                )
            if sp.least_divergent:
                logger.info(
                    "Sparse peptides most similar to real: "
                    + ", ".join(
                        f"{r.column} (div={r.divergence_score:.4f})"
                        for r in sp.least_divergent
                    )
                )

        # Privacy metrics CSV
        if results.privacy is not None:
            privacy_summary = results.privacy.summary()
            if privacy_summary:
                privacy_rows = [{"metric": k, "value": v} for k, v in privacy_summary.items()]
                privacy_csv_path = out_dir / "privacy_metrics.csv"
                pl.DataFrame(privacy_rows, schema={"metric": pl.Utf8, "value": pl.Float64}).write_csv(privacy_csv_path)
                logger.info(f"Privacy metrics written to {privacy_csv_path}")

        # Per-event: also write individual JSON files for convenience
        for event, ev_results in results.per_event.items():
            ev_json_path = out_dir / f"metrics_{event}.json"
            ev_json_path.write_text(ev_results.to_json())
            logger.info(f"Per-event metrics [{event}] written to {ev_json_path}")

            # Per-event per-column effect size CSV
            if ev_results.fidelity and ev_results.fidelity.effect_size:
                ev_csv_path = out_dir / f"effect_size_per_column_{event}.csv"
                ev_results.fidelity.effect_size.to_dataframe().write_csv(ev_csv_path)
                logger.info(
                    f"Per-column effect sizes [{event}] written to {ev_csv_path}"
                )

    summary = results.summary()
    logger.info("=== Evaluation Summary ===")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()

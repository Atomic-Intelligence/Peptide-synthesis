"""Fidelity analysis for sparse (histogram-imputed) peptide columns.

Identifies peptide columns whose real-data zero fraction exceeds a threshold
(assumed to be generated via histogram imputation) and ranks them by how much
the synthetic distribution deviates from the real one.

For each sparse peptide the full distribution — including the mass at zero —
is compared via the Overlap Coefficient (OVL) divergence score (1 − OVL),
so both the sparsity pattern *and* the non-zero values are taken into account.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger

from src.evaluation.fidelity.effect_size import (
    EffectSizeEstimator,
    ColumnEffectSizeResult,
    DEFAULT_CONTINUOUS,
    DEFAULT_CATEGORICAL,
)
from src.evaluation.utils.eval_utils import sparse_peptide_columns


@dataclass
class SparsePeptideFidelityResults:
    """Fidelity results restricted to sparse (histogram-imputed) peptide columns.

    Attributes
    ----------
    sparse_columns :
        Peptide columns whose real-data zero fraction exceeds *zero_threshold*.
    zero_threshold :
        The threshold used to define "sparse" (default 0.4 = 40 %).
    real_zero_pct :
        Per-column zero fraction in the real dataset.
    synth_zero_pct :
        Per-column zero fraction in the synthetic dataset.
    per_column :
        Full effect-size results for every sparse peptide column.
    most_divergent :
        Top-N sparse peptides with the highest divergence score
        (real ↔ synthetic differ the most).
    least_divergent :
        Top-N sparse peptides with the lowest divergence score
        (real ↔ synthetic are most similar).
    """

    sparse_columns: List[str] = field(default_factory=list)
    zero_threshold: float = 0.4

    real_zero_pct: Dict[str, float] = field(default_factory=dict)
    synth_zero_pct: Dict[str, float] = field(default_factory=dict)

    per_column: List[ColumnEffectSizeResult] = field(default_factory=list)

    most_divergent: List[ColumnEffectSizeResult] = field(default_factory=list)
    least_divergent: List[ColumnEffectSizeResult] = field(default_factory=list)

    # ── summary ───────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, float]:
        """Flat dict of headline metrics for mlflow.log_metrics()."""
        out: Dict[str, float] = {
            "fidelity/sparse_peptide/n_sparse_columns": float(len(self.sparse_columns)),
        }
        scores = [r.divergence_score for r in self.per_column if r.divergence_score is not None]
        if scores:
            out["fidelity/sparse_peptide/mean_divergence_score"] = float(np.mean(scores))
        if self.most_divergent and self.most_divergent[0].divergence_score is not None:
            out["fidelity/sparse_peptide/max_divergence_score"] = float(
                self.most_divergent[0].divergence_score
            )
        if self.least_divergent and self.least_divergent[0].divergence_score is not None:
            out["fidelity/sparse_peptide/min_divergence_score"] = float(
                self.least_divergent[0].divergence_score
            )
        return out

    # ── tabular export ────────────────────────────────────────────────────

    def to_dataframe(self) -> pl.DataFrame:
        """Return a Polars DataFrame with per-column metrics.

        Sorted by *divergence_score* descending (most divergent first).
        Columns included:

        - ``column``: peptide column name
        - ``real_zero_pct``: fraction of zeros in the real data
        - ``synth_zero_pct``: fraction of zeros in the synthetic data
        - ``zero_pct_diff``: synth − real zero fraction difference
        - ``divergence_score``: 1 − OVL (composite similarity metric)
        - ``overlap_coef``: histogram overlap coefficient
        - ``cohen_d``, ``rank_biserial``, ``normalized_wasserstein``
        """
        rows = []
        for r in self.per_column:
            real_z = self.real_zero_pct.get(r.column)
            synth_z = self.synth_zero_pct.get(r.column)
            rows.append(
                {
                    "column": r.column,
                    "real_zero_pct": real_z,
                    "synth_zero_pct": synth_z,
                    "zero_pct_diff": (
                        (synth_z - real_z)
                        if real_z is not None and synth_z is not None
                        else None
                    ),
                    "divergence_score": r.divergence_score,
                    "overlap_coef": r.overlap_coef,
                    "cohen_d": r.cohen_d,
                    "rank_biserial": r.rank_biserial,
                    "normalized_wasserstein": r.normalized_wasserstein,
                }
            )
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows).sort(
            "divergence_score", descending=True, nulls_last=True
        )


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class SparsePeptideFidelityEstimator:
    """Identify sparse peptides and rank them by real ↔ synthetic divergence.

    Parameters
    ----------
    zero_fraction_threshold :
        Peptide columns in *real_df* with a zero fraction strictly above this
        value are considered "sparse" (histogram-imputed).  Default is 0.4.
    n_top :
        Number of most- and least-divergent columns to surface.  Default 5.
    """

    def __init__(
        self,
        zero_fraction_threshold: float = 0.4,
        n_top: int = 3,
    ):
        self.zero_fraction_threshold = zero_fraction_threshold
        self.n_top = n_top
        # Only compute the metrics needed for the divergence score
        self._effect_estimator = EffectSizeEstimator(
            continuous_metrics=["overlap_coef", "cohen_d", "rank_biserial", "normalized_wasserstein"],
            categorical_metrics=[],
        )

    def estimate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
    ) -> SparsePeptideFidelityResults:
        """Run the sparse-peptide fidelity analysis.

        Parameters
        ----------
        real_df :
            Real (reference) dataset.
        synth_df :
            Synthetic dataset to evaluate.

        Returns
        -------
        SparsePeptideFidelityResults
        """
        results = SparsePeptideFidelityResults(
            zero_threshold=self.zero_fraction_threshold
        )

        # ── identify sparse peptides (based on real data) ─────────────────
        sparse_cols = sparse_peptide_columns(real_df, self.zero_fraction_threshold)
        # Only keep columns present in both frames
        sparse_cols = [c for c in sparse_cols if c in synth_df.columns]

        results.sparse_columns = sparse_cols

        if not sparse_cols:
            logger.info(
                f"SparsePeptideFidelity — no peptide columns exceed the "
                f"{self.zero_fraction_threshold:.0%} zero fraction threshold."
            )
            return results

        logger.info(
            f"SparsePeptideFidelity — analysing {len(sparse_cols)} sparse peptide columns "
            f"(zero fraction > {self.zero_fraction_threshold:.0%})."
        )

        # ── record zero percentages in both datasets ──────────────────────
        n_real = len(real_df)
        n_synth = len(synth_df)
        for col in sparse_cols:
            results.real_zero_pct[col] = float((real_df[col] == 0).sum() / n_real)
            results.synth_zero_pct[col] = float((synth_df[col] == 0).sum() / n_synth)

        # ── compute effect sizes for sparse columns only ──────────────────
        es_results = self._effect_estimator.estimate(
            real_df, synth_df, columns=sparse_cols
        )
        results.per_column = es_results.per_column

        # ── rank most / least divergent ───────────────────────────────────
        scored = sorted(
            [r for r in es_results.per_column if r.divergence_score is not None],
            key=lambda r: r.divergence_score,  # type: ignore[arg-type]
            reverse=True,
        )
        results.most_divergent = scored[: self.n_top]
        # Filter out constant columns (100% zeros in both datasets) from
        # least-divergent — they are uninformative.
        non_constant = [
            r for r in reversed(scored)
            if not (
                results.real_zero_pct.get(r.column, 0.0) >= 1.0 - 1e-9
                and results.synth_zero_pct.get(r.column, 0.0) >= 1.0 - 1e-9
            )
        ]
        results.least_divergent = non_constant[: self.n_top]

        if results.most_divergent:
            logger.info(
                "Most divergent sparse peptides: "
                + ", ".join(
                    f"{r.column} ({r.divergence_score:.4f})"
                    for r in results.most_divergent
                )
            )
        if results.least_divergent:
            logger.info(
                "Least divergent sparse peptides: "
                + ", ".join(
                    f"{r.column} ({r.divergence_score:.4f})"
                    for r in results.least_divergent
                )
            )

        return results

    # ── visualisation ──────────────────────────────────────────────────────

    def plot(
        self,
        results: SparsePeptideFidelityResults,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        n_show: int = 3,
    ) -> plt.Figure:
        """Produce a summary figure for sparse-peptide fidelity.

        The figure has two sections:

        1. **Ranked bar chart** — all sparse peptides ordered by divergence
           score.  Most-divergent bars are coloured red, least-divergent
           bars green, and the remainder grey.
        2. **Distribution panels** — overlapping histograms for the
           *n_show* most-divergent (top row) and *n_show* least-divergent
           (bottom row) sparse peptides, so the viewer can see both *how*
           and *how much* the distributions differ.

        Parameters
        ----------
        results :
            Output of :meth:`estimate`.
        real_df, synth_df :
            The original DataFrames (needed to draw distributions).
        n_show :
            Number of most- and least-divergent columns to plot.
        """
        if not results.sparse_columns:
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.text(
                0.5, 0.5,
                "No sparse peptide columns found\n"
                f"(threshold: {results.zero_threshold:.0%})",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.axis("off")
            fig.suptitle("Sparse-Peptide Fidelity (Histogram-Imputed)", fontsize=11)
            return fig

        scored = sorted(
            [r for r in results.per_column if r.divergence_score is not None],
            key=lambda r: r.divergence_score,  # type: ignore[arg-type]
            reverse=True,
        )
        cols_ordered = [r.column for r in scored]
        scores = [r.divergence_score for r in scored]

        n_top = min(n_show, len(results.most_divergent))
        n_bot = min(n_show, len(results.least_divergent))
        n_dist_cols = max(n_top, n_bot, 1)
        n_dist_rows = (1 if n_top > 0 else 0) + (1 if n_bot > 0 else 0)

        # ── layout ────────────────────────────────────────────────────────
        fig_height = 3.5 + 3.0 * n_dist_rows
        fig = plt.figure(figsize=(max(8, 3.5 * n_dist_cols), fig_height))
        gs = gridspec.GridSpec(
            1 + n_dist_rows, n_dist_cols,
            height_ratios=[2.5] + [2.8] * n_dist_rows,
            hspace=0.55, wspace=0.35,
        )

        # ── 1. histogram of divergence scores ────────────────────────────
        ax_hist = fig.add_subplot(gs[0, :])

        most_names = {r.column for r in results.most_divergent[:n_top]}
        least_names = {r.column for r in results.least_divergent[:n_bot]}

        scores_arr = np.array(scores)
        most_scores = [s for s, c in zip(scores, cols_ordered) if c in most_names]
        least_scores = [s for s, c in zip(scores, cols_ordered) if c in least_names]

        bins = np.linspace(0, max(scores_arr.max(), 0.01), 60)
        ax_hist.hist(scores_arr, bins=bins, color="#aec7e8", edgecolor="white",
                     linewidth=0.3, label="All columns")
        if most_scores:
            ax_hist.axvline(min(most_scores), color="#d62728", ls="--", lw=1.2,
                            label=f"Most divergent (top {n_top})")
            for s in most_scores:
                ax_hist.axvline(s, color="#d62728", ls="--", lw=0.7, alpha=0.5)
        if least_scores:
            ax_hist.axvline(max(least_scores), color="#2ca02c", ls="--", lw=1.2,
                            label=f"Least divergent (top {n_bot})")
            for s in least_scores:
                ax_hist.axvline(s, color="#2ca02c", ls="--", lw=0.7, alpha=0.5)

        ax_hist.set_xlabel("Divergence score  (1 − overlap coef)", fontsize=9)
        ax_hist.set_ylabel("Number of columns", fontsize=9)
        ax_hist.set_title(
            f"Sparse-Peptide Fidelity — {len(cols_ordered)} columns "
            f"(zero fraction > {results.zero_threshold:.0%})",
            fontsize=10, fontweight="bold",
        )
        ax_hist.legend(fontsize=7, loc="upper right")

        # ── 2. distribution panels ────────────────────────────────────────
        def _plot_dist(ax: plt.Axes, col: str, label: str) -> None:
            real_v = np.log1p(real_df[col].to_numpy().astype(float))
            synth_v = np.log1p(synth_df[col].to_numpy().astype(float))
            lo = min(float(real_v.min()), float(synth_v.min()))
            hi = max(float(real_v.max()), float(synth_v.max()))
            bins = np.linspace(lo, hi, 40) if lo < hi else np.array([lo - 0.5, lo + 0.5])
            ax.hist(real_v, bins=bins, density=True, alpha=0.5,
                    color="darkorange", label="Real")
            ax.hist(synth_v, bins=bins, density=True, alpha=0.5,
                    color="steelblue", label="Synthetic")
            ax.set_xlabel("log1p(value)", fontsize=6)
            r_z = results.real_zero_pct.get(col, float("nan"))
            s_z = results.synth_zero_pct.get(col, float("nan"))
            div = next(
                (r.divergence_score for r in results.per_column if r.column == col),
                None,
            )
            title = f"{col}\n{label}"
            if div is not None:
                title += f"  div={div:.3f}"
            title += f"\nzero%: real={r_z:.1%}, synth={s_z:.1%}"
            ax.set_title(title, fontsize=7)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)

        dist_row = 1
        if n_top > 0:
            for i in range(n_dist_cols):
                ax = fig.add_subplot(gs[dist_row, i])
                if i < n_top:
                    _plot_dist(ax, results.most_divergent[i].column, "most divergent")
                else:
                    ax.axis("off")
            dist_row += 1

        if n_bot > 0:
            for i in range(n_dist_cols):
                ax = fig.add_subplot(gs[dist_row, i])
                if i < n_bot:
                    _plot_dist(ax, results.least_divergent[i].column, "least divergent")
                else:
                    ax.axis("off")

        plt.tight_layout()
        return fig

"""Marginal (univariate) fidelity metrics.

For each column, compares the empirical distribution of real vs synthetic data.

Metrics
-------
Continuous columns
    - Kolmogorov-Smirnov statistic + p-value
    - Wasserstein-1 distance (Earth Mover's Distance)
    - Mean ratio  (synth_mean / real_mean)
    - Std ratio   (synth_std  / real_std)
    - Zero proportion difference

Categorical columns
    - Total Variation Distance (TVD) — half the L1 distance between PMFs
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ColumnMarginalResult:
    column: str
    is_categorical: bool
    ks_statistic: Optional[float] = None
    ks_pvalue: Optional[float] = None
    wasserstein: Optional[float] = None
    mean_ratio: Optional[float] = None
    std_ratio: Optional[float] = None
    zero_diff: Optional[float] = None   # synth_zero_pct - real_zero_pct
    tvd: Optional[float] = None         # for categoricals


@dataclass
class MarginalFidelityResults:
    per_column: List[ColumnMarginalResult] = field(default_factory=list)

    # Aggregates over all continuous columns
    mean_ks: float = 0.0
    mean_wasserstein: float = 0.0
    frac_ks_significant: float = 0.0   # fraction with p-value < 0.05

    # Aggregates over all categorical columns
    mean_tvd: float = 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "fidelity/mean_ks_statistic": self.mean_ks,
            "fidelity/mean_wasserstein": self.mean_wasserstein,
            "fidelity/frac_ks_significant": self.frac_ks_significant,
            "fidelity/mean_tvd": self.mean_tvd,
        }


def _tvd(real_col: np.ndarray, synth_col: np.ndarray) -> float:
    """Half the L1 distance between empirical probability mass functions."""
    categories = np.union1d(np.unique(real_col), np.unique(synth_col))
    real_counts = {c: 0 for c in categories}
    synth_counts = {c: 0 for c in categories}
    for v in real_col:
        real_counts[v] = real_counts.get(v, 0) + 1
    for v in synth_col:
        synth_counts[v] = synth_counts.get(v, 0) + 1
    n_real = len(real_col)
    n_synth = len(synth_col)
    tvd = 0.5 * sum(
        abs(real_counts[c] / n_real - synth_counts[c] / n_synth)
        for c in categories
    )
    return float(tvd)


class MarginalFidelityEstimator:
    """Compute per-column univariate fidelity metrics.

    Parameters
    ----------
    categorical_columns :
        Names of columns that should be treated as categorical.
    ks_alpha :
        Significance level for the KS test.  Default 0.05.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        ks_alpha: float = 0.05,
    ):
        self.categorical_columns: List[str] = categorical_columns or []
        self.ks_alpha = ks_alpha

    def estimate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> MarginalFidelityResults:
        """Compute marginal fidelity for all (or a subset of) columns.

        Parameters
        ----------
        real_df, synth_df :
            DataFrames with identical column sets.
        columns :
            Optional subset of column names to analyse.  If None, all shared
            columns are used.
        """
        shared = list(set(real_df.columns) & set(synth_df.columns))
        if columns is not None:
            shared = [c for c in columns if c in shared]

        per_column: List[ColumnMarginalResult] = []
        ks_stats: List[float] = []
        wass_vals: List[float] = []
        sig_flags: List[bool] = []
        tvd_vals: List[float] = []

        for col in shared:
            is_cat = col in self.categorical_columns
            real_vals = real_df[col].to_numpy()
            synth_vals = synth_df[col].to_numpy()

            if is_cat:
                tvd = _tvd(real_vals.astype(str), synth_vals.astype(str))
                tvd_vals.append(tvd)
                per_column.append(
                    ColumnMarginalResult(column=col, is_categorical=True, tvd=tvd)
                )
            else:
                real_float = real_vals.astype(float)
                synth_float = synth_vals.astype(float)

                ks_stat, ks_p = ks_2samp(real_float, synth_float)
                wass = wasserstein_distance(real_float, synth_float)

                real_mean = float(np.mean(real_float))
                synth_mean = float(np.mean(synth_float))
                real_std = float(np.std(real_float))
                synth_std = float(np.std(synth_float))

                mean_ratio = synth_mean / real_mean if abs(real_mean) > 1e-9 else None
                std_ratio = synth_std / real_std if real_std > 1e-9 else None

                real_zero_pct = float((real_float == 0).mean())
                synth_zero_pct = float((synth_float == 0).mean())
                zero_diff = synth_zero_pct - real_zero_pct

                ks_stats.append(ks_stat)
                wass_vals.append(wass)
                sig_flags.append(ks_p < self.ks_alpha)

                per_column.append(
                    ColumnMarginalResult(
                        column=col,
                        is_categorical=False,
                        ks_statistic=float(ks_stat),
                        ks_pvalue=float(ks_p),
                        wasserstein=float(wass),
                        mean_ratio=mean_ratio,
                        std_ratio=std_ratio,
                        zero_diff=float(zero_diff),
                    )
                )

        results = MarginalFidelityResults(per_column=per_column)
        if ks_stats:
            results.mean_ks = float(np.mean(ks_stats))
            results.mean_wasserstein = float(np.mean(wass_vals))
            results.frac_ks_significant = float(np.mean(sig_flags))
        if tvd_vals:
            results.mean_tvd = float(np.mean(tvd_vals))

        logger.info(
            f"MarginalFidelity — mean KS: {results.mean_ks:.4f}, "
            f"mean Wasserstein: {results.mean_wasserstein:.4f}, "
            f"frac significant: {results.frac_ks_significant:.2%}"
        )
        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_top_divergent(
        self,
        results: MarginalFidelityResults,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        top_n: int = 6,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot KDE of the top-N most divergent continuous columns."""
        continuous = [r for r in results.per_column if not r.is_categorical and r.ks_statistic is not None]
        if not continuous:
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            ax.text(0.5, 0.5, "No continuous columns", ha="center", va="center")
            return fig

        continuous_sorted = sorted(continuous, key=lambda r: r.ks_statistic, reverse=True)
        top = continuous_sorted[:top_n]
        ncols = min(3, len(top))
        nrows = (len(top) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes_flat = np.array(axes).flatten()

        from scipy.stats import gaussian_kde

        for i, res in enumerate(top):
            ax = axes_flat[i]
            real_v = real_df[res.column].to_numpy().astype(float)
            synth_v = synth_df[res.column].to_numpy().astype(float)
            lo = min(real_v.min(), synth_v.min())
            hi = max(real_v.max(), synth_v.max())
            xs = np.linspace(lo, hi, 200)
            if real_v.std() > 0:
                ax.plot(xs, gaussian_kde(real_v)(xs), label="Real", color="darkorange")
            if synth_v.std() > 0:
                ax.plot(xs, gaussian_kde(synth_v)(xs), label="Synthetic", color="steelblue")
            ax.set_title(f"{res.column}\nKS={res.ks_statistic:.3f}", fontsize=8)
            ax.legend(fontsize=7)

        for j in range(len(top), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Top Divergent Marginal Distributions", fontsize=11)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

"""Bootstrapped correlation uncertainty estimation — GPU-accelerated.

For a user-specified set of columns, this module computes 95% (or user-defined)
confidence intervals for each pairwise Pearson/Spearman correlation in a single
dataset.

Key differences from the CPU version
--------------------------------------
* Correlation matrices are computed as normalised matrix products — O(n_cols²)
  work per bootstrap instead of O(n_cols²) individual scipy calls.  For 10 000
  columns this is the difference between ~50 M function calls and a single GPU
  matmul per bootstrap sample.
* GPU acceleration via CuPy when available; falls back to vectorised NumPy on
  CPU (still orders-of-magnitude faster than the old pair-by-pair loop).
* Welford online accumulation of Fisher-z mean and variance across bootstraps —
  only two (n_cols × n_cols) arrays are kept in memory at once instead of
  (n_bootstrap × n_cols × n_cols).
* Parametric CI from normal approximation in z-space.  This is statistically
  valid: the whole point of the Fisher z-transform is that the sampling
  distribution of r becomes approximately normal, so mean ± z·σ gives the same
  CI as bootstrap percentiles without needing to store all samples.
* Only pairs where ``ci_width < |corr|`` are written into the summary table —
  the correlation is larger than its own uncertainty.  For 10 000 columns the
  raw upper triangle contains ~50 M pairs; filtering avoids building a
  500 MB DataFrame from noise.

Outputs
-------
CorrelationUncertaintyResults
    corr_mean    — (n_cols, n_cols) mean bootstrapped correlation matrix
    ci_lower     — lower CI bound matrix
    ci_upper     — upper CI bound matrix
    ci_width     — ci_upper − ci_lower
    column_names — list of analysed columns
    summary_table — polars DataFrame with one row per *significant* pair
                    (ci_width < |corr|)

Memory note
-----------
For n_cols = 10 000 the four full matrices (corr_mean, ci_lower, ci_upper,
ci_width) occupy ~1.6 GB of RAM as float32.  The summary_table contains only
the filtered pairs and is typically much smaller.  Heatmap visualisation is
skipped automatically for n_cols > 100 (individual cells are invisible anyway).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
from loguru import logger


# ── GPU backend ────────────────────────────────────────────────────────────────


def _init_backend() -> Tuple:
    """Return (xp, on_gpu).

    Tries CuPy first, then falls back to NumPy.  The returned *xp* is the
    array module to use; *on_gpu* is a bool indicating whether CuPy is active.
    """
    try:
        import cupy as cp

        cp.zeros(1)  # trigger context initialisation / verify device
        logger.info("GPU backend: CuPy — CUDA acceleration enabled")
        return cp, True
    except Exception as exc:
        logger.info(f"CuPy not available ({exc}); using vectorised NumPy on CPU")
        return np, False


# ── Fisher z helpers ───────────────────────────────────────────────────────────


def _fisher_z(xp, r):
    """Fisher z-transform, clipped away from ±1."""
    return xp.arctanh(xp.clip(r, -0.9999, 0.9999))


def _fisher_z_inv(xp, z):
    return xp.tanh(z)


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class CorrelationUncertaintyResults:
    corr_mean: np.ndarray  # (n_cols, n_cols)
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ci_width: np.ndarray   # ci_upper − ci_lower
    column_names: List[str]
    summary_table: pl.DataFrame  # only pairs where ci_width < |corr|

    def summary(self) -> Dict[str, float]:
        if len(self.summary_table) == 0:
            return {
                "correlation_uncertainty/mean_ci_width": float("nan"),
                "correlation_uncertainty/max_ci_width": float("nan"),
                "correlation_uncertainty/n_significant_pairs": 0,
            }
        return {
            "correlation_uncertainty/mean_ci_width": float(
                self.summary_table["ci_width"].mean()
            ),
            "correlation_uncertainty/max_ci_width": float(
                self.summary_table["ci_width"].max()
            ),
            "correlation_uncertainty/n_significant_pairs": len(self.summary_table),
        }


# ── Column selection helpers ───────────────────────────────────────────────────


def _get_peptide_columns_by_zero_fraction(
    df: pl.DataFrame,
    zero_fraction_range: Tuple[float, float],
    peptide_identifier: str = "Peptide",
) -> List[str]:
    """Return peptide columns whose zero-fraction falls within [lo, hi]."""
    lo, hi = zero_fraction_range
    peptide_cols = [c for c in df.columns if peptide_identifier in c]
    numeric_types = (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    selected = []
    for c in peptide_cols:
        if df[c].dtype not in numeric_types:
            continue
        zf = float((df[c].to_numpy() == 0).mean())
        if lo <= zf <= hi:
            selected.append(c)
    logger.info(
        f"Auto-selected {len(selected)} peptide columns with zero-fraction in [{lo}, {hi}]"
    )
    return selected


# ── GPU-accelerated correlation ────────────────────────────────────────────────


def _rank_columns(xp, X):
    """Column-wise ranks (1-based) via double argsort.

    Parameters
    ----------
    X : array (n_samples, n_cols)

    Returns
    -------
    ranks : array (n_samples, n_cols), dtype float32
    """
    return (xp.argsort(xp.argsort(X, axis=0), axis=0) + 1).astype(xp.float32)


def _pearson_corr_matrix(xp, X):
    """Pearson correlation via normalised matrix product.

    Parameters
    ----------
    X : array (n_samples, n_cols)

    Returns
    -------
    C : array (n_cols, n_cols), values in [-1, 1]
    """
    X_c = X - X.mean(axis=0)
    norms = xp.sqrt((X_c ** 2).sum(axis=0))
    # Avoid division by zero for constant / all-zero columns
    norms = xp.where(norms == 0, xp.ones_like(norms), norms)
    X_n = X_c / norms
    C = X_n.T @ X_n
    C = xp.clip(C, -1.0, 1.0)
    C = xp.nan_to_num(C, nan=0.0)
    return C


def _spearman_corr_matrix(xp, X):
    """Spearman correlation = Pearson on column ranks."""
    return _pearson_corr_matrix(xp, _rank_columns(xp, X))


# ── Estimator ─────────────────────────────────────────────────────────────────


class CorrelationUncertaintyEstimator:
    """Estimate bootstrap confidence intervals for pairwise correlations.

    Parameters
    ----------
    method : "spearman" | "pearson"
    n_bootstrap : int
        Number of bootstrap samples.  Default 1000.
    confidence_level : float
        CI level.  Default 0.95 (→ ±1.96 σ in z-space).
    max_columns : int
        Column cap when not using zero_fraction_range.
    """

    def __init__(
        self,
        method: Literal["spearman", "pearson"] = "spearman",
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        max_columns: int = 50,
    ):
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.max_columns = max_columns

    # ── column selection ───────────────────────────────────────────────────

    def _select_columns(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]],
        zero_fraction_range: Optional[Tuple[float, float]] = None,
        peptide_identifier: str = "Peptide",
    ) -> List[str]:
        if zero_fraction_range is not None:
            return _get_peptide_columns_by_zero_fraction(
                df, zero_fraction_range, peptide_identifier
            )

        numeric = [
            c
            for c in df.columns
            if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        ]
        if columns:
            numeric = [c for c in columns if c in numeric]
        if len(numeric) > self.max_columns:
            zero_fracs = {c: float((df[c].to_numpy() == 0).mean()) for c in numeric}
            numeric = sorted(numeric, key=lambda c: zero_fracs[c])[: self.max_columns]
        return numeric

    # ── bootstrap loop ────────────────────────────────────────────────────

    def _bootstrap_streaming(
        self,
        xp,
        on_gpu: bool,
        data_gpu,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulate Fisher-z mean and variance across bootstraps (Welford).

        Uses Welford's online algorithm so memory usage is O(n_cols²) regardless
        of n_bootstrap — only z_mean and z_M2 are kept in memory at any time.

        Returns
        -------
        z_mean : np.ndarray (n_cols, n_cols)
        z_var  : np.ndarray (n_cols, n_cols)  biased population variance
        """
        n, n_cols = data_gpu.shape
        corr_fn = (
            _spearman_corr_matrix if self.method == "spearman" else _pearson_corr_matrix
        )

        z_mean = xp.zeros((n_cols, n_cols), dtype=xp.float32)
        z_M2 = xp.zeros((n_cols, n_cols), dtype=xp.float32)

        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)

            if on_gpu:
                import cupy as cp

                boot = data_gpu[cp.asarray(idx)]
            else:
                boot = data_gpu[idx]

            C = corr_fn(xp, boot)
            z = _fisher_z(xp, C)

            # Welford update
            delta = z - z_mean
            z_mean = z_mean + delta / (b + 1)
            delta2 = z - z_mean
            z_M2 = z_M2 + delta * delta2

            if (b + 1) % 100 == 0:
                logger.info(f"  Bootstrap {b + 1}/{self.n_bootstrap}")

        # Population variance (biased); fine for large n_bootstrap
        z_var = z_M2 / self.n_bootstrap

        if on_gpu:
            import cupy as cp

            z_mean = cp.asnumpy(z_mean)
            z_var = cp.asnumpy(z_var)

        return z_mean, z_var

    # ── main estimation ────────────────────────────────────────────────────

    def estimate(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        zero_fraction_range: Optional[Tuple[float, float]] = None,
        peptide_identifier: str = "Peptide",
        random_seed: int = 42,
    ) -> CorrelationUncertaintyResults:
        cols = self._select_columns(df, columns, zero_fraction_range, peptide_identifier)
        if len(cols) < 2:
            raise ValueError(
                "Need at least 2 numeric columns for correlation uncertainty."
            )

        n_cols = len(cols)
        n_pairs = n_cols * (n_cols - 1) // 2
        logger.info(
            f"CorrelationUncertainty: {n_cols} columns ({n_pairs:,} pairs), "
            f"{self.n_bootstrap} bootstraps, method={self.method}, "
            f"CI={self.confidence_level:.0%}"
        )

        data = df.select(cols).to_numpy().astype(np.float32)
        rng = np.random.default_rng(random_seed)

        xp, on_gpu = _init_backend()

        if on_gpu:
            import cupy as cp

            data_gpu = cp.asarray(data)
        else:
            data_gpu = data

        logger.info("Bootstrapping (Welford streaming — constant memory)...")
        z_mean, z_var = self._bootstrap_streaming(xp, on_gpu, data_gpu, rng)

        # ── CI from normal approximation in z-space ────────────────────────
        # Valid because Fisher z is approximately normal; equivalent to
        # bootstrap percentile CIs without storing all bootstrap samples.
        alpha = 1 - self.confidence_level
        z_alpha = float(scipy_norm.ppf(1 - alpha / 2))  # e.g. 1.96 for 95 %

        z_std = np.sqrt(np.maximum(z_var, 0.0))
        ci_lo_z = z_mean - z_alpha * z_std
        ci_hi_z = z_mean + z_alpha * z_std

        corr_mean = np.tanh(z_mean)
        ci_lo = np.tanh(ci_lo_z)
        ci_hi = np.tanh(ci_hi_z)
        ci_width = ci_hi - ci_lo

        # ── filter: keep only pairs where ci_width < |corr| ───────────────
        triu_i, triu_j = np.triu_indices(n_cols, k=1)
        corr_vals = corr_mean[triu_i, triu_j]
        ci_lo_vals = ci_lo[triu_i, triu_j]
        ci_hi_vals = ci_hi[triu_i, triu_j]
        width_vals = ci_width[triu_i, triu_j]

        significant = width_vals < np.abs(corr_vals)
        sig_i = triu_i[significant]
        sig_j = triu_j[significant]

        logger.info(
            f"Filtering: {significant.sum():,} / {n_pairs:,} pairs "
            f"have ci_width < |corr|  ({100.0 * significant.mean():.1f} %)"
        )

        summary_table = pl.DataFrame(
            {
                "col_a": [cols[i] for i in sig_i],
                "col_b": [cols[j] for j in sig_j],
                "corr": corr_vals[significant].tolist(),
                "ci_lower": ci_lo_vals[significant].tolist(),
                "ci_upper": ci_hi_vals[significant].tolist(),
                "ci_width": width_vals[significant].tolist(),
            }
        )

        if len(summary_table) > 0:
            logger.info(
                f"CorrelationUncertainty complete — {len(summary_table):,} significant pairs, "
                f"mean CI width = {summary_table['ci_width'].mean():.4f}"
            )
        else:
            logger.info("CorrelationUncertainty complete — 0 significant pairs")

        return CorrelationUncertaintyResults(
            corr_mean=corr_mean,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ci_width=ci_width,
            column_names=cols,
            summary_table=summary_table,
        )

    # ── visualisation ──────────────────────────────────────────────────────

    def plot(
        self,
        results: CorrelationUncertaintyResults,
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """Two heatmaps: mean correlations and CI width (uncertainty).

        Skipped automatically for n_cols > 100 — individual cells are invisible
        at that scale and seaborn becomes prohibitively slow.
        """
        import seaborn as sns

        n = len(results.column_names)
        if n > 100:
            logger.warning(
                f"Skipping heatmap: {n} columns is too large to visualise meaningfully "
                f"(limit is 100).  Use the summary_table CSV instead."
            )
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        tick_labels = results.column_names if n <= 25 else []

        sns.heatmap(
            results.corr_mean,
            ax=axes[0],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            square=True,
            linewidths=0,
            cbar_kws={"label": f"{self.method.capitalize()} correlation"},
        )
        axes[0].set_title("Mean bootstrapped correlation")

        sns.heatmap(
            results.ci_width,
            ax=axes[1],
            cmap="YlOrRd",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            square=True,
            linewidths=0,
            cbar_kws={"label": "CI width"},
        )
        axes[1].set_title(f"CI width ({self.confidence_level:.0%} bootstrap CI)")

        fig.suptitle(
            f"Correlation Uncertainty  |  {self.method.capitalize()}  |  "
            f"{self.n_bootstrap} bootstraps  |  CI={self.confidence_level:.0%}",
            fontsize=11,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig


# ── Standalone entry point ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrapped correlation uncertainty for a single dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset CSV.")
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory for plot and CSV summary."
    )
    parser.add_argument(
        "--method", type=str, default="spearman", choices=["spearman", "pearson"]
    )
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument(
        "--max-columns",
        type=int,
        default=50,
        help="Column cap when not using --zero-fraction-range.",
    )
    parser.add_argument("--columns", nargs="*", default=None)
    parser.add_argument(
        "--zero-fraction-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=(
            "Auto-select peptide columns whose fraction of zeros falls in [LO, HI]. "
            "e.g. --zero-fraction-range 0.0 0.5. "
            "When set, --columns and --max-columns are ignored."
        ),
    )
    parser.add_argument("--peptide-identifier", type=str, default="Peptide")
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    zero_fraction_range: Optional[Tuple[float, float]] = None
    if args.zero_fraction_range is not None:
        zero_fraction_range = (args.zero_fraction_range[0], args.zero_fraction_range[1])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {args.data}")
    df = pl.read_csv(args.data)

    estimator = CorrelationUncertaintyEstimator(
        method=args.method,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        max_columns=args.max_columns,
    )

    results = estimator.estimate(
        df=df,
        columns=args.columns,
        zero_fraction_range=zero_fraction_range,
        peptide_identifier=args.peptide_identifier,
        random_seed=args.seed,
    )

    csv_path = output_dir / "correlation_uncertainty_summary.csv"
    results.summary_table.write_csv(csv_path)
    logger.info(f"Summary table saved to {csv_path}")

    plot_path = output_dir / "correlation_uncertainty.png"
    estimator.plot(results, save_path=str(plot_path))

    for k, v in results.summary().items():
        logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main()

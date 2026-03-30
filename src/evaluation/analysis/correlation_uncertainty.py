"""Bootstrapped correlation uncertainty estimation.

For a user-specified set of columns, this module computes 95% (or user-defined)
confidence intervals for each pairwise Pearson/Spearman correlation in a single
dataset.  The CI width reflects how reliably each correlation can be estimated
given the sample size — wide CIs indicate pairs where the correlation is
uncertain, narrow CIs indicate well-estimated relationships.

Steps
-----
1. Bootstrap the dataset B times → distribution of correlation estimates per pair.
2. Apply Fisher z-transform before computing percentile CIs, then back-transform.

Outputs
-------
CorrelationUncertaintyResults
    - corr_mean    : mean bootstrapped correlation matrix
    - ci_lower     : lower CI bound matrix
    - ci_upper     : upper CI bound matrix
    - ci_width     : ci_upper - ci_lower
    - column_names : list of analysed columns
    - summary_table: polars DataFrame with one row per column pair

Visualisations
--------------
- Heatmap of mean correlations
- Heatmap of CI width (uncertainty)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass
from loguru import logger
from rich.progress import track

# ---------------------------------------------------------------------------
# Fisher z helpers
# ---------------------------------------------------------------------------


def _fisher_z(r: np.ndarray) -> np.ndarray:
    """Fisher z-transform:  z = atanh(r), clipped away from ±1."""
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def _fisher_z_inv(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CorrelationUncertaintyResults:
    corr_mean: np.ndarray  # (n_cols, n_cols)
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ci_width: np.ndarray  # ci_upper - ci_lower
    column_names: List[str]
    summary_table: pl.DataFrame  # one row per pair

    def summary(self) -> Dict[str, float]:
        mask = np.triu(np.ones_like(self.ci_width, dtype=bool), k=1)
        return {
            "correlation_uncertainty/mean_ci_width": float(self.ci_width[mask].mean()),
            "correlation_uncertainty/max_ci_width": float(self.ci_width[mask].max()),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_peptide_columns_by_zero_fraction(
    df: pl.DataFrame,
    zero_fraction_range: Tuple[float, float],
    peptide_identifier: str = "Peptide",
) -> List[str]:
    """Return peptide columns whose zero-fraction falls within [lo, hi].

    Parameters
    ----------
    df :
        The dataset to inspect.
    zero_fraction_range :
        A ``(lo, hi)`` tuple, e.g. ``(0.0, 0.5)``.  Both bounds are inclusive.
    peptide_identifier :
        Sub-string used to detect peptide columns (default ``"Peptide"``).
    """
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


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class CorrelationUncertaintyEstimator:
    """Estimate bootstrap confidence intervals for pairwise correlations.

    Parameters
    ----------
    method :
        ``"spearman"`` or ``"pearson"``.
    n_bootstrap :
        Number of bootstrap samples.  Default 1000.
    confidence_level :
        CI level.  Default 0.95 (→ 2.5th and 97.5th percentiles).
    max_columns :
        Cap to avoid O(n²) memory explosion.  Columns with the fewest zeros
        are prioritised when trimming an explicit ``columns`` list.
        Not applied when ``zero_fraction_range`` is used.
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_columns(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]],
        zero_fraction_range: Optional[Tuple[float, float]] = None,
        peptide_identifier: str = "Peptide",
    ) -> List[str]:
        """Resolve which columns to analyse.

        Priority:
        1. If *zero_fraction_range* is given → auto-select peptide columns by
           sparsity; *columns* and *max_columns* are ignored.
        2. If *columns* is given → use that explicit list (trimmed to
           *max_columns* by lowest zero-fraction if needed).
        3. Otherwise → all numeric columns, trimmed to *max_columns*.
        """
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

    def _corr_matrix(self, data: np.ndarray) -> np.ndarray:
        n_cols = data.shape[1]
        C = np.eye(n_cols)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                if self.method == "spearman":
                    r, _ = spearmanr(data[:, i], data[:, j])
                else:
                    r, _ = pearsonr(data[:, i], data[:, j])
                if np.isnan(r):
                    r = 0.0
                C[i, j] = C[j, i] = r
        return C

    def _bootstrap_corr_distribution(
        self, data: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Return array of shape (n_bootstrap, n_cols, n_cols)."""
        n = data.shape[0]
        n_cols = data.shape[1]
        boot_corrs = np.empty((self.n_bootstrap, n_cols, n_cols))
        for b in track(range(self.n_bootstrap), description=f"Bootrstapping..."):
            idx = rng.integers(0, n, size=n)
            boot_corrs[b] = self._corr_matrix(data[idx])
        return boot_corrs

    # ------------------------------------------------------------------
    # Main estimation
    # ------------------------------------------------------------------

    def estimate(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        zero_fraction_range: Optional[Tuple[float, float]] = None,
        peptide_identifier: str = "Peptide",
        random_seed: int = 42,
    ) -> CorrelationUncertaintyResults:
        cols = self._select_columns(
            df, columns, zero_fraction_range, peptide_identifier
        )
        if len(cols) < 2:
            raise ValueError(
                "Need at least 2 numeric columns for correlation uncertainty."
            )

        logger.info(
            f"CorrelationUncertainty: {len(cols)} columns, {self.n_bootstrap} bootstraps, "
            f"method={self.method}, CI={self.confidence_level:.0%}"
        )

        data = df.select(cols).to_numpy().astype(float)
        rng = np.random.default_rng(random_seed)

        logger.info("Bootstrapping...")
        boot_corrs = self._bootstrap_corr_distribution(data, rng)

        alpha = 1 - self.confidence_level
        lo_pct = 100 * alpha / 2
        hi_pct = 100 * (1 - alpha / 2)

        z = _fisher_z(boot_corrs)
        corr_mean = _fisher_z_inv(z.mean(axis=0))
        ci_lo = _fisher_z_inv(np.percentile(z, lo_pct, axis=0))
        ci_hi = _fisher_z_inv(np.percentile(z, hi_pct, axis=0))
        ci_width = ci_hi - ci_lo

        n_cols = len(cols)
        rows = []
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                rows.append(
                    {
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "corr": float(corr_mean[i, j]),
                        "ci_lower": float(ci_lo[i, j]),
                        "ci_upper": float(ci_hi[i, j]),
                        "ci_width": float(ci_width[i, j]),
                    }
                )

        summary_table = pl.DataFrame(rows)
        logger.info(
            f"CorrelationUncertainty complete — {len(rows)} pairs, "
            f"mean CI width = {summary_table['ci_width'].mean():.4f}"
        )

        return CorrelationUncertaintyResults(
            corr_mean=corr_mean,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ci_width=ci_width,
            column_names=cols,
            summary_table=summary_table,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        results: CorrelationUncertaintyResults,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Two heatmaps: mean correlations and CI width (uncertainty)."""
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        n = len(results.column_names)
        tick_labels = results.column_names if n <= 25 else []

        # --- Mean correlation heatmap ---
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

        # --- CI width heatmap ---
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


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrapped correlation uncertainty for a single dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the dataset CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where the plot and CSV summary are saved.",
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
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Explicit list of column names to analyse.",
    )
    parser.add_argument(
        "--zero-fraction-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=(
            "Auto-select peptide columns whose fraction of zeros falls "
            "in [LO, HI].  e.g. --zero-fraction-range 0.0 0.5. "
            "When set, --columns and --max-columns are ignored."
        ),
    )
    parser.add_argument(
        "--peptide-identifier",
        type=str,
        default="Peptide",
        help="Sub-string used to identify peptide columns.",
    )
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
    logger.info(f"Plot saved to {plot_path}")

    for k, v in results.summary().items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

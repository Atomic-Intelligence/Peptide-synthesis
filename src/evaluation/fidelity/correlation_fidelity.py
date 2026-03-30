"""Correlation fidelity metrics.

Compares the pairwise correlation structure of real vs synthetic data.

Metrics
-------
- Frobenius norm of the correlation matrix difference: ||C_real - C_synth||_F
- Maximum absolute correlation error (max over all pairs)
- Mean absolute correlation error (MACE)
- Heatmaps: real, synthetic, and difference matrices
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class CorrelationFidelityResults:
    real_corr: np.ndarray           # shape (n_cols, n_cols)
    synth_corr: np.ndarray
    diff_corr: np.ndarray           # real_corr - synth_corr
    column_names: List[str]

    frobenius_norm: float
    max_abs_error: float
    mean_abs_error: float           # MACE

    def summary(self) -> Dict[str, float]:
        return {
            "fidelity/correlation_frobenius": self.frobenius_norm,
            "fidelity/correlation_max_abs_error": self.max_abs_error,
            "fidelity/correlation_mace": self.mean_abs_error,
        }


class CorrelationFidelityEstimator:
    """Compute correlation matrix fidelity between real and synthetic data.

    Parameters
    ----------
    method :
        ``"pearson"`` or ``"spearman"``.  Spearman is more robust to the
        heavy-tailed marginal distributions typical of peptide data.
    max_columns :
        Cap the number of columns to avoid O(n²) blowup.  The columns with
        the most non-zero values are selected when the cap is active.
    """

    def __init__(
        self,
        method: str = "spearman",
        max_columns: int = 60,
    ):
        assert method in ("pearson", "spearman"), "method must be 'pearson' or 'spearman'"
        self.method = method
        self.max_columns = max_columns

    def _select_columns(
        self, real_df: pl.DataFrame, synth_df: pl.DataFrame, columns: Optional[List[str]]
    ) -> List[str]:
        shared = list(set(real_df.columns) & set(synth_df.columns))
        if columns is not None:
            shared = [c for c in columns if c in shared]
        # Keep only numeric columns
        numeric = [c for c in shared if real_df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
        if len(numeric) > self.max_columns:
            # Prioritise columns with fewest zeros (most informative)
            zero_fracs = {
                c: float((real_df[c].to_numpy() == 0).mean()) for c in numeric
            }
            numeric = sorted(numeric, key=lambda c: zero_fracs[c])[: self.max_columns]
        return numeric

    def _corr_matrix(self, df: pl.DataFrame, columns: List[str]) -> np.ndarray:
        data = df.select(columns).to_numpy().astype(float)
        if self.method == "spearman":
            from scipy.stats import spearmanr
            corr, _ = spearmanr(data)
            if data.shape[1] == 1:
                return np.array([[1.0]])
            return np.array(corr)
        else:
            return np.corrcoef(data.T)

    def estimate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> CorrelationFidelityResults:
        cols = self._select_columns(real_df, synth_df, columns)
        if len(cols) < 2:
            logger.warning("CorrelationFidelity requires at least 2 numeric columns.")
            empty = np.array([[]])
            return CorrelationFidelityResults(
                real_corr=empty, synth_corr=empty, diff_corr=empty,
                column_names=cols, frobenius_norm=0.0,
                max_abs_error=0.0, mean_abs_error=0.0,
            )

        logger.info(f"Computing {self.method} correlation matrix over {len(cols)} columns.")
        C_real = self._corr_matrix(real_df, cols)
        C_synth = self._corr_matrix(synth_df, cols)

        diff = C_real - C_synth
        # Use only upper triangle (excluding diagonal) to avoid double-counting
        mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
        off_diag = diff[mask]

        frob = float(np.linalg.norm(diff, "fro"))
        max_err = float(np.abs(off_diag).max())
        mace = float(np.abs(off_diag).mean())

        logger.info(
            f"CorrelationFidelity — Frobenius: {frob:.4f}, MACE: {mace:.4f}, "
            f"max error: {max_err:.4f}"
        )

        return CorrelationFidelityResults(
            real_corr=C_real,
            synth_corr=C_synth,
            diff_corr=diff,
            column_names=cols,
            frobenius_norm=frob,
            max_abs_error=max_err,
            mean_abs_error=mace,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        results: CorrelationFidelityResults,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Side-by-side heatmaps: real, synthetic, difference."""
        if results.real_corr.size == 0:
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            ax.text(0.5, 0.5, "Insufficient columns", ha="center", va="center")
            return fig

        n = len(results.column_names)
        tick_labels = results.column_names if n <= 30 else []

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        kw = dict(vmin=-1, vmax=1, cmap="coolwarm", square=True, linewidths=0)

        import seaborn as sns
        for ax, matrix, title in [
            (axes[0], results.real_corr, "Real"),
            (axes[1], results.synth_corr, "Synthetic"),
        ]:
            sns.heatmap(
                matrix, ax=ax, xticklabels=tick_labels, yticklabels=tick_labels,
                **kw
            )
            ax.set_title(f"{title} correlation ({results.__class__.__name__})")

        diff_kw = dict(vmin=-0.5, vmax=0.5, cmap="RdBu", square=True, linewidths=0)
        sns.heatmap(
            results.diff_corr, ax=axes[2],
            xticklabels=tick_labels, yticklabels=tick_labels,
            **diff_kw,
        )
        axes[2].set_title(
            f"Difference (Real − Synth)\nFrob={results.frobenius_norm:.3f}  "
            f"MACE={results.mean_abs_error:.3f}"
        )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

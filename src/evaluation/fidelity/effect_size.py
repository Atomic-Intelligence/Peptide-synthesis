"""Effect size metrics for comparing real vs. synthetic distributions.

For each column, quantifies the *practical magnitude* of distribution shift
independent of sample size (unlike p-values).

Continuous columns
------------------
cohen_d
    Standardized mean difference (pooled-std). |d| ≥ 0.2 small, 0.5 medium,
    0.8 large (Cohen 1988).
rank_biserial
    Rank-biserial correlation from the Mann-Whitney U statistic.  Equivalent
    to Cliff's delta.  Range [-1, 1]; 0 = no shift.
median_abs_shift
    Normalized median shift: (median_synth − median_real) / MAD_real.
    Robust to outliers; scale-free.
cles
    Common Language Effect Size: P(synth > real).  0.5 = no effect; 1.0 =
    synthetic always larger.
overlap_coef
    Histogram-based Weitzman Overlap Coefficient (OVL).  1.0 = identical
    distributions; 0.0 = disjoint.
normalized_wasserstein
    Wasserstein-1 distance divided by the real IQR — makes Wasserstein
    comparable across columns with different scales.

Categorical columns
-------------------
hellinger
    Hellinger distance between empirical PMFs.  Range [0, 1].
js_divergence
    Jensen-Shannon divergence (base-2) between empirical PMFs.  Range [0, 1].
"""

from __future__ import annotations

import numpy as np
import polars as pl
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger
from scipy.stats import mannwhitneyu, wasserstein_distance, iqr


# ---------------------------------------------------------------------------
# Valid metric names
# ---------------------------------------------------------------------------

CONTINUOUS_METRICS: frozenset[str] = frozenset(
    [
        "cohen_d",
        "rank_biserial",
        "median_abs_shift",
        "cles",
        "overlap_coef",
        "normalized_wasserstein",
    ]
)

CATEGORICAL_METRICS: frozenset[str] = frozenset(
    [
        "hellinger",
        "js_divergence",
    ]
)

DEFAULT_CONTINUOUS: List[str] = [
    "cohen_d",
    "rank_biserial",
    "median_abs_shift",
    "cles",
    "overlap_coef",
    "normalized_wasserstein",
]

DEFAULT_CATEGORICAL: List[str] = [
    "hellinger",
    "js_divergence",
]


# ---------------------------------------------------------------------------
# Low-level helpers — all operate on 1-D numpy float arrays
# ---------------------------------------------------------------------------


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-std Cohen's d: (mean_b − mean_a) / pooled_std.

    a = real, b = synthetic.  Positive value → synthetic mean is higher.
    """
    n_a, n_b = len(a), len(b)
    var_a = float(np.var(a, ddof=1)) if n_a > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if n_b > 1 else 0.0
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((b.mean() - a.mean()) / pooled_std)


def _rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation via Mann-Whitney U.

    r = 1 − 2·U / (n_a · n_b).  Equivalent to Cliff's delta.
    Range [-1, 1]; positive → synthetic tends to be larger.
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    u_stat, _ = mannwhitneyu(b, a, alternative="two-sided")
    return float(1.0 - 2.0 * u_stat / (n_a * n_b))


def _median_abs_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized median shift: (median_b − median_a) / (MAD_a + ε).

    Uses the median absolute deviation of the *real* data as the scale
    factor, making it robust to outliers.
    """
    mad_a = float(np.median(np.abs(a - np.median(a))))
    if mad_a < 1e-12:
        return 0.0
    return float((np.median(b) - np.median(a)) / mad_a)


def _cles(a: np.ndarray, b: np.ndarray) -> float:
    """Common Language Effect Size: P(synth > real).

    Uses the Mann-Whitney U statistic.  0.5 = no effect.
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.5
    # mannwhitneyu(..., alternative='greater') gives U = # pairs where b > a
    u_stat, _ = mannwhitneyu(b, a, alternative="greater")
    return float(u_stat / (n_a * n_b))


def _overlap_coef(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    """Weitzman Overlap Coefficient (OVL) via histogram.

    Approximates ∫ min(f_real, f_synth) dx.  1.0 = identical distributions.
    """
    lo = min(float(a.min()), float(b.min()))
    hi = max(float(a.max()), float(b.max()))
    if lo >= hi:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    bin_width = edges[1] - edges[0]
    hist_a = np.histogram(a, bins=edges, density=True)[0] * bin_width
    hist_b = np.histogram(b, bins=edges, density=True)[0] * bin_width
    return float(np.sum(np.minimum(hist_a, hist_b)))


def _normalized_wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 distance divided by the real distribution's IQR.

    Makes the metric scale-free so it is comparable across columns.
    """
    w = wasserstein_distance(a, b)

    iqr_a = float(iqr(a))
    if iqr_a < 1e-12:
        return 0.0

    return float(w / iqr_a)


# ---------------------------------------------------------------------------
# Categorical helpers
# ---------------------------------------------------------------------------


def _aligned_pmfs(
    col_a: np.ndarray, col_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute aligned empirical PMFs (same category order)."""
    categories = np.union1d(np.unique(col_a), np.unique(col_b))
    n_a, n_b = len(col_a), len(col_b)

    def freq(arr: np.ndarray) -> np.ndarray:
        counts = {c: 0 for c in categories}
        for v in arr:
            counts[v] = counts.get(v, 0) + 1
        return np.array(
            [counts[c] / n_a if arr is col_a else counts[c] / n_b for c in categories],
            dtype=float,
        )

    # rebuild cleanly to avoid closure bug
    a_counts = {c: 0 for c in categories}
    b_counts = {c: 0 for c in categories}
    for v in col_a:
        a_counts[v] = a_counts.get(v, 0) + 1
    for v in col_b:
        b_counts[v] = b_counts.get(v, 0) + 1

    pmf_a = np.array([a_counts[c] / n_a for c in categories], dtype=float)
    pmf_b = np.array([b_counts[c] / n_b for c in categories], dtype=float)
    return pmf_a, pmf_b


def _hellinger(pmf_a: np.ndarray, pmf_b: np.ndarray) -> float:
    """Hellinger distance: sqrt(½ · Σ(√p_i − √q_i)²).  Range [0, 1]."""
    return float(np.sqrt(0.5 * np.sum((np.sqrt(pmf_a) - np.sqrt(pmf_b)) ** 2)))


def _js_divergence(pmf_a: np.ndarray, pmf_b: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2 nats, range [0, 1])."""
    eps = 1e-12
    m = 0.5 * (pmf_a + pmf_b)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > eps
        return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], eps))))

    return float(0.5 * _kl(pmf_a, m) + 0.5 * _kl(pmf_b, m))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ColumnEffectSizeResult:
    """Per-column effect size metrics."""

    column: str
    is_categorical: bool

    # Continuous
    cohen_d: Optional[float] = None
    rank_biserial: Optional[float] = None
    median_abs_shift: Optional[float] = None
    cles: Optional[float] = None
    overlap_coef: Optional[float] = None
    normalized_wasserstein: Optional[float] = None

    # Categorical
    hellinger: Optional[float] = None
    js_divergence: Optional[float] = None

    # Composite divergence score (0 = identical, 1 = maximally divergent)
    # Continuous: 1 - overlap_coef; Categorical: hellinger
    divergence_score: Optional[float] = None


@dataclass
class EffectSizeResults:
    """Aggregate and per-column effect size results."""

    per_column: List[ColumnEffectSizeResult] = field(default_factory=list)
    enabled_continuous: List[str] = field(default_factory=list)
    enabled_categorical: List[str] = field(default_factory=list)

    # Aggregate means (mean absolute value for signed metrics)
    mean_cohen_d: Optional[float] = None
    mean_rank_biserial: Optional[float] = None
    mean_median_abs_shift: Optional[float] = None
    mean_cles: Optional[float] = None
    mean_overlap_coef: Optional[float] = None
    mean_normalized_wasserstein: Optional[float] = None
    mean_hellinger: Optional[float] = None
    mean_js_divergence: Optional[float] = None

    # Top-5 most / least divergent columns by divergence_score
    most_divergent: List[ColumnEffectSizeResult] = field(default_factory=list)
    least_divergent: List[ColumnEffectSizeResult] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        """Flat dict of headline effect size metrics for mlflow.log_metrics()."""
        out: Dict[str, float] = {}
        for metric in self.enabled_continuous:
            val = getattr(self, f"mean_{metric}", None)
            if val is not None:
                out[f"fidelity/effect_size/{metric}"] = val
        for metric in self.enabled_categorical:
            val = getattr(self, f"mean_{metric}", None)
            if val is not None:
                out[f"fidelity/effect_size/{metric}"] = val
        return out

    def to_dataframe(self) -> pl.DataFrame:
        """Return a Polars DataFrame with one row per column and all metrics.

        Sorted by ``divergence_score`` descending (most divergent first).
        """
        rows = []
        for r in self.per_column:
            rows.append(
                {
                    "column": r.column,
                    "type": "categorical" if r.is_categorical else "continuous",
                    "divergence_score": r.divergence_score,
                    "cohen_d": r.cohen_d,
                    "rank_biserial": r.rank_biserial,
                    "median_abs_shift": r.median_abs_shift,
                    "cles": r.cles,
                    "overlap_coef": r.overlap_coef,
                    "normalized_wasserstein": r.normalized_wasserstein,
                    "hellinger": r.hellinger,
                    "js_divergence": r.js_divergence,
                }
            )
        if not rows:
            return pl.DataFrame()
        schema = {
            "column": pl.Utf8,
            "type": pl.Utf8,
            "divergence_score": pl.Float64,
            "cohen_d": pl.Float64,
            "rank_biserial": pl.Float64,
            "median_abs_shift": pl.Float64,
            "cles": pl.Float64,
            "overlap_coef": pl.Float64,
            "normalized_wasserstein": pl.Float64,
            "hellinger": pl.Float64,
            "js_divergence": pl.Float64,
        }
        return pl.DataFrame(rows, schema=schema).sort(
            "divergence_score", descending=True, nulls_last=True
        )


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class EffectSizeEstimator:
    """Compute per-column effect size metrics between real and synthetic data.

    Parameters
    ----------
    continuous_metrics :
        Which continuous effect size metrics to compute.  Must be a subset of
        ``CONTINUOUS_METRICS``.  Defaults to all available metrics.
    categorical_metrics :
        Which categorical effect size metrics to compute.  Must be a subset of
        ``CATEGORICAL_METRICS``.  Defaults to all available metrics.
    categorical_columns :
        Column names to treat as categorical.
    """

    def __init__(
        self,
        continuous_metrics: Optional[List[str]] = None,
        categorical_metrics: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ):
        self.continuous_metrics: List[str] = (
            list(continuous_metrics)
            if continuous_metrics is not None
            else DEFAULT_CONTINUOUS
        )
        self.categorical_metrics: List[str] = (
            list(categorical_metrics)
            if categorical_metrics is not None
            else DEFAULT_CATEGORICAL
        )
        self.categorical_columns: List[str] = categorical_columns or []

        unknown_cont = set(self.continuous_metrics) - CONTINUOUS_METRICS
        unknown_cat = set(self.categorical_metrics) - CATEGORICAL_METRICS
        if unknown_cont:
            raise ValueError(
                f"Unknown continuous effect size metrics: {unknown_cont}. "
                f"Valid options: {sorted(CONTINUOUS_METRICS)}"
            )
        if unknown_cat:
            raise ValueError(
                f"Unknown categorical effect size metrics: {unknown_cat}. "
                f"Valid options: {sorted(CATEGORICAL_METRICS)}"
            )

    def estimate(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> EffectSizeResults:
        """Compute effect sizes for all (or a subset of) columns.

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

        per_column: List[ColumnEffectSizeResult] = []
        # Accumulators: store raw values; compute mean at the end
        agg: Dict[str, List[float]] = {
            m: [] for m in self.continuous_metrics + self.categorical_metrics
        }

        for col in shared:
            is_cat = col in self.categorical_columns
            real_vals = real_df[col].to_numpy()
            synth_vals = synth_df[col].to_numpy()
            result = ColumnEffectSizeResult(column=col, is_categorical=is_cat)

            if is_cat:
                pmf_a, pmf_b = _aligned_pmfs(
                    real_vals.astype(str), synth_vals.astype(str)
                )
                for metric in self.categorical_metrics:
                    if metric == "hellinger":
                        val = _hellinger(pmf_a, pmf_b)
                    elif metric == "js_divergence":
                        val = _js_divergence(pmf_a, pmf_b)
                    else:
                        continue
                    setattr(result, metric, val)
                    agg[metric].append(val)
            else:
                a = real_vals.astype(float)
                b = synth_vals.astype(float)
                a = a[~np.isnan(a)]
                b = b[~np.isnan(b)]

                if len(a) < 2 or len(b) < 2:
                    per_column.append(result)
                    continue

                for metric in self.continuous_metrics:
                    if metric == "cohen_d":
                        val = _cohen_d(a, b)
                    elif metric == "rank_biserial":
                        val = _rank_biserial(a, b)
                    elif metric == "median_abs_shift":
                        val = _median_abs_shift(a, b)
                    elif metric == "cles":
                        val = _cles(a, b)
                    elif metric == "overlap_coef":
                        val = _overlap_coef(a, b)
                    elif metric == "normalized_wasserstein":
                        val = _normalized_wasserstein(a, b)
                    else:
                        continue
                    setattr(result, metric, val)
                    agg[metric].append(val)

            per_column.append(result)

        results = EffectSizeResults(
            per_column=per_column,
            enabled_continuous=self.continuous_metrics,
            enabled_categorical=self.categorical_metrics,
        )

        # Aggregate: mean-absolute for signed metrics (cohen_d, rank_biserial,
        # median_abs_shift) to report effect size magnitude; plain mean for others.
        _signed = {"cohen_d", "rank_biserial", "median_abs_shift"}
        for metric in self.continuous_metrics + self.categorical_metrics:
            vals = agg.get(metric, [])
            if vals:
                arr = np.array(vals)
                agg_val = (
                    float(np.mean(np.abs(arr)))
                    if metric in _signed
                    else float(np.mean(arr))
                )
                setattr(results, f"mean_{metric}", agg_val)

        # Compute per-column divergence score and rank most/least divergent
        for r in per_column:
            if r.is_categorical:
                if r.hellinger is not None:
                    r.divergence_score = r.hellinger
                elif r.js_divergence is not None:
                    r.divergence_score = r.js_divergence
            else:
                if r.overlap_coef is not None:
                    r.divergence_score = 1.0 - r.overlap_coef
                elif r.normalized_wasserstein is not None:
                    w = r.normalized_wasserstein
                    r.divergence_score = w / (w + 1.0)

        scored = sorted(
            [r for r in per_column if r.divergence_score is not None],
            key=lambda r: r.divergence_score,  # type: ignore[arg-type]
            reverse=True,
        )
        _n = 5
        results.most_divergent = scored[:_n]
        results.least_divergent = list(reversed(scored[-_n:]))

        n_cont = sum(not r.is_categorical for r in per_column)
        n_cat = sum(r.is_categorical for r in per_column)
        logger.info(
            f"EffectSize — {n_cont} continuous cols, {n_cat} categorical cols | "
            + ", ".join(
                f"{m}: {getattr(results, f'mean_{m}', None):.4f}"
                for m in (self.continuous_metrics + self.categorical_metrics)
                if getattr(results, f"mean_{m}", None) is not None
            )
        )
        if results.most_divergent:
            logger.info(
                "Most divergent columns: "
                + ", ".join(
                    f"{r.column} ({r.divergence_score:.4f})"
                    for r in results.most_divergent
                )
            )
        if results.least_divergent:
            logger.info(
                "Least divergent columns: "
                + ", ".join(
                    f"{r.column} ({r.divergence_score:.4f})"
                    for r in results.least_divergent
                )
            )
        return results

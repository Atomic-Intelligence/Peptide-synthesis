# Correlation Uncertainty Estimation

## Purpose

Measures how reliably each pairwise correlation in a dataset can be estimated given the sample size. Wide confidence intervals mean the correlation is uncertain; narrow ones mean it is well-determined.

## Pipeline

### 1. Column Selection

Columns are chosen by one of three strategies (in priority order):

1. **Zero-fraction range** — auto-select peptide columns whose fraction of zeros falls within `[lo, hi]`. Bypasses `max_columns`.
2. **Explicit list** — user-provided column names, trimmed to `max_columns` by preferring columns with the fewest zeros.
3. **All numeric columns** — fallback, also trimmed to `max_columns` (default 50) to avoid O(n^2) blowup.

### 2. Bootstrap Resampling

The dataset (N rows) is resampled **with replacement** B times (default 1000). For each bootstrap sample, the full pairwise correlation matrix is computed (Spearman or Pearson). This yields a distribution of B correlation estimates per column pair.

### 3. Fisher z-Transform and Confidence Intervals

Raw correlations are bounded to [-1, 1] and have a skewed sampling distribution near the extremes. To get valid CIs:

1. **Transform** each bootstrap correlation matrix via Fisher z: `z = arctanh(r)` (approximately normal).
2. **Compute percentile CIs** in z-space (e.g. 2.5th and 97.5th percentiles for a 95% CI).
3. **Back-transform** to correlation space: `r = tanh(z)`.

The mean correlation is also computed as `tanh(mean(z))` rather than a raw average of r values, which avoids bias at extreme correlations.

### 4. Outputs

| Output | Shape | Description |
|--------|-------|-------------|
| `corr_mean` | (n_cols, n_cols) | Mean bootstrapped correlation matrix |
| `ci_lower` | (n_cols, n_cols) | Lower CI bound |
| `ci_upper` | (n_cols, n_cols) | Upper CI bound |
| `ci_width` | (n_cols, n_cols) | `ci_upper - ci_lower` |
| `summary_table` | DataFrame | One row per unique pair with corr, CI bounds, and width |

Summary scalars (logged to MLflow): **mean CI width** and **max CI width** across all upper-triangle pairs.

### 5. Visualisation

Two heatmaps side by side:

- **Left** — mean bootstrapped correlation (coolwarm, [-1, 1]).
- **Right** — CI width (YlOrRd). Hot spots flag pairs where the correlation estimate is unreliable.

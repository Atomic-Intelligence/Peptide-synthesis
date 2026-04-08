# Fidelity Metrics Reference

All metrics are computed by `FidelityReport` in `src/evaluation/fidelity/fidelity_report.py` and logged to MLflow under the keys shown below.  Each sub-module can be toggled independently via `configs/evaluation/fidelity.yaml`.

---

## 1. Marginal Fidelity
**Module:** `src/evaluation/fidelity/marginal_fidelity.py`  
**Scope:** Per-column, univariate distribution comparison.

### Continuous columns

| MLflow key | Description |
|---|---|
| `fidelity/mean_ks_statistic` | Mean Kolmogorov-Smirnov statistic across all continuous columns.  Range [0, 1]; lower is better. |
| `fidelity/mean_wasserstein` | Mean Wasserstein-1 (Earth Mover's Distance) across columns.  Measures the cost of transforming one distribution into the other. |
| `fidelity/frac_ks_significant` | Fraction of columns where the KS p-value < 0.05 — i.e., distributions are statistically distinguishable. |

Per-column (stored in `MarginalFidelityResults.per_column`, not logged to MLflow):

- `ks_statistic` / `ks_pvalue` — raw KS test result
- `wasserstein` — per-column Wasserstein distance
- `mean_ratio` — `synth_mean / real_mean`
- `std_ratio` — `synth_std / real_std`
- `zero_diff` — `synth_zero_pct − real_zero_pct`

### Categorical columns

| MLflow key | Description |
|---|---|
| `fidelity/mean_tvd` | Mean Total Variation Distance (TVD) = ½ · Σ\|p_i − q_i\|.  Range [0, 1]; 0 = identical PMFs. |

---

## 2. Correlation Fidelity
**Module:** `src/evaluation/fidelity/correlation_fidelity.py`  
**Scope:** Pairwise correlation structure (up to `max_correlation_cols` columns, default 60).  Uses Spearman or Pearson correlation (configurable).

| MLflow key | Description |
|---|---|
| `fidelity/correlation_frobenius` | Frobenius norm of the difference matrix `\|C_real − C_synth\|_F`.  Captures overall correlation structure divergence. |
| `fidelity/correlation_mace` | Mean Absolute Correlation Error — mean of `\|C_real[i,j] − C_synth[i,j]\|` over all pairs. |
| `fidelity/correlation_max_abs_error` | Maximum absolute correlation error over all column pairs. |

---

## 3. Joint Fidelity
**Module:** `src/evaluation/fidelity/joint_fidelity.py`  
**Scope:** Full multivariate distribution (uses scaled + one-hot encoded feature space).

| MLflow key | Description |
|---|---|
| `fidelity/mmd` | Maximum Mean Discrepancy with RBF kernel (median-heuristic bandwidth), unbiased U-statistic estimator.  Lower is better; 0 = identical distributions. |
| `fidelity/precision` | Fraction of synthetic samples that fall within the support of the real data (k-NN balls).  High precision → no hallucinated modes. |
| `fidelity/recall` | Fraction of real data modes covered by at least one synthetic point.  High recall → no mode dropping. |
| `fidelity/coverage` | Fraction of real data k-NN balls that contain ≥1 synthetic point (Naeem et al. 2020).  More robust alternative to Recall. |

---

## 4. Two-Sample Classifier Test
**Module:** `src/evaluation/fidelity/classifier_test.py`  
**Scope:** Trains a binary classifier (real=0 vs. synthetic=1) using stratified k-fold CV.  If AUC ≈ 0.5 the synthetic data is indistinguishable from real.

| MLflow key | Description |
|---|---|
| `fidelity/two_sample_auc` | Mean ROC-AUC across CV folds.  Target: ≤ 0.55 (Excellent). |
| `fidelity/two_sample_auc_std` | Standard deviation of AUC across folds — measures stability. |
| `fidelity/two_sample_accuracy` | Mean classification accuracy across folds. |
| `fidelity/two_sample_f1` | Mean F1-score across folds. |

**Fidelity grade thresholds:**

| AUC range | Grade |
|---|---|
| ≤ 0.55 | Excellent |
| 0.55 – 0.65 | Good |
| 0.65 – 0.80 | Moderate |
| > 0.80 | Poor |

Top-N most discriminative features are stored in `TwoSampleClassifierResults.feature_importances` (not logged to MLflow scalar metrics).

---

## 5. Correlation Uncertainty
**Module:** `src/evaluation/analysis/correlation_uncertainty.py`  
**Scope:** Bootstrapped 95% confidence intervals for each pairwise correlation in the *real* dataset (up to `max_corr_uncertainty_cols` columns, default 30).  Wide CIs indicate unreliable correlation estimates given the sample size.

| MLflow key | Description |
|---|---|
| `correlation_uncertainty/mean_ci_width` | Mean CI width across all column pairs (upper triangle). |
| `correlation_uncertainty/max_ci_width` | Maximum CI width — identifies the least-reliably estimated correlation pair. |

Per-pair results (in `CorrelationUncertaintyResults.summary_table`):

- `corr` — mean bootstrapped correlation
- `ci_lower` / `ci_upper` — confidence interval bounds (Fisher z back-transformed)
- `ci_width` — `ci_upper − ci_lower`

---

## 6. Effect Size Metrics
**Module:** `src/evaluation/fidelity/effect_size.py`  
**Scope:** Per-column effect sizes quantifying the *practical magnitude* of distribution shift, independent of sample size.  Reported as mean absolute values over all columns in the aggregate MLflow metrics.

**Configure which metrics to compute in `configs/evaluation/fidelity.yaml` under `effect_size_metrics`.**

### Continuous columns

| MLflow key | Description | Range | Reference |
|---|---|---|---|
| `fidelity/effect_size/cohen_d` | Pooled-std Cohen's d: `(mean_synth − mean_real) / pooled_std`.  Aggregate = mean\|d\|. | (−∞, ∞); \|d\| ≥ 0.2 small, 0.5 medium, 0.8 large | Cohen 1988 |
| `fidelity/effect_size/rank_biserial` | Rank-biserial correlation from Mann-Whitney U: `1 − 2U/(n1·n2)`.  Equivalent to Cliff's delta.  Aggregate = mean\|r\|. | [−1, 1]; 0 = no shift | Cureton 1956 |
| `fidelity/effect_size/median_abs_shift` | `(median_synth − median_real) / MAD_real`.  Robust to outliers and heavy tails.  Aggregate = mean\|shift\|. | (−∞, ∞); 0 = identical medians | — |
| `fidelity/effect_size/cles` | Common Language Effect Size: P(synth > real), estimated from Mann-Whitney U.  Aggregate = mean CLES. | [0, 1]; 0.5 = no effect | McGraw & Wong 1992 |
| `fidelity/effect_size/overlap_coef` | Weitzman Overlap Coefficient (OVL) — histogram-approximated ∫ min(f_real, f_synth) dx.  Aggregate = mean OVL. | [0, 1]; 1 = identical distributions | Weitzman 1970 |
| `fidelity/effect_size/normalized_wasserstein` | Wasserstein-1 distance divided by the real column's IQR — scale-free.  Aggregate = mean. | [0, ∞); 0 = identical | — |

### Categorical columns

| MLflow key | Description | Range |
|---|---|---|
| `fidelity/effect_size/hellinger` | Hellinger distance between empirical PMFs: `sqrt(½·Σ(√p_i − √q_i)²)`.  Aggregate = mean. | [0, 1]; 0 = identical PMFs |
| `fidelity/effect_size/js_divergence` | Jensen-Shannon divergence (base-2) between empirical PMFs.  Aggregate = mean. | [0, 1]; 0 = identical PMFs |

---

## Configuration Reference

`configs/evaluation/fidelity.yaml`:

```yaml
enabled: true

# Correlation
corr_method: spearman           # pearson | spearman
max_correlation_cols: 60

# Correlation uncertainty (bootstrap)
run_corr_uncertainty: true
n_bootstrap: 200                # 1000 for publication-quality CIs
max_corr_uncertainty_cols: 30

# Two-sample classifier
n_classifier_folds: 5
classifier_type: random_forest  # random_forest | logistic_regression | gradient_boosted

# Effect size (remove any metric name to skip it)
run_effect_size: true
effect_size_metrics:
  continuous:
    - cohen_d
    - rank_biserial
    - median_abs_shift
    - cles
    - overlap_coef
    - normalized_wasserstein
  categorical:
    - hellinger
    - js_divergence
```

---

## Summary Table

| # | Sub-module | MLflow prefix | # metrics |
|---|---|---|---|
| 1 | Marginal Fidelity | `fidelity/` | 4 aggregate + per-column |
| 2 | Correlation Fidelity | `fidelity/correlation_*` | 3 |
| 3 | Joint Fidelity | `fidelity/` | 4 |
| 4 | Two-Sample Classifier | `fidelity/two_sample_*` | 4 |
| 5 | Correlation Uncertainty | `correlation_uncertainty/` | 2 |
| 6 | Effect Size | `fidelity/effect_size/` | up to 8 (configurable) |
| | **Total** | | **25** |

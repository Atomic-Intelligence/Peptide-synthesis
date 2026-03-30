# Copula Generative Model: Analysis & Improvement Plan

## Current Architecture Summary

The model uses a **Gaussian Copula** (or Student-T) to capture multivariate dependencies between clinical columns and peptide measurements. The pipeline is:

1. **Preprocessing**: Zero-to-null conversion (peptides), categorical-to-continuous mapping via truncated Gaussian sampling, missing value imputation with indicator columns
2. **Marginal fitting**: Per-column MLE across a pool of scipy distributions, selected by KL divergence
3. **Correlation estimation**: QuantileTransformer to normal space, Pearson correlation, nearest valid correlation matrix via `corr_clipped`
4. **Sampling**: `CopulaDistribution.rvs()` from statsmodels, then reverse preprocessing
5. **Post-processing**: Histogram imputation for sparse peptide columns filtered out before training

---

## Identified Weaknesses

### W1. Linear Dependency Structure Only

The Gaussian copula captures **linear (Pearson) correlations** only. It cannot model:
- **Tail dependencies** (joint extreme events, e.g., simultaneously high peptide levels and adverse outcomes)
- **Asymmetric dependencies** (e.g., strong correlation in low-expression regime but weak in high)
- **Non-monotone relationships** (e.g., U-shaped or threshold effects)

This is a fundamental limitation. The correlation uncertainty evaluation already flags many column pairs where synthetic CIs fail to overlap real CIs, likely due to this.

### W2. Correlation Estimation in Transformed Space

The current approach:
1. Fit marginals
2. Apply `QuantileTransformer(output_distribution="normal")` to map data to Gaussian space
3. Compute `np.corrcoef` (Pearson) on the transformed data

**Problem**: The QuantileTransformer and the fitted marginals are two separate transformations. The copula's theoretical guarantee is that dependencies are captured in the *copula space* (uniform marginals), but here we're computing correlation after an additional quantile transform that is **disconnected from the marginals** used in `CopulaDistribution`. This mismatch means the correlation matrix doesn't perfectly correspond to the marginals it's paired with, introducing systematic bias.

### W3. Categorical Encoding Is Lossy

The truncated Gaussian interval mapping is clever but has issues:
- **Ordinal assumption**: Categories are mapped to ordered intervals based on frequency rank, imposing a false ordering on nominal variables (e.g., disease presence/absence)
- **Information loss during inversion**: When reversing, any value in an interval maps to the same category, but the copula doesn't know about interval boundaries, so generated values near boundaries are unreliable
- **Binary variables waste capacity**: A binary column (e.g., diabetes yes/no) gets the same treatment as a multi-category column, when a simpler threshold would suffice

### W4. Missing Data Handling Doubles Dimensionality

Each column with any missing values spawns a `_missing_indicator` column. For a dataset with 100 peptide columns all having some missing values, the copula fits a 200-dimensional correlation matrix. This:
- Inflates the correlation matrix (O(d^2) parameters)
- Dilutes signal: indicator columns are essentially binary (uniform marginal), adding noise to correlation estimation
- Makes the nearest-PSD correction work harder, potentially distorting true correlations

### W5. Marginal Distribution Pool Is Generic

The same set of 8 distributions is tried for all continuous columns. Peptide expression data is often:
- **Zero-inflated**: Many exact zeros mixed with a continuous positive distribution (this is partially handled by the zero-to-null conversion, but the nulls are then filled by sampling)
- **Heavy-tailed**: A few patients with extremely high expression
- **Multimodal**: Subpopulations (e.g., event vs. control) may create bimodal marginals

None of the current candidate distributions (norm, gamma, lognorm, etc.) can capture zero-inflation or multimodality.

### W6. No Conditional Generation

The model generates all columns jointly. There's no mechanism to:
- Condition on clinical variables and generate peptides
- Generate event-specific synthetic patients
- Preserve known constraints (e.g., if `event_type = CKD`, then time-to-CKD-event should follow a specific distribution)

Currently this is handled externally by training separate models per event group, but conditional generation would be more statistically efficient.

### W7. Sparse Peptide Columns Are Discarded

Peptides below the `non_zero_threshold` (40%) are removed from copula training and handled by `HistogramImputation`, which:
- Generates each sparse column **independently** (no inter-column dependency)
- Uses simple histogram binning (loses fine structure)
- Cannot capture the dependency between sparse and non-sparse peptides

### W8. Post-Generation Quality

- `fill_null(0.0)` after reverse preprocessing forces any remaining nulls to zero, which may not be appropriate for all columns
- No clipping or constraint enforcement: generated values can be negative for inherently positive quantities (e.g., age, BMI, peptide concentrations)
- No validation that categorical columns only contain valid categories after reverse transformation

### W9. Student-T Copula Is Underutilized

The Student-T copula with `df=1` (Cauchy-like) is extremely heavy-tailed. There's no mechanism to:
- Estimate the optimal degrees of freedom from data
- Cross-validate between Gaussian and Student-T
- The `df=1` hardcode makes Student-T practically unusable for realistic data

---

## Improvement Plan

### Phase 1: Fix Core Statistical Issues (High Impact, Moderate Effort)

#### 1.1 Fix Correlation-Marginal Consistency

**Problem**: QuantileTransformer and fitted marginals are disconnected.

**Solution**: Use the **fitted marginal CDFs** themselves to transform data to uniform space, then apply `norm.ppf` to get Gaussian space for correlation estimation.

```
For each column j:
    U_j = F_j(X_j)           # CDF of fitted marginal → uniform
    Z_j = Phi_inv(U_j)       # inverse standard normal → Gaussian space
Correlation = corrcoef(Z)    # Now consistent with CopulaDistribution
```

This ensures the correlation matrix and marginals are **jointly consistent**, which is the theoretical requirement of the copula model.

**Files to modify**: `gaussian_copula_basic.py` (replace `transform_to_corr_space`)

#### 1.2 Estimate Student-T Degrees of Freedom

**Problem**: `df=1` is hardcoded.

**Solution**: Use maximum likelihood or method-of-moments to estimate `df` from the Gaussian-transformed data. Alternatively, use profile likelihood over a grid (df = 2, 3, 5, 10, 20, 50, ∞) and pick the best BIC.

**Files to modify**: `gaussian_copula_basic.py`, `gaussian_copula.yaml`

#### 1.3 Improve Marginal Distribution Pool

**Add distributions**:
- **Zero-inflated mixtures**: `ZeroInflatedGamma`, `ZeroInflatedLognormal` (custom scipy-compatible wrappers)
- **Kernel Density Estimation (KDE)**: Non-parametric fallback for columns that fit poorly
- **Mixture of Gaussians** (2-3 components): Captures bimodality

**Add selection logic**:
- Use BIC instead of (or alongside) KL divergence for model selection — KL can overfit to distribution shape
- Add a goodness-of-fit diagnostic: if the best-fit KL divergence is still above a threshold, fall back to KDE

**Files to modify**: `marginal_distribution_estimator.py`, `marginal_distribution_metrics.py`, new file for custom distributions

#### 1.4 Domain-Aware Value Constraints

**Add post-generation constraint enforcement**:
- Clip peptide values to [0, +inf) — concentrations can't be negative
- Clip age, BMI to plausible ranges
- Enforce categorical columns contain only observed categories
- Replace `fill_null(0.0)` with column-type-aware imputation

**Files to modify**: `gaussian_copula_basic.py` (`_generate` method)

---

### Phase 2: Better Categorical and Missing Data Handling (High Impact, Moderate Effort)

#### 2.1 Replace Truncated Gaussian Encoding with Extended Marginals

**Current**: Frequency-based intervals + truncated Gaussian sampling.

**Proposed**: Treat categorical columns as discrete marginals directly:
- Use **empirical PMF** as the marginal distribution for categorical columns
- During copula fitting, use the **normal scores** approach: assign each category its expected normal order statistic based on cumulative probability
- During generation, the copula generates a continuous value which is then mapped back through the inverse CDF of the empirical PMF (i.e., the quantile function of the discrete distribution)

This is the standard approach in the copula literature for mixed continuous-discrete data and avoids the artificial ordering problem.

**Files to modify**: `preprocessing.py` (simplify or remove categorical conversion), `gaussian_copula_basic.py` (integrate discrete marginals into `CopulaDistribution`)

#### 2.2 Reduce Missing Indicator Overhead

**Options** (pick based on evaluation results):

**Option A — Model missing pattern separately**:
1. Fit a separate model (e.g., logistic regression or a simple Bernoulli copula) for the missing/not-missing pattern
2. Fit the copula only on observed (non-missing) data
3. During generation: first generate missingness pattern, then generate values, then mask

**Option B — Selection model (Heckman-style)**:
1. Model P(observed | other columns) per column
2. Generate copula on complete cases
3. Post-hoc apply missingness

Either approach halves the correlation matrix dimensionality for columns with missing data.

**Files to modify**: `preprocessing.py`, `gaussian_copula_basic.py`

---

### Phase 3: Non-Linear Dependencies (High Impact, High Effort)

#### 3.1 Vine Copulas

**Motivation**: Vine copulas decompose the multivariate dependency into a hierarchy of bivariate copulas, each of which can be from a different family (Gaussian, Clayton, Gumbel, Frank, Joe, etc.). This allows:
- **Tail dependence** (Clayton, Gumbel)
- **Asymmetric dependence** (rotated copulas)
- **Mixed dependency structures** across different variable pairs

**Implementation approach**:
- Use the `pyvinecopulib` library (C++ backend, fast)
- Structure selection: R-vine with AIC-based pair-copula family selection
- Falls back gracefully to Gaussian for pairs with no special dependency

**Trade-off**: Significantly more parameters (one copula family + parameters per pair), but vine copulas handle high dimensions well through conditional independence pruning (truncation at level k).

**Files to modify**: New model class `VineCopulaModel` extending `SynthetizationModelInterface`, new config `vine_copula.yaml`

#### 3.2 Rank-Based Correlation (Kendall's Tau / Spearman)

Even without vine copulas, switching from Pearson to **Kendall's tau** for Gaussian copula parameter estimation is more robust:
- Kendall's tau directly estimates the copula parameter via `sin(pi/2 * tau)`
- Handles non-linear monotone relationships correctly
- Less sensitive to outliers

**Files to modify**: `gaussian_copula_basic.py` (correlation estimation block)

---

### Phase 4: Sparse Column Integration (Medium Impact, Moderate Effort)

#### 4.1 Unified Zero-Inflated Model

Instead of splitting columns into dense (copula) and sparse (histogram), model all columns together using zero-inflated marginals:

1. For each column, model as a **two-component mixture**:
   - Component 1: Point mass at zero with probability p_0
   - Component 2: Continuous distribution (gamma, lognorm, etc.) with probability 1-p_0
2. The copula operates on the latent continuous variable
3. During generation: first decide zero/non-zero (from the mixture), then generate the continuous value

This preserves cross-column dependencies even for sparse columns.

**Files to modify**: `marginal_distribution_estimator.py`, `gaussian_copula_basic.py`, remove `HistogramImputation` dependency from pipeline

#### 4.2 If Keeping Histogram Imputation — Add Dependencies

If the two-model approach is retained for practical reasons:
- After copula generation for dense columns, use **conditional histograms**: bin sparse columns conditional on the generated dense column values
- Use a simple conditional model (e.g., k-NN lookup in real data, or conditional density estimation)

**Files to modify**: `HistogramImputation.py`, `run_pipeline.py`

---

### Phase 5: Conditional Generation (Medium Impact, Medium Effort)

#### 5.1 Conditional Sampling from Copula

The Gaussian copula supports **exact conditional sampling**:
- Given observed values for a subset of columns (e.g., clinical variables), compute the conditional distribution of the remaining columns
- This is just conditional multivariate normal math (Schur complement)

**Implementation**:
```python
def generate_conditional(self, conditions: dict, n_samples: int):
    # 1. Transform conditions to copula space
    # 2. Partition correlation matrix into observed/unobserved blocks
    # 3. Compute conditional mean and covariance
    # 4. Sample from conditional normal
    # 5. Transform back through marginal inverse CDFs
```

This enables:
- Generating peptide profiles for specific patient demographics
- Preserving exact clinical variable values while generating synthetic peptides
- Event-type-specific generation without separate models

**Files to modify**: `gaussian_copula_basic.py` (new method), expose in pipeline config

---

### Phase 6: Evaluation-Driven Iteration (Ongoing)

#### 6.1 Use Correlation Uncertainty Results to Guide Improvements
Skip this subsection.

#### 6.2 Track Privacy-Fidelity Trade-off

Each improvement should be evaluated on **both** fidelity and privacy:
- Higher fidelity should not come at the cost of lower DCR ratios or higher MIA AUC
- Plot Pareto frontier: fidelity metrics vs. privacy metrics across model variants

#### 6.3 Ablation Study

Run the evaluation suite with:
1. Current model (baseline)
2. + Fix 1.1 (consistent correlation)
3. + Fix 1.3 (better marginals)
4. + Phase 2 (categorical + missing)
5. + Phase 3 (vine copula or rank correlation)
6. + Phase 4 (unified sparse handling)

This isolates the contribution of each improvement.

---

## Priority Matrix

| Improvement | Impact | Effort | Priority |
|---|---|---|---|
| 1.1 Correlation-marginal consistency | High | Low | **P0 — Do first** |
| 1.4 Domain-aware constraints | High | Low | **P0 — Do first** |
| 1.3 Better marginal pool | High | Medium | **P1** |
| 3.2 Rank-based correlation | Medium | Low | **P1** |
| 2.1 Better categorical encoding | High | Medium | **P1** |
| 1.2 Student-T df estimation | Medium | Low | **P1** |
| 2.2 Reduce missing indicators | Medium | Medium | **P2** |
| 4.1 Zero-inflated marginals | Medium | Medium | **P2** |
| 5.1 Conditional generation | Medium | Medium | **P2** |
| 3.1 Vine copulas | High | High | **P3** |
| 4.2 Conditional histogram imputation | Low | Medium | **P3** |

---

## Summary

The most impactful quick wins are fixing the **correlation-marginal consistency** (1.1) and adding **domain constraints** (1.4) — these address real statistical errors in the current implementation. The medium-term wins are **better marginals** (1.3, including zero-inflated distributions), **rank-based correlation** (3.2), and **improved categorical encoding** (2.1). The long-term architectural improvement is moving to **vine copulas** (3.1) for non-linear dependency modeling, paired with **unified sparse column handling** (4.1) to eliminate the histogram imputation workaround.

Each phase should be validated against the existing evaluation suite (fidelity + privacy) before proceeding to the next.

# Positive Semi-Definite Corrections in the Gaussian Copula

## 1. Where corrections are applied

There are two places in `gaussian_copula_basic.py` where `statsmodels.stats.correlation_tools.corr_clipped` is called with `threshold=1e-5`.

**Training (`_fit`, line 436)**
```python
estimated_correlation = statsmodels.stats.correlation_tools.corr_clipped(
    estimated_correlation, threshold=1e-5
)
```
Applied immediately after `_estimate_correlation_matrix` returns, before the copula object is built and before the matrix is stored in `_corr_matrix`.

**Conditional generation (`generate_conditional`, line 607)**
```python
Sigma_2_given_1 = statsmodels.stats.correlation_tools.corr_clipped(
    Sigma_2_given_1, threshold=1e-5
)
```
Applied to the Schur complement `Σ_{22} - Σ_{21} Σ_{11}^{-1} Σ_{12}` before sampling from the conditional multivariate normal.

A full Higham nearest-correlation implementation (`nearcorr` in `nearest_psd.py`) is available in the codebase but is not used anywhere.

---

## 2. What `corr_clipped` does mechanically

`corr_clipped` is eigenvalue clipping, not Frobenius-optimal projection. The steps are:

1. Symmetric eigendecomposition: `C = V Λ Vᵀ`
2. Clip: `Λ′ = max(Λ, threshold)` element-wise
3. Reconstruct: `C′ = V Λ′ Vᵀ`
4. Renormalise: divide every entry `C′ᵢⱼ` by `√(C′ᵢᵢ C′ⱼⱼ)` so the diagonal is 1 again

The result is guaranteed to be a valid correlation matrix (PSD, unit diagonal, entries in [−1, 1]).

---

## 3. How PSD violations arise in our setting

### 3a. Pearson on copula-space data

`np.corrcoef` on the probit-transformed data `Z` produces a matrix whose eigenvalues are non-negative by construction when `n > d`. However, in the event-filtered splits (say ~500 patients per event class) the number of columns after preprocessing can approach or exceed the sample size, which makes the sample covariance matrix rank-deficient and introduces zero or numerically negative eigenvalues. Even with `n > d`, near-collinear peptide pairs accumulate floating-point negatives at `O(ε_machine · d)` magnitude.

### 3b. Rank-based transforms (Kendall / Spearman)

The conversions `sin(π/2 · τ)` and `2·sin(π/6 · ρ)` are valid formulas for recovering the Gaussian copula correlation from rank correlations asymptotically, but the *sample* matrix output by these formulas is not guaranteed to be PSD for finite `n`. The pairwise-computed Kendall path in particular never forms a joint matrix object during estimation — it builds the upper triangle independently — so no global PSD constraint is imposed during construction. The resulting matrix can have negative eigenvalues of magnitude `O(1/√n)`, which is far larger than the `1e-5` clipping floor.

### 3c. Schur complement

Even after training-time clipping, the stored `_corr_matrix` is only *approximately* PSD (all eigenvalues ≥ 1e-5 before renormalisation, but renormalisation can bring some back below zero numerically). The Schur complement subtracts `Σ_{21} Σ_{11}^{-1} Σ_{12}`, which amplifies small errors through the matrix inverse. With many conditioning variables the residual covariance `Σ_{2|1}` can have clearly negative eigenvalues even when the original matrix was numerically PSD.

---

## 4. Effect of the correction on the correlation structure

### 4a. Implicit shrinkage toward independence

When eigenvalue clipping is applied and the matrix is renormalised, the off-diagonal entries are systematically reduced in magnitude. To see why: clipping negative eigenvalues adds positive-definite mass to the matrix; after renormalisation the diagonal stays at 1 but the off-diagonal entries, which carry the correlation signal, absorb the relative reduction. The effect is **strongest when many eigenvalues are negative**, i.e. exactly the high-dimensional regime with rank-based estimators and small per-event sample sizes.

Concretely, if the raw matrix has eigenvalues spread from `−λ_min` to `λ_max` (both > 0), the clipping adds roughly `λ_min · I` before renormalisation, which is equivalent to shrinking every off-diagonal entry by a factor of approximately `1 / (1 + λ_min / λ_max)`. In the worst case (many large negative eigenvalues), this can attenuate correlations by 10–30 %.

### 4b. Loss of rank-based correction benefits

The Kendall and Spearman estimators were added precisely because they are more robust to outliers and non-Gaussian marginals. Applying eigenvalue clipping afterwards partially re-introduces bias by uniformly shrinking correlations, which offsets some of the distributional robustness the rank-based path provides.

### 4c. Non-optimality relative to Higham's algorithm

Eigenvalue clipping does not minimise `‖C − C′‖_F` over valid correlation matrices. Higham's algorithm (`nearcorr`, already in `nearest_psd.py`) does. In practice, for matrices with modest violations, the Frobenius distance from clipping is only slightly worse than Higham's. But for matrices with many negative eigenvalues (which is common at high dimension), clipping introduces more distortion than necessary. The unused `nearcorr` function is a drop-in replacement that would reduce this distortion.

### 4d. Conditional generation distortion

The Schur complement after PSD correction is only used in `generate_conditional`, not in the unconditional `_generate` path. For unconditional generation `allow_singular=True` is passed to `GaussianCopula`, which means statsmodels handles rank-deficiency internally and the correction at training time is the only intervention. For conditional generation the two corrections compound: a slightly distorted `_corr_matrix` feeds into the Schur complement, and then `corr_clipped` is called again on the residual. This double-correction can noticeably flatten the conditional distribution, especially when the observed variables `Σ_{11}` are small in number and strongly correlated with the target variables.

---

## 5. Shrinkage as an alternative

Instead of correcting a PSD-violating matrix after the fact, shrinkage estimators produce PSD matrices *by construction* and simultaneously address the high-dimension / low-sample problem.

### 5a. Ledoit-Wolf

The Ledoit-Wolf estimator produces:

```
Ĉ = (1 − α) · S + α · μ · I
```

where `S` is the sample correlation matrix, `μ = tr(S)/d` is the average eigenvalue, and `α ∈ [0, 1]` is chosen analytically to minimise the expected Frobenius error. The result is always PSD. The shrinkage intensity `α` is automatically larger when `d/n` is large, exactly tracking when PSD violations would otherwise be severe.

For the copula, shrinkage should be applied to the copula-space matrix — i.e., to `Z = Φ⁻¹(F(X))` for the Pearson path or directly to the rank-correlation matrix for the Kendall/Spearman paths.

```python
from sklearn.covariance import LedoitWolf
lw = LedoitWolf(assume_centered=True)
lw.fit(Z)
estimated_correlation = cov_to_corr(lw.covariance_)
```

`cov_to_corr` normalises the covariance to a correlation matrix; `sklearn` does not expose it but the operation is `C = D^{-1/2} Σ D^{-1/2}` where `D = diag(Σ)`.

### 5b. Oracle Approximating Shrinkage (OAS)

OAS (`sklearn.covariance.OAS`) improves on Ledoit-Wolf for Gaussian data by using a better estimator of the squared Frobenius error. It has similar API and the same PSD guarantee. The practical difference versus Ledoit-Wolf is small for `n ≥ 100`.

### 5c. Constant-correlation shrinkage target

An alternative to the identity target is the constant-correlation matrix (all off-diagonal entries equal to the average pairwise correlation). This preserves the average level of dependence while regularising the eigenvalue spread. It is particularly appropriate for the peptide block where correlations cluster tightly around a common background value.

### 5d. Relationship between shrinkage intensity and current threshold

The `threshold=1e-5` in `corr_clipped` is very small: it only removes eigenvalues that are numerically negative (below machine precision times `d`). This is essentially not shrinking at all. It corrects exact numerical failures but does nothing about the statistical estimation variance that causes the eigenvalue spectrum to be poorly conditioned. Ledoit-Wolf with `d ≈ n` would typically produce `α ≈ 0.1–0.4`, which is a materially stronger regularisation and would substantially improve the conditioning of the matrix fed into `GaussianCopula`.

---

## 6. Summary

| Aspect | Current (`corr_clipped`) | Higham (`nearcorr`) | Ledoit-Wolf |
|---|---|---|---|
| PSD guarantee | Yes | Yes | Yes (by construction) |
| Frobenius-optimal | No | Yes | No (minimises expected MSE, not Frobenius) |
| Effect on correlations | Implicit shrinkage, proportional to eigenvalue mass below threshold | Minimal distortion | Explicit, controlled shrinkage toward identity |
| When violations are large | Over-shrinks correlations | Corrects with minimal distortion | Avoids violations entirely |
| Computational cost | O(d³) eigendecomposition | O(d³ · iterations) | O(d³) |
| Handles rank deficiency (d ≥ n) | Clips zero eigenvalues, does not regularise | Same | Regularises by design |
| Applied to Schur complement | Yes, second time | Could be substituted | Would apply to training matrix only |

**Recommended path:** apply Ledoit-Wolf to the copula-space correlation estimate as a first step, replacing `corr_clipped` at training time. The `corr_clipped` call in `generate_conditional` on the Schur complement can stay as a cheap numerical safety net, since the Schur complement correction is secondary and the underlying matrix will already be well-conditioned. If preserving exact pairwise-correlation structure is more important than conditioning, Higham's `nearcorr` (already present and tested in `nearest_psd.py`) is a better drop-in than the current eigenvalue clipper.

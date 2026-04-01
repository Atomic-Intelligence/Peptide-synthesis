# Correlation Matrix Corrections

## Why corrections are necessary

The Gaussian copula at the core of our peptide generation pipeline requires a valid (positive semi-definite) correlation matrix. In practice, the estimated matrix often violates this requirement for three reasons:

1. **High dimension vs. small sample size.** After event-class filtering we can have ~500 samples with a comparable or larger number of peptide columns. The sample correlation matrix becomes rank-deficient and acquires numerically negative eigenvalues.
2. **Rank-based estimators.** The Kendall and Spearman paths estimate each pairwise correlation independently and then map it to the Gaussian copula parameter via $\sin(\pi/2 \cdot \tau)$ or $2\sin(\pi/6 \cdot \rho)$. These element-wise transforms do not enforce joint positive semi-definiteness; the resulting matrix can have negative eigenvalues of order $O(1/\sqrt{n})$.
3. **Schur complement in conditional generation.** Even a barely-valid matrix can lose positive semi-definiteness after computing $\Sigma_{22} - \Sigma_{21} \Sigma_{11}^{-1} \Sigma_{12}$, because the matrix inverse amplifies small errors.

Without correction, sampling from a multivariate normal with an invalid covariance would either fail outright or produce statistically meaningless synthetic data.

## How the correction is done

We use **eigenvalue clipping** via `statsmodels.stats.correlation_tools.corr_clipped` with a threshold of $10^{-5}$. The procedure is:

1. Eigendecompose: $C = V \Lambda V^\top$
2. Clip: $\Lambda' = \max(\Lambda,\; 10^{-5})$ element-wise
3. Reconstruct: $C' = V \Lambda' V^\top$
4. Renormalise the diagonal to 1 so the result is a proper correlation matrix

This is applied in two places in `gaussian_copula_basic.py`:

- **At training time** (line 436), immediately after estimating the correlation matrix.
- **At conditional generation time** (line 607), on the Schur complement before sampling.

A Higham nearest-correlation implementation (`nearest_psd.py`) exists in the codebase but is currently unused.

## Is the correction large?

It depends on the regime. The $10^{-5}$ threshold only removes eigenvalues that are essentially zero or negative, so when the matrix is already close to PSD (e.g. Pearson path with $n \gg d$) the correction is negligible.

However, for the rank-based estimators (Kendall/Spearman) with our typical sample sizes, many eigenvalues can be meaningfully negative. Clipping then adds positive-definite mass proportional to the largest negative eigenvalue. After diagonal renormalisation this **implicitly shrinks all off-diagonal correlations by roughly 10--30%** in the worst case. This is an uncontrolled side-effect: the clipping was meant to fix a numerical issue, but it silently attenuates the correlation structure that the rank-based estimators were chosen to preserve.

The conditional generation path compounds this: the training-time correction slightly distorts the stored matrix, and then a second correction is applied to the Schur complement, further flattening the conditional distribution.

## Should we use shrinkage?

Yes, Ledoit-Wolf shrinkage is the natural upgrade. Instead of correcting a broken matrix after the fact, it produces a valid PSD matrix by construction:

$$\hat{C} = (1 - \alpha) \cdot S + \alpha \cdot \mu \cdot I$$

where $\alpha$ is chosen analytically to minimise expected estimation error and scales automatically with the $d/n$ ratio.

**Key differences from the current approach:**

| | Eigenvalue clipping (current) | Ledoit-Wolf shrinkage |
|---|---|---|
| PSD guarantee | Yes (post-hoc) | Yes (by construction) |
| Shrinkage | Implicit, uncontrolled (10--30%) | Explicit, optimal ($\alpha \approx 0.1$--$0.4$) |
| Handles $d \sim n$ | Clips zeros, does not regularise | Regularises by design |
| Effect on fidelity metrics | Over-shrinks strong correlations | Controlled, data-adaptive trade-off |

## Expected impact on results

Switching to Ledoit-Wolf would:

- **Improve correlation fidelity** (lower Frobenius norm and MACE between real and synthetic matrices) by replacing uncontrolled attenuation with an optimal bias-variance trade-off.
- **Stabilise conditional generation** by feeding a well-conditioned matrix into the Schur complement, reducing the need for the second correction.
- **Preserve rank-estimator benefits** -- the Kendall/Spearman paths were added for their robustness; the current clipping partially undoes that robustness by uniformly shrinking correlations.

The `corr_clipped` call on the Schur complement can remain as a cheap numerical safety net, since the underlying matrix will already be well-conditioned after shrinkage.

"""PCA + Gaussian Copula for Synthetic Data Generation in High-Dimensional Regimes.

This script demonstrates how to combine dimensionality reduction (PCA) with a
Gaussian copula to generate realistic synthetic data when the number of features
(d) greatly exceeds the number of samples (n).

============================================================================
THE PROBLEM
============================================================================
A Gaussian copula requires a d x d positive semi-definite (PSD) correlation
matrix.  When d >> n (e.g., ~1187 peptide features with only ~100 samples
for minority event classes), the sample correlation matrix is at most rank
n-1, meaning ~d - n + 1 eigenvalues are zero or negative.  The standard fix
-- eigenvalue clipping via `corr_clipped` -- projects to the nearest PSD
matrix, but introduces implicit shrinkage that distorts the learned
correlation structure (up to ~3.7% for the minority classes in our dataset).

============================================================================
THE SOLUTION
============================================================================
Instead of fitting a copula on all d features and then correcting, we:

  1. Fit marginal distributions per column (univariate -- no d>>n issue).
  2. Transform data to Gaussian copula space: X -> U=F(X) -> Z=phi_inv(U).
  3. Apply PCA in copula space, keeping k << n components.
  4. Fit a k x k Gaussian copula on PCA scores (well-conditioned since k < n).
  5. Generate synthetic PCA scores from the k-dim copula.
  6. Inverse-PCA back to d dimensions.
  7. Inverse copula transform: Z -> U=phi(Z) -> X=F_inv(U).

This avoids the d x d correlation estimation entirely; the copula only ever
sees a k x k matrix with k < n, so it is always well-conditioned.

============================================================================
USAGE
============================================================================
    python scripts/pca_copula_synthetic_data.py \
        --real resources/merged_peptide_and_clinical.csv \
        --output evaluation_output/pca_copula \
        --event-column event_type \
        --event-type hf \
        --n-components 50 \
        --n-synthetic 200 \
        --variance-threshold 0.95

If --n-components is not given, the script auto-selects the number of PCA
components to explain --variance-threshold of the total variance (default 95%).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import rankdata, norm, kstest, yeojohnson
from sklearn.decomposition import PCA
from statsmodels.distributions.copula.copulas import CopulaDistribution
from statsmodels.distributions.copula.elliptical import GaussianCopula
from statsmodels.stats.correlation_tools import corr_clipped
from tqdm import tqdm


# ============================================================================
# POWER-TRANSFORMED MARGINAL
# ============================================================================
# Peptide intensity data is often heavily right-skewed with a long tail and a
# spike at zero (below-detection-limit measurements).  Standard parametric
# families (lognorm, gamma, ...) can struggle to capture both the spike and
# the tail simultaneously.
#
# A Yeo-Johnson power transform maps the data to a more symmetric, roughly
# Gaussian-like scale *before* we fit the parametric marginal.  This yields:
#   - Lower BIC (better marginal fits)
#   - Fewer extreme values when mapping to copula space
#   - More stable inverse transforms during generation
#
# Why Yeo-Johnson over Box-Cox?
#   Box-Cox requires strictly positive data.  Peptide columns often contain
#   exact zeros (detection limit), which would need an arbitrary additive
#   shift.  Yeo-Johnson is defined for all real numbers (including zero)
#   and naturally handles the zero case.
#
# The class below wraps a power transform + parametric marginal into an
# object that exposes .cdf() and .ppf(), so it plugs transparently into
# the copula pipeline (transform_to_copula_space / inverse_copula_transform).


class PowerTransformedMarginal:
    """A marginal distribution that applies a Yeo-Johnson power transform
    before the parametric fit.

    Composites two transformations:
        original scale  --[Yeo-Johnson]--> transformed scale --[parametric dist]--> probability

    The Yeo-Johnson transform for x >= 0 (our case for peptides):
        lambda != 0:  y = ((x + 1)^lambda - 1) / lambda
        lambda == 0:  y = ln(x + 1)

    Inverse (x >= 0 case):
        lambda != 0:  x = (y * lambda + 1)^(1/lambda) - 1
        lambda == 0:  x = exp(y) - 1
    """

    def __init__(self, frozen_dist, lmbda: float, y_clip: tuple[float, float]):
        """
        Args:
            frozen_dist: A frozen scipy distribution fitted on the
                Yeo-Johnson-transformed data.
            lmbda: The Yeo-Johnson lambda parameter (estimated from data).
            y_clip: (y_lo, y_hi) — the safe range for values in the
                transformed space.  Derived from training data; extended by
                3 standard deviations to allow reasonable extrapolation.
                ppf() clamps the inverted distribution output to this range
                before applying the inverse power transform, preventing
                overflow when the copula generates extreme tail samples.
        """
        self.frozen_dist = frozen_dist
        self.lmbda = lmbda
        self.y_clip = y_clip

    @staticmethod
    def _yj_forward(x: np.ndarray, lmbda: float) -> np.ndarray:
        """Yeo-Johnson forward transform (vectorised, non-negative x)."""
        x = np.asarray(x, dtype=np.float64)
        if abs(lmbda) < 1e-10:
            return np.log1p(x)
        return (np.power(x + 1.0, lmbda) - 1.0) / lmbda

    @staticmethod
    def _yj_inverse(y: np.ndarray, lmbda: float) -> np.ndarray:
        """Yeo-Johnson inverse transform (vectorised, non-negative x)."""
        y = np.asarray(y, dtype=np.float64)
        if abs(lmbda) < 1e-10:
            return np.expm1(y)
        inner = y * lmbda + 1.0
        # Guard against negative base (can occur at extreme tails with
        # fractional lambda).
        inner = np.maximum(inner, 1e-15)
        return np.power(inner, 1.0 / lmbda) - 1.0

    def cdf(self, x):
        """CDF: power-transform x, then apply the fitted distribution's CDF."""
        x = np.asarray(x, dtype=np.float64)
        y = self._yj_forward(x, self.lmbda)
        return self.frozen_dist.cdf(y)

    def ppf(self, q):
        """PPF (quantile function): invert the fitted dist, then undo the
        power transform to return to the original peptide scale.

        Overflow prevention:
          1. q is clipped to [1e-6, 1-1e-6] so the parametric PPF never
             returns ±inf (which would explode in the inverse power transform).
          2. The inverted y value is clamped to y_clip — the training-data
             range extended by 3 std — so extreme copula samples stay within
             a region where the inverse power transform is finite.
        """
        q = np.asarray(q, dtype=np.float64)
        # Step 1: guard against q at the boundary of (0, 1)
        q = np.clip(q, 1e-6, 1.0 - 1e-6)
        y = self.frozen_dist.ppf(q)
        # Step 2: clamp to the safe y range (training range + 3-sigma buffer)
        y = np.clip(y, self.y_clip[0], self.y_clip[1])
        return self._yj_inverse(y, self.lmbda)


# ============================================================================
# STEP 1: Data Loading and Preparation
# ============================================================================
# We load the merged peptide + clinical dataset, optionally filter by event
# type, and separate peptide features from clinical variables.  Peptide
# columns are identified by the naming pattern "Peptide_XXXXX".


def load_and_prepare_data(
    csv_path: str,
    event_column: str | None = None,
    event_type: str | None = None,
    max_zero_frac: float = 0.4,
) -> tuple[np.ndarray, list[str], pl.DataFrame]:
    """Load CSV, filter by event, select peptide columns, drop sparse ones.

    Args:
        csv_path: Path to the merged peptide + clinical CSV.
        event_column: Column name for event-type stratification.
        event_type: If given, keep only rows where event_column == event_type.
        max_zero_frac: Drop peptide columns where more than this fraction
            of values are zero (very sparse peptides add noise without signal).

    Returns:
        X: (n_samples, n_features) numpy array of peptide values.
        peptide_cols: List of peptide column names that were kept.
        df: The filtered Polars DataFrame (for reference / clinical columns).
    """
    df = pl.read_csv(csv_path).drop_nulls()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # -- Optionally stratify by event type --
    if event_column and event_type:
        df = df.filter(pl.col(event_column) == event_type)
        print(f"Filtered to event_type='{event_type}': {len(df)} rows")

    # -- Identify peptide columns --
    peptide_pattern = re.compile(r"(?i)peptide")
    peptide_cols = [c for c in df.columns if peptide_pattern.search(c)]
    print(f"Found {len(peptide_cols)} peptide columns")

    # -- Drop columns that are too sparse (high zero fraction) --
    # Peptide measurements often have many exact zeros which are effectively
    # "below detection limit".  Columns that are mostly zeros don't carry
    # useful covariance information and just inflate d.
    kept = []
    for c in tqdm(peptide_cols, desc="Filtering sparse columns"):
        zero_frac = float((df[c] == 0).sum()) / len(df)
        if zero_frac <= max_zero_frac:
            kept.append(c)
    print(f"Kept {len(kept)} peptide columns (dropped {len(peptide_cols) - len(kept)} "
          f"with >{max_zero_frac*100:.0f}% zeros)")
    peptide_cols = kept

    X = df.select(peptide_cols).to_numpy().astype(np.float64)
    return X, peptide_cols, df


# ============================================================================
# STEP 2: Fit Marginal Distributions Per Column
# ============================================================================
# The copula model separates the joint distribution into:
#   - Marginals F_1, ..., F_d  (one per feature)
#   - A copula C that captures the dependence structure
#
# Fitting marginals is a univariate problem for each column independently,
# so it's not affected by d >> n.
#
# When use_power_transform=True (default), each column is first mapped
# through a Yeo-Johnson power transform to achieve a more symmetric scale,
# then a parametric distribution is fitted on the transformed values.  The
# result is a PowerTransformedMarginal that chains both transformations
# transparently.  This dramatically improves BIC for the heavy-tailed,
# zero-spiked peptide distributions.
#
# When use_power_transform=False, distributions are fitted directly on the
# raw (non-negative) peptide values with loc=0, matching the existing
# pipeline behaviour.


def fit_marginals(
    X: np.ndarray,
    column_names: list[str],
    candidate_dists: list | None = None,
    use_power_transform: bool = True,
) -> list:
    """Fit the best marginal distribution per column using BIC.

    For each column:
      1. (Optional) Apply a Yeo-Johnson power transform.  The optimal lambda
         is estimated via maximum likelihood (scipy.stats.yeojohnson).
      2. Try each candidate distribution, fit via MLE, compute BIC.
      3. Return the distribution (or PowerTransformedMarginal) with the
         lowest BIC.

    Args:
        X: (n, d) data matrix.
        column_names: Names for each column (for logging).
        candidate_dists: List of scipy distribution objects to try.
            Defaults to [lognorm, gamma, beta, expon] for peptide-like data.
        use_power_transform: If True, apply a Yeo-Johnson transform before
            fitting.  The returned marginals are PowerTransformedMarginal
            instances that handle the forward/inverse transform internally.

    Returns:
        List of marginal objects (each exposing .cdf() and .ppf()),
        one per column.
    """
    if candidate_dists is None:
        candidate_dists = [stats.lognorm, stats.gamma, stats.beta, stats.expon]

    # When the power transform produces a nice symmetric scale, a normal
    # distribution often wins.  Include it as a candidate so BIC can pick it.
    if use_power_transform:
        candidate_dists = list(candidate_dists) + [stats.norm]

    n, d = X.shape
    marginals = []
    n_power_transformed = 0

    for j in tqdm(range(d), desc="Fitting marginals"):
        col_data = X[:, j]
        finite_data = col_data[np.isfinite(col_data)]

        # Handle degenerate columns: constant or empty -> uniform fallback
        if len(finite_data) == 0 or np.ptp(finite_data) == 0:
            marginals.append(stats.uniform(loc=finite_data[0] if len(finite_data) > 0 else 0,
                                           scale=1e-6))
            continue

        # -- Yeo-Johnson power transform --
        # The transform maps skewed, zero-heavy peptide intensities to a
        # more Gaussian-like scale where parametric families fit better.
        # scipy.stats.yeojohnson estimates the optimal lambda via MLE and
        # returns both the transformed data and the lambda.
        lmbda = None
        if use_power_transform:
            fit_data, lmbda = yeojohnson(finite_data)
        else:
            fit_data = finite_data

        best_bic = np.inf
        best_frozen = None

        for dist in candidate_dists:
            try:
                # When power-transformed, the data lives on an unrestricted
                # real scale, so we do NOT fix floc=0 (the transform already
                # handled the non-negativity).  Without the transform, we
                # fix loc=0 for non-negative peptide data.
                if use_power_transform:
                    params = dist.fit(fit_data)
                else:
                    params = dist.fit(fit_data, floc=0)

                # BIC = k*ln(n) - 2*ln(L)
                # Lower BIC is better (penalises model complexity).
                ll = np.sum(dist.logpdf(fit_data, *params))
                k = len(params)  # number of distribution parameters
                bic = k * np.log(len(fit_data)) - 2 * ll

                if np.isfinite(bic) and bic < best_bic:
                    best_bic = bic
                    best_frozen = dist(*params)
            except Exception:
                continue

        if best_frozen is None:
            mu, sigma = np.mean(fit_data), np.std(fit_data) + 1e-8
            best_frozen = stats.norm(loc=mu, scale=sigma)

        # Wrap in PowerTransformedMarginal so cdf/ppf handle the
        # Yeo-Johnson transform transparently.
        if use_power_transform and lmbda is not None:
            # Compute safe clip bounds in the transformed space:
            # training range extended by 3 standard deviations in each
            # direction.  This allows slight extrapolation beyond the
            # observed data while preventing overflow in the inverse power
            # transform when the copula generates extreme tail samples.
            y_std = float(np.std(fit_data))
            y_clip = (
                float(fit_data.min()) - 3.0 * y_std,
                float(fit_data.max()) + 3.0 * y_std,
            )
            marginals.append(PowerTransformedMarginal(best_frozen, lmbda, y_clip))
            n_power_transformed += 1
        else:
            marginals.append(best_frozen)

    if use_power_transform:
        print(f"  Power-transformed {n_power_transformed}/{d} columns (Yeo-Johnson)")

    return marginals


# ============================================================================
# STEP 3: Transform to Gaussian Copula Space
# ============================================================================
# Given fitted marginals F_1, ..., F_d, we transform each column:
#
#   X_j  ->  U_j = F_j(X_j)         # Probability Integral Transform -> Uniform(0,1)
#        ->  Z_j = phi_inv(U_j)      # Probit transform -> Standard Normal
#
# If the Gaussian copula is the correct model, then Z = (Z_1,...,Z_d) follows
# a multivariate normal N(0, Sigma) where Sigma is the copula correlation
# matrix.  This is the space where we'll apply PCA.


def transform_to_copula_space(
    X: np.ndarray,
    marginals: list[stats.rv_frozen],
    eps: float = 1e-8,
) -> np.ndarray:
    """Transform data X to Gaussian copula (normal scores) space.

    For each column j:
        U_j = F_j(X_j)           # CDF of fitted marginal -> pseudo-uniform
        Z_j = Phi^{-1}(U_j)      # inverse standard normal CDF -> normal scores

    The clipping to [eps, 1-eps] prevents infinite values from Phi^{-1}(0) or
    Phi^{-1}(1).

    Args:
        X: (n, d) original data.
        marginals: List of d frozen scipy distributions.
        eps: Clipping epsilon for the uniform values.

    Returns:
        Z: (n, d) data in Gaussian copula space.
    """
    n, d = X.shape
    Z = np.empty_like(X, dtype=np.float64)

    for j in range(d):
        u = marginals[j].cdf(X[:, j])       # X -> Uniform via marginal CDF
        u = np.clip(u, eps, 1.0 - eps)       # avoid infinities at boundaries
        Z[:, j] = norm.ppf(u)                # Uniform -> Standard Normal

    return Z


def inverse_copula_transform(
    Z: np.ndarray,
    marginals: list[stats.rv_frozen],
) -> np.ndarray:
    """Reverse the copula-space transformation: Z -> U -> X.

    For each column j:
        U_j = Phi(Z_j)           # standard normal CDF -> pseudo-uniform
        X_j = F_j^{-1}(U_j)     # inverse marginal CDF -> original scale

    Args:
        Z: (n, d) data in Gaussian copula (normal scores) space.
        marginals: List of d frozen scipy distributions.

    Returns:
        X: (n, d) data in original scale.
    """
    n, d = Z.shape
    X = np.empty_like(Z, dtype=np.float64)

    for j in range(d):
        u = norm.cdf(Z[:, j])                # Normal -> Uniform
        X[:, j] = marginals[j].ppf(u)        # Uniform -> Original scale

    return X


# ============================================================================
# STEP 4: PCA in Copula Space
# ============================================================================
# This is the key innovation.  In copula space, Z ~ N(0, Sigma).  PCA finds
# the principal axes of Sigma and projects Z onto the top-k directions.
#
# Why PCA in copula space (not original space)?
# - PCA in original space conflates marginal effects with dependence.
# - In copula space, marginals have been factored out; PCA captures pure
#   dependence structure.
# - The resulting PCA scores S = Z @ V_k are (approximately) jointly normal,
#   which is exactly what the Gaussian copula assumes.
#
# Choosing k:
# - k must be < n to avoid the rank-deficiency problem.
# - We pick k to explain a target fraction of variance (e.g., 95%).
# - Alternatively, k can be set manually (e.g., 50 components).


def fit_pca(
    Z: np.ndarray,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
) -> tuple[PCA, int]:
    """Fit PCA on copula-space data and select the number of components.

    If n_components is given, use that directly.  Otherwise, use the minimum
    number of components that explains at least variance_threshold of the
    total variance, with an upper bound of n_samples - 1 to stay within the
    well-conditioned regime.

    Args:
        Z: (n, d) data in copula space.
        n_components: Fixed number of PCA components (overrides auto-selection).
        variance_threshold: Fraction of variance to explain (default 0.95).

    Returns:
        pca: Fitted sklearn PCA object.
        k: The chosen number of components.
    """
    n, d = Z.shape

    if n_components is not None:
        # Use the user-specified number, but cap at n-1
        k = min(n_components, n - 1, d)
        print(f"Using user-specified n_components={k}")
    else:
        # Fit full PCA first to examine the explained variance curve, then
        # select the smallest k that explains >= variance_threshold.
        max_components = min(n - 1, d)
        pca_full = PCA(n_components=max_components)
        pca_full.fit(Z)

        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        k = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
        k = min(k, max_components)
        print(f"Auto-selected k={k} components "
              f"(explains {cumulative_variance[k-1]*100:.1f}% of variance, "
              f"threshold={variance_threshold*100:.0f}%)")

    # Fit the final PCA with the chosen k
    pca = PCA(n_components=k)
    pca.fit(Z)
    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA: {d} features -> {k} components ({explained:.1f}% variance explained)")

    return pca, k


# ============================================================================
# STEP 5: Fit Gaussian Copula on PCA Scores
# ============================================================================
# The PCA scores S = Z @ V_k have shape (n, k) with k < n.  Now the
# correlation estimation problem is well-conditioned: the k x k sample
# correlation matrix is full rank (since n > k), so corr_clipped introduces
# minimal to no distortion.
#
# We use Spearman rank correlation converted to the Gaussian copula parameter
# via rho_copula = 2 * sin(pi/6 * rho_spearman), which is the standard
# approach for Gaussian copulas with non-Gaussian marginals.


def fit_copula_on_pca_scores(
    S: np.ndarray,
) -> tuple[CopulaDistribution, np.ndarray]:
    """Fit a Gaussian copula on PCA scores.

    The PCA scores are approximately multivariate normal (since Z was normal
    and PCA is a linear transform).  We estimate the correlation matrix using
    Spearman rank correlation for robustness, then project to the nearest
    PSD matrix via corr_clipped (which should be nearly a no-op since k < n).

    Args:
        S: (n, k) PCA scores.

    Returns:
        copula_dist: Fitted CopulaDistribution for sampling.
        corr_matrix: The estimated k x k correlation matrix.
    """
    n, k = S.shape
    print(f"\nFitting Gaussian copula on PCA scores: n={n}, k={k}, k/n={k/n:.2f}")

    # -- Estimate correlation via Spearman rank method --
    # Spearman rho is computed as the Pearson correlation of ranks.
    # The conversion rho_copula = 2*sin(pi/6 * rho_spearman) maps from
    # Spearman's rho to the Gaussian copula correlation parameter.
    ranks = np.apply_along_axis(rankdata, 0, S)
    rho_spearman = np.corrcoef(ranks, rowvar=False)
    corr_matrix = 2.0 * np.sin(np.pi / 6.0 * rho_spearman)

    # -- Check conditioning before clipping --
    eigenvalues_raw = np.linalg.eigvalsh(corr_matrix)
    n_negative = int((eigenvalues_raw < 0).sum())
    print(f"  Raw correlation: {n_negative} negative eigenvalues "
          f"(min={eigenvalues_raw.min():.6f})")

    # -- Project to nearest PSD correlation matrix --
    # With k < n, this should be nearly a no-op (few or no negative eigenvalues).
    corr_clipped_matrix = corr_clipped(corr_matrix, threshold=1e-5)

    # -- Measure distortion from clipping --
    diff = corr_clipped_matrix - corr_matrix
    frob_correction = float(np.linalg.norm(diff, "fro"))
    frob_raw = float(np.linalg.norm(corr_matrix, "fro"))
    relative_frob = frob_correction / frob_raw if frob_raw > 0 else 0.0
    print(f"  Clipping distortion: ||correction||_F / ||raw||_F = {relative_frob*100:.4f}%")

    eigenvalues_clipped = np.linalg.eigvalsh(corr_clipped_matrix)
    print(f"  After clipping: min eigenvalue = {eigenvalues_clipped.min():.6f}")

    # -- Build the copula with normal marginals for PCA scores --
    # PCA scores are approximately N(0, lambda_k) by construction, but since
    # the copula separates marginals from dependence, we use standard normal
    # marginals N(0,1) for each PCA score dimension.  (The actual marginal
    # variances are captured by pca.explained_variance_ and will be restored
    # during inverse PCA.)
    score_marginals = [stats.norm(loc=0, scale=1) for _ in range(k)]
    gaussian_cop = GaussianCopula(corr=corr_clipped_matrix, allow_singular=True)
    copula_dist = CopulaDistribution(copula=gaussian_cop, marginals=score_marginals)

    return copula_dist, corr_clipped_matrix


# ============================================================================
# STEP 6: Generate Synthetic Samples
# ============================================================================
# The generation pipeline runs the entire process in reverse:
#
#   1. Sample synthetic PCA scores S_syn from the k-dim copula.
#   2. Inverse PCA: Z_syn = S_syn @ V_k^T + mean  (back to d-dim copula space).
#   3. Inverse copula transform: Z_syn -> U -> X via Phi(Z) and F_j^{-1}(U).
#   4. Apply domain constraints (e.g., peptides >= 0).


def generate_synthetic_data(
    copula_dist: CopulaDistribution,
    pca: PCA,
    marginals: list[stats.rv_frozen],
    n_samples: int,
    clip_non_negative: bool = True,
) -> np.ndarray:
    """Generate synthetic data through the PCA-copula pipeline.

    Pipeline: copula sample -> inverse PCA -> inverse copula transform.

    Args:
        copula_dist: Fitted CopulaDistribution on PCA scores.
        pca: Fitted PCA object (for inverse_transform).
        marginals: Per-column marginals in the original d-dimensional space.
        n_samples: Number of synthetic samples to generate.
        clip_non_negative: If True, clip negative values to 0 (peptides are
            non-negative by definition).

    Returns:
        X_syn: (n_samples, d) synthetic data in the original feature scale.
    """
    # -- Step 6a: Sample from the k-dimensional copula --
    # CopulaDistribution.rvs() generates samples that go through:
    #   copula -> uniform -> marginal PPF.
    # Since our PCA-score marginals are N(0,1), the output is just MVN samples
    # with the fitted correlation structure.
    S_syn = copula_dist.rvs(nobs=n_samples)
    print(f"\nGenerated {n_samples} synthetic PCA scores with shape {S_syn.shape}")

    # -- Step 6b: Inverse PCA to get back to d-dimensional copula space --
    # PCA stores: S = (Z - mean) @ V_k  =>  Z_approx = S @ V_k^T + mean
    # This maps from k dimensions back to d, but only reconstructs the
    # variance along the top-k principal directions.  The residual variance
    # (in the discarded components) is lost -- this is the dimensionality
    # reduction trade-off.
    Z_syn = pca.inverse_transform(S_syn)
    print(f"Inverse PCA: {S_syn.shape} -> {Z_syn.shape}")

    # -- Step 6c: Inverse copula transform --
    # Z -> U = Phi(Z), then U -> X = F_j^{-1}(U) per column.
    X_syn = inverse_copula_transform(Z_syn, marginals)
    print(f"Inverse copula transform: back to original scale")

    # -- Step 6d: Domain constraints --
    # Peptide values should be non-negative.  The inverse transform can
    # produce small negative values due to numerical imprecision at the
    # tails of the marginal distributions.
    if clip_non_negative:
        n_negative = int((X_syn < 0).sum())
        X_syn = np.clip(X_syn, 0, None)
        if n_negative > 0:
            print(f"Clipped {n_negative} negative values to 0")

    return X_syn


# ============================================================================
# STEP 7: Evaluation and Diagnostics
# ============================================================================
# We compare the real and synthetic data on several dimensions:
#   a) Per-column marginal statistics (mean, std, quantiles)
#   b) Correlation structure preservation
#   c) Eigenvalue spectrum comparison
#   d) PCA variance explained curve


def evaluate_synthetic_data(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    column_names: list[str],
    output_dir: Path,
    pca: PCA,
) -> dict:
    """Run diagnostic comparisons between real and synthetic data.

    Returns a dict of summary metrics and saves diagnostic plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_real, d = X_real.shape
    n_syn = X_syn.shape[0]
    metrics = {}

    # -- Sanitise synthetic data before computing any metrics --
    # Even with the y_clip guard in PowerTransformedMarginal.ppf, a tiny
    # number of values can still be NaN/inf due to edge-case interactions
    # between the inverse PCA and the copula space transform.  Replace them
    # with the per-column median of the real data so downstream stats don't
    # blow up.
    bad_mask = ~np.isfinite(X_syn)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        col_medians = np.nanmedian(X_real, axis=0)
        X_syn = np.where(bad_mask, col_medians[None, :], X_syn)
        print(f"  WARNING: replaced {n_bad} non-finite values in synthetic "
              f"data with per-column real medians ({n_bad / X_syn.size * 100:.3f}% of cells)")

    # ------------------------------------------------------------------
    # 7a: Per-column marginal comparison
    # ------------------------------------------------------------------
    # For each column, compare the mean, std, and key quantiles between
    # real and synthetic data.  Good synthetic data should closely match
    # the marginal statistics (since we fit per-column marginals).
    mean_real = np.mean(X_real, axis=0)
    mean_syn = np.mean(X_syn, axis=0)
    std_real = np.std(X_real, axis=0)
    std_syn = np.std(X_syn, axis=0)

    # Relative mean absolute error of column means
    mean_mae = np.mean(np.abs(mean_real - mean_syn) / (np.abs(mean_real) + 1e-8))
    std_mae = np.mean(np.abs(std_real - std_syn) / (np.abs(std_real) + 1e-8))
    metrics["marginal_mean_relative_mae"] = float(mean_mae)
    metrics["marginal_std_relative_mae"] = float(std_mae)
    print(f"\n--- Marginal Quality ---")
    print(f"  Mean relative MAE: {mean_mae:.4f}")
    print(f"  Std relative MAE:  {std_mae:.4f}")

    # ------------------------------------------------------------------
    # 7b: Correlation structure comparison
    # ------------------------------------------------------------------
    # Compare the Spearman correlation matrices of real vs. synthetic data.
    # This is the most important metric: the whole point of the copula is
    # to capture inter-column dependencies.
    def spearman_corr(X):
        ranks = np.apply_along_axis(rankdata, 0, X)
        return np.corrcoef(ranks, rowvar=False)

    corr_real = spearman_corr(X_real)
    corr_syn = spearman_corr(X_syn)

    # Extract upper triangle (off-diagonal correlations)
    mask = np.triu(np.ones_like(corr_real, dtype=bool), k=1)
    corr_diff = corr_syn[mask] - corr_real[mask]

    metrics["correlation_mae"] = float(np.mean(np.abs(corr_diff)))
    metrics["correlation_max_error"] = float(np.max(np.abs(corr_diff)))
    metrics["correlation_rmse"] = float(np.sqrt(np.mean(corr_diff ** 2)))
    print(f"\n--- Correlation Fidelity ---")
    print(f"  MAE of pairwise Spearman correlations: {metrics['correlation_mae']:.4f}")
    print(f"  Max absolute error:                    {metrics['correlation_max_error']:.4f}")
    print(f"  RMSE:                                  {metrics['correlation_rmse']:.4f}")

    # ------------------------------------------------------------------
    # 7c: Eigenvalue spectrum comparison
    # ------------------------------------------------------------------
    # The eigenvalue spectrum of the correlation matrix reveals the
    # effective dimensionality of the data.  A faithful synthetic dataset
    # should have a similar spectrum (at least for the top eigenvalues).
    eigs_real = np.sort(np.linalg.eigvalsh(corr_real))[::-1]
    eigs_syn = np.sort(np.linalg.eigvalsh(corr_syn))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    k_plot = min(100, len(eigs_real))
    ax.plot(range(k_plot), eigs_real[:k_plot], "b-", label="Real", alpha=0.8)
    ax.plot(range(k_plot), eigs_syn[:k_plot], "r--", label="Synthetic", alpha=0.8)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue Spectrum (top 100)")
    ax.legend()

    ax = axes[1]
    ax.scatter(corr_real[mask], corr_syn[mask], s=1, alpha=0.1)
    lim = max(abs(corr_real[mask]).max(), abs(corr_syn[mask]).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_xlabel("Real pairwise Spearman")
    ax.set_ylabel("Synthetic pairwise Spearman")
    ax.set_title(f"Correlation Scatter (MAE={metrics['correlation_mae']:.3f})")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_dir / "correlation_fidelity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved correlation_fidelity.png")

    # ------------------------------------------------------------------
    # 7d: PCA variance explained curve
    # ------------------------------------------------------------------
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumulative) + 1), cumulative * 100, "b-o", ms=3)
    ax.axhline(95, color="red", ls="--", lw=0.8, label="95% threshold")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("PCA Variance Explained (in copula space)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "pca_variance_explained.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pca_variance_explained.png")

    # ------------------------------------------------------------------
    # 7e: Marginal distribution comparison (random subset of columns)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    sample_cols = rng.choice(d, size=min(8, d), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (ax, col_idx) in enumerate(zip(axes.flat, sample_cols)):
        ax.hist(X_real[:, col_idx], bins=30, alpha=0.5, density=True, label="Real")
        ax.hist(X_syn[:, col_idx], bins=30, alpha=0.5, density=True, label="Synthetic")
        ax.set_title(column_names[col_idx][:20], fontsize=9)
        if idx == 0:
            ax.legend(fontsize=8)
    plt.suptitle("Marginal Distributions: Real vs Synthetic", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "marginal_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved marginal_comparison.png")

    return metrics


# ============================================================================
# STEP 8: Compare with Baseline (No PCA)
# ============================================================================
# To demonstrate that PCA actually helps, we also fit a standard copula
# directly on all d features (with corr_clipped) and compare the eigenvalue
# distortion.


def compare_with_baseline(
    Z: np.ndarray,
    output_dir: Path,
    pca_k: int,
) -> None:
    """Compare eigenvalue clipping distortion: full d-dim vs PCA k-dim.

    This directly shows the benefit of dimensionality reduction: the full
    d-dim correlation requires aggressive clipping (many negative eigenvalues),
    while the PCA k-dim correlation is well-conditioned.
    """
    n, d = Z.shape
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON: full d={d} vs PCA k={pca_k}")
    print(f"{'='*60}")

    # -- Full d-dimensional Spearman correlation --
    ranks_full = np.apply_along_axis(rankdata, 0, Z)
    rho_full = np.corrcoef(ranks_full, rowvar=False)
    corr_full = 2.0 * np.sin(np.pi / 6.0 * rho_full)

    eigs_full_raw = np.linalg.eigvalsh(corr_full)
    n_neg_full = int((eigs_full_raw < 0).sum())

    corr_full_clipped = corr_clipped(corr_full, threshold=1e-5)
    diff_full = corr_full_clipped - corr_full
    mask = np.triu(np.ones_like(diff_full, dtype=bool), k=1)
    frob_full = float(np.linalg.norm(diff_full, "fro"))
    frob_raw_full = float(np.linalg.norm(corr_full, "fro"))

    print(f"\n  Full {d}x{d} correlation (n={n}, d/n={d/n:.1f}):")
    print(f"    Negative eigenvalues: {n_neg_full} / {d}")
    print(f"    Most negative: {eigs_full_raw.min():.6f}")
    print(f"    Clipping distortion: {frob_full/frob_raw_full*100:.4f}%")

    # -- PCA k-dimensional correlation (already computed, but redo for comparison) --
    pca_temp = PCA(n_components=pca_k)
    S_temp = pca_temp.fit_transform(Z)
    ranks_pca = np.apply_along_axis(rankdata, 0, S_temp)
    rho_pca = np.corrcoef(ranks_pca, rowvar=False)
    corr_pca = 2.0 * np.sin(np.pi / 6.0 * rho_pca)

    eigs_pca_raw = np.linalg.eigvalsh(corr_pca)
    n_neg_pca = int((eigs_pca_raw < 0).sum())

    corr_pca_clipped = corr_clipped(corr_pca, threshold=1e-5)
    diff_pca = corr_pca_clipped - corr_pca
    frob_pca = float(np.linalg.norm(diff_pca, "fro"))
    frob_raw_pca = float(np.linalg.norm(corr_pca, "fro"))

    print(f"\n  PCA {pca_k}x{pca_k} correlation (n={n}, k/n={pca_k/n:.2f}):")
    print(f"    Negative eigenvalues: {n_neg_pca} / {pca_k}")
    print(f"    Most negative: {eigs_pca_raw.min():.6f}")
    print(f"    Clipping distortion: {frob_pca/frob_raw_pca*100:.4f}%")

    # -- Plot comparison --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    idx_full = np.arange(len(eigs_full_raw))
    ax.plot(idx_full, np.sort(eigs_full_raw), label=f"Full d={d}", alpha=0.7)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Eigenvalue index (sorted)")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Full correlation: {n_neg_full}/{d} negative eigenvalues\n"
                 f"(n={n}, d/n={d/n:.1f})")
    ax.legend()

    ax = axes[1]
    idx_pca = np.arange(len(eigs_pca_raw))
    ax.plot(idx_pca, np.sort(eigs_pca_raw), label=f"PCA k={pca_k}", alpha=0.7, color="green")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Eigenvalue index (sorted)")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"PCA correlation: {n_neg_pca}/{pca_k} negative eigenvalues\n"
                 f"(n={n}, k/n={pca_k/n:.2f})")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved baseline_comparison.png")


# ============================================================================
# MAIN: Full Pipeline
# ============================================================================


def run(
    real_path: str,
    output_dir: str,
    event_column: str | None = None,
    event_type: str | None = None,
    n_components: int | None = None,
    variance_threshold: float = 0.95,
    n_synthetic: int = 200,
    max_zero_frac: float = 0.4,
    use_power_transform: bool = True,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PCA + GAUSSIAN COPULA SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # --- Step 1: Load data ---
    print("\n[Step 1] Loading and preparing data...")
    X_real, peptide_cols, df = load_and_prepare_data(
        real_path, event_column, event_type, max_zero_frac
    )
    n, d = X_real.shape
    print(f"  Data matrix: {n} samples x {d} features")
    print(f"  d/n ratio: {d/n:.1f} ({'PROBLEMATIC' if d > n else 'OK'} for direct copula)")

    # --- Step 2: Fit marginals ---
    pt_label = "with Yeo-Johnson power transform" if use_power_transform else "raw scale"
    print(f"\n[Step 2] Fitting marginal distributions ({pt_label})...")
    marginals = fit_marginals(X_real, peptide_cols, use_power_transform=use_power_transform)

    # --- Step 3: Transform to copula space ---
    print("\n[Step 3] Transforming to Gaussian copula space...")
    Z = transform_to_copula_space(X_real, marginals)
    print(f"  Z shape: {Z.shape}, Z range: [{Z.min():.2f}, {Z.max():.2f}]")

    # --- Step 4: PCA in copula space ---
    print("\n[Step 4] Applying PCA in copula space...")
    pca, k = fit_pca(Z, n_components=n_components, variance_threshold=variance_threshold)
    S = pca.transform(Z)
    print(f"  PCA scores shape: {S.shape}")

    # --- Step 5: Fit copula on PCA scores ---
    print("\n[Step 5] Fitting Gaussian copula on PCA scores...")
    copula_dist, corr_matrix = fit_copula_on_pca_scores(S)

    # --- Step 6: Generate synthetic data ---
    print(f"\n[Step 6] Generating {n_synthetic} synthetic samples...")
    X_syn = generate_synthetic_data(
        copula_dist, pca, marginals, n_synthetic, clip_non_negative=True
    )
    print(f"  Synthetic data shape: {X_syn.shape}")

    # --- Step 7: Evaluate ---
    print("\n[Step 7] Evaluating synthetic data quality...")
    metrics = evaluate_synthetic_data(X_real, X_syn, peptide_cols, out, pca)

    # --- Step 8: Baseline comparison ---
    compare_with_baseline(Z, out, pca_k=k)

    # --- Save synthetic data ---
    syn_df = pl.DataFrame(X_syn, schema=peptide_cols)
    syn_df.write_csv(out / "synthetic_data.csv")
    print(f"\nSaved synthetic data to {out / 'synthetic_data.csv'}")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Real data:       {n} samples x {d} features")
    print(f"  PCA components:  {k} (explains {sum(pca.explained_variance_ratio_)*100:.1f}% variance)")
    print(f"  Synthetic data:  {n_synthetic} samples x {d} features")
    print(f"  Correlation MAE: {metrics['correlation_mae']:.4f}")
    print(f"  Correlation RMSE:{metrics['correlation_rmse']:.4f}")
    print(f"  Marginal mean relative MAE: {metrics['marginal_mean_relative_mae']:.4f}")
    print(f"  Output: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA + Gaussian Copula for high-dimensional synthetic data generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--real", required=True, help="Path to the real data CSV."
    )
    parser.add_argument(
        "--output", default="evaluation_output/pca_copula",
        help="Output directory for results and figures.",
    )
    parser.add_argument(
        "--event-column", default=None,
        help="Column for event-type stratification (e.g., 'event_type').",
    )
    parser.add_argument(
        "--event-type", default=None,
        help="Event type to filter to (e.g., 'hf'). Requires --event-column.",
    )
    parser.add_argument(
        "--n-components", type=int, default=None,
        help="Number of PCA components (overrides --variance-threshold).",
    )
    parser.add_argument(
        "--variance-threshold", type=float, default=0.95,
        help="Auto-select PCA components to explain this fraction of variance (default: 0.95).",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=200,
        help="Number of synthetic samples to generate (default: 200).",
    )
    parser.add_argument(
        "--max-zero-frac", type=float, default=0.4,
        help="Drop peptide columns with zero fraction above this (default: 0.4).",
    )
    parser.add_argument(
        "--no-power-transform", action="store_true", default=False,
        help="Disable Yeo-Johnson power transform on marginals (fit directly on raw scale).",
    )
    args = parser.parse_args()

    run(
        real_path=args.real,
        output_dir=args.output,
        event_column=args.event_column,
        event_type=args.event_type,
        n_components=args.n_components,
        variance_threshold=args.variance_threshold,
        n_synthetic=args.n_synthetic,
        max_zero_frac=args.max_zero_frac,
        use_power_transform=not args.no_power_transform,
    )


if __name__ == "__main__":
    main()

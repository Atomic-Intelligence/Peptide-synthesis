"""Custom marginal distribution wrappers for the copula model (improvement 1.3).

Provides:
- ZeroInflatedGamma       — point mass at 0 + Gamma continuous tail
- ZeroInflatedLognormal   — point mass at 0 + LogNormal continuous tail
- KDEDistribution         — non-parametric kernel density estimate (fallback)

All classes expose the same interface as scipy frozen distributions:
    .cdf(x), .ppf(u), .rvs(size), .pdf(x), .logpdf(x)

They can be instantiated from a flat parameter tuple so they integrate with the
existing EstimatedMarginalDistribution / MarginalDistributionInfo storage pattern:
    ZeroInflatedGamma(p0, a, loc, scale)
    ZeroInflatedLognormal(p0, s, loc, scale)
    KDEDistribution(*grid_x_then_grid_cdf)   # 2*N_GRID floats
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_EPS = 1e-8


def _clip_u(u: np.ndarray, eps: float = _EPS) -> np.ndarray:
    return np.clip(u, eps, 1.0 - eps)


# ──────────────────────────────────────────────────────────────────────────────
# Zero-inflated Gamma
# ──────────────────────────────────────────────────────────────────────────────

class ZeroInflatedGamma:
    """Frozen zero-inflated Gamma distribution.

    Mixture model:
        P(X = 0)       = p0
        P(X | X > 0)  ~ Gamma(a, loc=loc, scale=scale)

    CDF:
        F(x) = 0                                         for x < 0
        F(0) = p0
        F(x) = p0 + (1 - p0) * Gamma_CDF(x, ...)        for x > 0

    PPF:
        Q(u) = 0                                          for u ≤ p0
        Q(u) = Gamma_PPF((u - p0) / (1 - p0), ...)       for u > p0

    Parameters (positional, for serialisation compatibility):
        p0    — zero-inflation probability  ∈ (0, 1)
        a     — Gamma shape > 0
        loc   — location (usually 0 for peptide concentrations)
        scale — Gamma scale > 0
    """

    name = "zero_inflated_gamma"

    def __init__(self, p0: float, a: float, loc: float = 0.0, scale: float = 1.0):
        self.p0 = float(np.clip(p0, _EPS, 1.0 - _EPS))
        self.a = float(a)
        self.loc = float(loc)
        self.scale = float(scale)
        self._cont = stats.gamma(a, loc=loc, scale=scale)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, data: np.ndarray) -> tuple[float, float, float, float]:
        """MLE-style fit. Returns (p0, a, loc, scale)."""
        p0 = float(np.clip(np.mean(data == 0), _EPS, 1.0 - _EPS))
        nonzero = data[data > 0]
        if len(nonzero) < 5:
            std = float(np.std(data[data != 0])) if np.any(data != 0) else 1.0
            return (p0, 1.0, 0.0, max(std, _EPS))
        a, loc, scale = stats.gamma.fit(nonzero, floc=0)
        return (p0, float(a), float(loc), float(scale))

    # ── Distribution interface ────────────────────────────────────────────────

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(
            x < 0.0,
            0.0,
            np.where(x == 0.0, self.p0, self.p0 + (1.0 - self.p0) * self._cont.cdf(x)),
        )

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return np.where(
            u <= self.p0,
            0.0,
            self._cont.ppf(_clip_u((u - self.p0) / (1.0 - self.p0))),
        )

    def rvs(self, size: int = 1) -> np.ndarray:
        rng = np.random.default_rng()
        is_zero = rng.random(size) < self.p0
        cont_samples = self._cont.rvs(size=size)
        return np.where(is_zero, 0.0, cont_samples)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Mixed log-likelihood: log(p0) for zeros, log((1-p0)*f(x)) for positives."""
        x = np.asarray(x, dtype=float)
        result = np.full(x.shape, -np.inf, dtype=float)
        zero_mask = x == 0.0
        result[zero_mask] = np.log(self.p0)
        nz = ~zero_mask
        if nz.any():
            result[nz] = np.log(1.0 - self.p0) + self._cont.logpdf(x[nz])
        return result

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(np.asarray(x, dtype=float)))


# ──────────────────────────────────────────────────────────────────────────────
# Zero-inflated LogNormal
# ──────────────────────────────────────────────────────────────────────────────

class ZeroInflatedLognormal:
    """Frozen zero-inflated LogNormal distribution.

    Analogous to ZeroInflatedGamma but uses LogNormal for the continuous part.

    Parameters (positional):
        p0    — zero-inflation probability  ∈ (0, 1)
        s     — LogNormal shape (sigma of underlying normal) > 0
        loc   — location (usually 0)
        scale — LogNormal scale (= exp(mu) of underlying normal) > 0
    """

    name = "zero_inflated_lognormal"

    def __init__(self, p0: float, s: float, loc: float = 0.0, scale: float = 1.0):
        self.p0 = float(np.clip(p0, _EPS, 1.0 - _EPS))
        self.s = float(s)
        self.loc = float(loc)
        self.scale = float(scale)
        self._cont = stats.lognorm(s, loc=loc, scale=scale)

    @classmethod
    def fit(cls, data: np.ndarray) -> tuple[float, float, float, float]:
        """MLE-style fit. Returns (p0, s, loc, scale)."""
        p0 = float(np.clip(np.mean(data == 0), _EPS, 1.0 - _EPS))
        nonzero = data[data > 0]
        if len(nonzero) < 5:
            return (p0, 1.0, 0.0, 1.0)
        s, loc, scale = stats.lognorm.fit(nonzero, floc=0)
        return (p0, float(s), float(loc), float(scale))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(
            x < 0.0,
            0.0,
            np.where(x == 0.0, self.p0, self.p0 + (1.0 - self.p0) * self._cont.cdf(x)),
        )

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return np.where(
            u <= self.p0,
            0.0,
            self._cont.ppf(_clip_u((u - self.p0) / (1.0 - self.p0))),
        )

    def rvs(self, size: int = 1) -> np.ndarray:
        rng = np.random.default_rng()
        is_zero = rng.random(size) < self.p0
        cont_samples = self._cont.rvs(size=size)
        return np.where(is_zero, 0.0, cont_samples)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        result = np.full(x.shape, -np.inf, dtype=float)
        zero_mask = x == 0.0
        result[zero_mask] = np.log(self.p0)
        nz = ~zero_mask
        if nz.any():
            result[nz] = np.log(1.0 - self.p0) + self._cont.logpdf(x[nz])
        return result

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(np.asarray(x, dtype=float)))


# ──────────────────────────────────────────────────────────────────────────────
# KDE Distribution
# ──────────────────────────────────────────────────────────────────────────────

class KDEDistribution:
    """Non-parametric kernel density estimate with piecewise-linear CDF/PPF.

    Fits a Gaussian KDE and pre-computes a CDF on a grid for fast evaluation.
    Used as a fallback when all parametric distributions fail a KS goodness-of-fit
    test.

    Serialisation: parameters are stored as a flat list [*grid_x, *grid_cdf]
    (2 * N_GRID floats).  The constructor accepts this flat form via `*params`.

    Usage:
        # Fit
        kde_dist = KDEDistribution.fit(data)
        params = kde_dist.to_params()          # list[float]

        # Reconstruct from params
        kde_dist2 = KDEDistribution(*params)   # same as KDEDistribution.from_params(params)
    """

    name = "kde"
    _N_GRID = 500  # Grid density; 500 balances precision vs serialisation size.

    def __init__(self, *params: float):
        """Construct from a flat parameter list [*grid_x, *grid_cdf]."""
        n = len(params) // 2
        if n < 2:
            raise ValueError(
                f"KDEDistribution expects at least 4 params (2 grid_x + 2 grid_cdf), "
                f"got {len(params)}."
            )
        self._grid_x = np.array(params[:n], dtype=float)
        self._grid_cdf = np.array(params[n:], dtype=float)
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        self._cdf_fn = interp1d(
            self._grid_x,
            self._grid_cdf,
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, 1.0),
        )
        # PPF requires monotone CDF
        unique = np.concatenate([[True], np.diff(self._grid_cdf) > 0])
        self._ppf_fn = interp1d(
            self._grid_cdf[unique],
            self._grid_x[unique],
            kind="linear",
            bounds_error=False,
            fill_value=(self._grid_x[0], self._grid_x[-1]),
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, data: np.ndarray, n_grid: int = _N_GRID) -> "KDEDistribution":
        """Fit Gaussian KDE and build the grid CDF."""
        kde = stats.gaussian_kde(data)
        std_est = float(np.std(data))
        lo = float(np.min(data)) - 3.0 * std_est
        hi = float(np.max(data)) + 3.0 * std_est
        grid_x = np.linspace(lo, hi, n_grid)
        pdf_vals = kde.evaluate(grid_x)
        dx = grid_x[1] - grid_x[0]
        cdf_vals = np.cumsum(pdf_vals) * dx
        cdf_vals = cdf_vals / cdf_vals[-1]  # normalise to [0, 1]
        flat_params = grid_x.tolist() + cdf_vals.tolist()
        return cls(*flat_params)

    @classmethod
    def from_params(cls, params: list[float]) -> "KDEDistribution":
        return cls(*params)

    def to_params(self) -> list[float]:
        return self._grid_x.tolist() + self._grid_cdf.tolist()

    # ── Distribution interface ────────────────────────────────────────────────

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._cdf_fn(np.asarray(x, dtype=float))

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self._ppf_fn(np.asarray(u, dtype=float))

    def rvs(self, size: int = 1) -> np.ndarray:
        u = np.random.uniform(0.0, 1.0, size=size)
        return self.ppf(u)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = self._grid_x[1] - self._grid_x[0]
        pdf_approx = np.gradient(self._grid_cdf, dx)
        pdf_fn = interp1d(
            self._grid_x,
            np.maximum(pdf_approx, 1e-300),
            kind="linear",
            bounds_error=False,
            fill_value=1e-300,
        )
        return pdf_fn(x)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(self.pdf(x), 1e-300))


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_DISTRIBUTION_REGISTRY: dict[str, type] = {
    ZeroInflatedGamma.name: ZeroInflatedGamma,
    ZeroInflatedLognormal.name: ZeroInflatedLognormal,
    KDEDistribution.name: KDEDistribution,
}

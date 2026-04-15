import numpy as np
import polars as pl
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed
from tqdm import tqdm
from loguru import logger

_BOUNDARY_CORRECTIONS = ("none", "clip", "reflect")


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope for joblib to pickle it.
# ---------------------------------------------------------------------------

def _fit_column(
    col_values: np.ndarray,
    bandwidth: str | float,
    lower_bound: float | None,
    handle_zero_inflation: bool,
) -> dict | None:
    """Fit a single column.

    Returns None if the column should be skipped entirely (all-NaN or KDE
    fit failure).  Otherwise returns a dict with keys:
        zero_frac  : float
        kind       : "zero_only" | "constant" | "kde"
        constant   : float | None
        kde_data   : np.ndarray | None  — training points stored for sampling
        kde_bw     : float | None       — kernel std dev in data units
        warning    : str | None
    """
    col_values = col_values[~np.isnan(col_values)]
    if len(col_values) == 0:
        return None

    # ── Zero-inflation ────────────────────────────────────────────────────
    zero_frac = 0.0
    if handle_zero_inflation and lower_bound is not None:
        at_bound = col_values == lower_bound
        zero_frac = float(at_bound.mean())
        fit_values = col_values[~at_bound]
    else:
        fit_values = col_values

    base = dict(zero_frac=zero_frac, constant=None, kde_data=None, kde_bw=None, warning=None)

    # All values at the lower bound — no KDE needed.
    if len(fit_values) == 0:
        return {**base, "kind": "zero_only"}

    # Degenerate: too few or zero-variance values.
    if len(fit_values) < 2 or np.std(fit_values) == 0:
        return {
            **base,
            "kind": "constant",
            "constant": float(fit_values[0]),
            "warning": f"insufficient variance — using constant ({float(fit_values[0]):.4g})",
        }

    try:
        kde = gaussian_kde(fit_values, bw_method=bandwidth)
        # Store training data + bandwidth so generate() can sample with pure
        # numpy instead of calling kde.resample() in a Python loop.
        kde_bw = float(kde.factor * np.std(fit_values, ddof=1))
        return {**base, "kind": "kde", "kde_data": fit_values.copy(), "kde_bw": kde_bw}
    except Exception as exc:
        return {**base, "kind": None, "warning": f"KDE fit failed: {exc}"}


class KDEImputation:
    """KDE-based imputation for sparse / zero-inflated columns.

    Uses scipy.stats.gaussian_kde at fit time, but stores the training points
    and bandwidth so that generate() samples with pure numpy — equivalent to
    kde.resample() but without the per-column scipy call overhead.

    Parameters
    ----------
    column_names : list[str]
    bandwidth : str | float
        Passed to gaussian_kde bw_method.  "scott" or "silverman" use the
        respective rule-of-thumb estimator; a float sets the bandwidth factor.
    boundary_correction : str
        How to handle probability mass that leaks below lower_bound:
        "none"    — no correction.
        "clip"    — clamp samples to [lower_bound, +inf).
        "reflect" — reflect samples below the bound back above it
                    (2*lower_bound - sample).  Recommended for biological data.
    lower_bound : float | None
        Hard lower boundary.  None disables boundary correction and
        zero-inflation handling.
    handle_zero_inflation : bool
        Model the point mass at exactly lower_bound separately; fit KDE only
        on strictly-above-bound values.  Recommended for sparse peptide data.
    n_jobs : int
        Parallel workers for fit.  -1 uses all cores.  Uses threads (scipy
        releases the GIL during KDE fitting).
    """

    def __init__(
        self,
        column_names: list[str],
        bandwidth: str | float = "scott",
        boundary_correction: str = "reflect",
        lower_bound: float | None = 0.0,
        handle_zero_inflation: bool = True,
        n_jobs: int = -1,
    ):
        if boundary_correction not in _BOUNDARY_CORRECTIONS:
            raise ValueError(
                f"boundary_correction must be one of {_BOUNDARY_CORRECTIONS}, "
                f"got {boundary_correction!r}"
            )
        self.column_names = column_names
        self.bandwidth = bandwidth
        self.boundary_correction = boundary_correction
        self.lower_bound = lower_bound
        self.handle_zero_inflation = handle_zero_inflation
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame):
        # Extract all numpy arrays in the main thread before going parallel.
        logger.info("Extracting column arrays...")
        col_arrays = [
            (col, data[col].cast(pl.Float64).to_numpy())
            for col in tqdm(self.column_names, desc="Extracting columns")
        ]

        logger.info(f"Fitting KDEs for {len(col_arrays)} columns (n_jobs={self.n_jobs})...")
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_column)(arr, self.bandwidth, self.lower_bound, self.handle_zero_inflation)
            for _, arr in col_arrays
        )

        self.zero_fractions: dict[str, float] = {}
        self.constants: dict[str, float] = {}
        self.kde_data: dict[str, np.ndarray] = {}
        self.kde_bw: dict[str, float] = {}

        valid_columns: list[str] = []
        for (col, _), result in zip(col_arrays, results):
            if result is None:
                logger.warning(
                    f"Column '{col}' is entirely NaN — skipping imputation for this column."
                )
                continue
            if result["kind"] is None:
                logger.warning(f"Column '{col}': {result['warning']} — skipping.")
                continue
            if result["warning"]:
                logger.warning(f"Column '{col}': {result['warning']}.")

            self.zero_fractions[col] = result["zero_frac"]
            if result["kind"] == "constant":
                self.constants[col] = result["constant"]
            elif result["kind"] == "kde":
                self.kde_data[col] = result["kde_data"]
                self.kde_bw[col] = result["kde_bw"]
            # "zero_only" needs no extra state — generate() uses zero_frac=1.0.
            valid_columns.append(col)

        self.column_names = valid_columns
        logger.success("KDE imputation fitting done!")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _sample_col(self, col: str, n: int) -> np.ndarray:
        """Sample n values for one column, applying boundary correction.

        Uses stored training data + bandwidth so the inner work is pure numpy
        (equivalent to kde.resample but without per-call scipy overhead).
        """
        if col in self.constants:
            return np.full(n, self.constants[col])

        train = self.kde_data[col]                         # (n_train,)
        bw = self.kde_bw[col]
        idx = np.random.randint(0, len(train), size=n)
        samples = train[idx] + np.random.normal(0.0, bw, size=n)

        if self.lower_bound is not None:
            if self.boundary_correction == "reflect":
                below = samples < self.lower_bound
                samples[below] = 2.0 * self.lower_bound - samples[below]
            elif self.boundary_correction == "clip":
                samples = np.clip(samples, self.lower_bound, None)

        return samples

    def generate(self, n: int) -> tuple[list[str], np.ndarray | None]:
        if not self.column_names:
            return self.column_names, None

        generated_data = []
        for col in self.column_names:
            zero_frac = self.zero_fractions.get(col, 0.0)
            use_zero_inflation = (
                self.handle_zero_inflation
                and self.lower_bound is not None
                and zero_frac > 0.0
            )

            if use_zero_inflation:
                is_zero = np.random.random(n) < zero_frac
                samples = np.empty(n)
                samples[is_zero] = self.lower_bound  # type: ignore[index]
                n_nonzero = int((~is_zero).sum())
                if n_nonzero > 0:
                    samples[~is_zero] = self._sample_col(col, n_nonzero)
            else:
                samples = self._sample_col(col, n)

            generated_data.append(samples[:, None])

        return self.column_names, np.concatenate(generated_data, axis=-1)

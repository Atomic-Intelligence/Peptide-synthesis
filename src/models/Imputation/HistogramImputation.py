import numpy as np
<<<<<<< HEAD
from numpy.random import choice
import polars as pl
from tqdm import tqdm
from loguru import logger


class HistogramImputation:
    def __init__(self, column_names: list[str], num_bins: int):
        self.column_names = column_names
        self.num_bins = num_bins

    def fit(self, data: pl.DataFrame):
        self.col_densities = {col: [] for col in self.column_names}
        self.medians = {col: [] for col in self.column_names}

        for col in tqdm(self.column_names, desc="Calculating histogram imputation..."):
            col_values = data.select(col).to_numpy()
            hist, bin_edges = np.histogram(col_values, bins=self.num_bins)
            for i in range(len(bin_edges) - 1):
                up, low = bin_edges[i + 1], bin_edges[i]
                bin_idx = np.logical_and(col_values >= low, col_values <= up)
                if bin_idx.any():
                    self.medians[col].append(np.median(col_values[bin_idx]))
                else:
                    self.medians[col].append((up + low) / 2)

            self.col_densities[col] = hist / np.sum(hist)
        logger.success("Imputation done!")

    def generate(self, n: int) -> tuple[list[str], np.ndarray]:
        if len(self.column_names) == 0:
            return self.column_names, None

        generated_data = [
            choice(self.medians[col], p=self.col_densities[col], size=n)[:, None]
            for col in self.column_names
        ]
        generated_data = np.concatenate(generated_data, axis=-1)
        return self.column_names, generated_data
=======
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm
from loguru import logger

_SAMPLING_STRATEGIES = ("uniform", "median")


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope for joblib to pickle it.
# ---------------------------------------------------------------------------

def _fit_column(
    col_values: np.ndarray,
    num_bins: int,
    sampling_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Fit a single column.  Returns None if the column is all-NaN."""
    col_values = col_values[~np.isnan(col_values)]
    if len(col_values) == 0:
        return None

    hist, edges = np.histogram(col_values, bins=num_bins)
    densities = hist / hist.sum()

    medians = None
    if sampling_strategy == "median":
        # np.digitize with interior edges → 0-indexed bin membership,
        # matching np.histogram's half-open [low, up) convention.
        bin_indices = np.digitize(col_values, edges[1:-1])
        medians = np.empty(num_bins)
        for i in range(num_bins):
            mask = bin_indices == i
            medians[i] = (
                np.median(col_values[mask]) if mask.any() else (edges[i] + edges[i + 1]) / 2
            )

    return densities, edges, medians


class HistogramImputation:
    """Histogram-based imputation for sparse / zero-inflated columns.

    sampling_strategy:
        "uniform" (default) — sample uniformly within the chosen bin.
            Produces a piecewise-uniform (continuous) marginal; avoids the
            discrete-spike artefact of median-only sampling.
        "median" — sample the per-bin median (original behaviour).
            Use this only when imputed values must be observed data points.

    n_jobs:
        Number of parallel workers used during fit.  -1 uses all CPU cores.
        Uses threads (numpy releases the GIL), so there is no pickling overhead.
    """

    def __init__(
        self,
        column_names: list[str],
        num_bins: int,
        sampling_strategy: str = "uniform",
        n_jobs: int = -1,
    ):
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        if sampling_strategy not in _SAMPLING_STRATEGIES:
            raise ValueError(
                f"sampling_strategy must be one of {_SAMPLING_STRATEGIES}, got {sampling_strategy!r}"
            )
        self.column_names = column_names
        self.num_bins = num_bins
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame):
        # Extract all numpy arrays up-front in the main thread to avoid
        # passing a Polars DataFrame across thread boundaries.
        logger.info("Extracting column arrays...")
        col_arrays = [
            (col, data[col].cast(pl.Float64).to_numpy())
            for col in tqdm(self.column_names, desc="Extracting columns")
        ]

        logger.info(f"Fitting histograms for {len(col_arrays)} columns (n_jobs={self.n_jobs})...")
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_column)(arr, self.num_bins, self.sampling_strategy)
            for _, arr in col_arrays
        )

        self.col_densities: dict[str, np.ndarray] = {}
        self.bin_edges: dict[str, np.ndarray] = {}
        if self.sampling_strategy == "median":
            self.medians: dict[str, np.ndarray] = {}

        valid_columns: list[str] = []
        for (col, _), result in zip(col_arrays, results):
            if result is None:
                logger.warning(
                    f"Column '{col}' is entirely NaN — skipping imputation for this column."
                )
                continue
            densities, edges, medians = result
            self.col_densities[col] = densities
            self.bin_edges[col] = edges
            if medians is not None:
                self.medians[col] = medians
            valid_columns.append(col)

        self.column_names = valid_columns
        logger.success("Imputation done!")

    # ------------------------------------------------------------------
    # Generate  (fully vectorised — no Python loop over columns)
    # ------------------------------------------------------------------

    def generate(self, n: int) -> tuple[list[str], np.ndarray | None]:
        if not self.column_names:
            return self.column_names, None

        d = len(self.column_names)

        # Stack per-column arrays.  All columns share the same num_bins so
        # shapes are uniform and stacking is safe.
        all_densities = np.stack(
            [self.col_densities[col] for col in self.column_names]
        )  # (d, num_bins)
        all_edges = np.stack(
            [self.bin_edges[col] for col in self.column_names]
        )  # (d, num_bins+1)

        # Vectorised inverse-CDF bin sampling.
        # searchsorted processes n values per column in C — far faster than
        # d separate np.random.choice calls.
        cumsum = np.cumsum(all_densities, axis=1)           # (d, num_bins)
        u = np.random.random((d, n))                        # (d, n)
        chosen = np.array(
            [np.searchsorted(cumsum[i], u[i]) for i in range(d)]
        )                                                   # (d, n)
        chosen = np.clip(chosen, 0, self.num_bins - 1)

        row_idx = np.arange(d)[:, None]                    # (d, 1)  broadcast helper

        if self.sampling_strategy == "uniform":
            low  = all_edges[row_idx, chosen]              # (d, n)
            high = all_edges[row_idx, chosen + 1]          # (d, n)
            result = np.random.uniform(low, high)          # (d, n)
        else:  # median
            all_medians = np.stack(
                [self.medians[col] for col in self.column_names]
            )  # (d, num_bins)
            result = all_medians[row_idx, chosen]          # (d, n)

        return self.column_names, result.T                 # (n, d)
>>>>>>> troubleshooting

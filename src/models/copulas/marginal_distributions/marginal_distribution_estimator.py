import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import kstest

from src.logger import setup_logger
from src.models.copulas.gaussian_copula.data_models import (
    EstimatedMarginalDistribution,
    MarginalDistributionInfo,
)
from src.models.copulas.marginal_distributions.marginal_distribution_metrics import (
    UnivariateDistributionMetric,
    UnivariateDistribution,
)
from src.models.copulas.marginal_distributions.custom_distributions import (
    ZeroInflatedGamma,
    ZeroInflatedLognormal,
    KDEDistribution,
    CUSTOM_DISTRIBUTION_REGISTRY,
)

logger = setup_logger()

_ENCODING_NORMAL_SCORES = "normal_scores"

_ZERO_INFLATED_CANDIDATES = [ZeroInflatedGamma, ZeroInflatedLognormal]


def get_distribution_class(distribution_name: str) -> UnivariateDistribution:
    # Check custom registry first
    if distribution_name in CUSTOM_DISTRIBUTION_REGISTRY:
        return CUSTOM_DISTRIBUTION_REGISTRY[distribution_name]
    match distribution_name:
        case "stats.beta":
            return stats.beta
        case "stats.norm":
            return stats.norm
        case "stats.lognorm":
            return stats.lognorm
        case "stats.truncnorm":
            return stats.truncnorm
        case "stats.gamma":
            return stats.gamma
        case "stats.t":
            return stats.t
        case "stats.expon":
            return stats.expon
        case "stats.uniform":
            return stats.uniform
        case _:
            raise ValueError(f"Unknown distribution name: {distribution_name}")


class MarginalDistributionEstimator:
    def __init__(
        self,
        continuous_distributions: list[UnivariateDistribution | str],
        univariate_distribution_metric: UnivariateDistributionMetric,
        categorical_columns: Optional[list[str]] = None,
        categorical_encoding: str = "truncated_gaussian",
        use_zero_inflated: bool = True,
        kde_fallback: bool = False,
        kde_min_ks_pvalue: float = 0.05,
    ):
        """
        Args:
            continuous_distributions: Pool of scipy distributions to try per column.
            univariate_distribution_metric: Metric used to rank distributions
                (higher = better).  BIC is recommended for small n.
            categorical_columns: Columns treated as categorical (skip parametric
                fitting; assigned normal or uniform marginals).
            categorical_encoding: ``"truncated_gaussian"`` or ``"normal_scores"``.
            use_zero_inflated: If True, also try ZeroInflatedGamma and
                ZeroInflatedLognormal for columns that contain exact zeros
                (improvement 1.3).
            kde_fallback: If True, fall back to a non-parametric KDE when the
                best parametric distribution fails a KS goodness-of-fit test at
                the ``kde_min_ks_pvalue`` significance level (improvement 1.3).
            kde_min_ks_pvalue: KS test significance threshold.  If the best
                parametric CDF achieves a KS p-value below this on the data,
                the column is instead modelled with KDEDistribution.
        """
        self.continuous_distributions = [
            get_distribution_class(d) if isinstance(d, str) else d
            for d in continuous_distributions
        ]
        self.univariate_distribution_metric = univariate_distribution_metric
        self.categorical_columns = set(categorical_columns or [])
        self.categorical_encoding = categorical_encoding
        self.use_zero_inflated = use_zero_inflated
        self.kde_fallback = kde_fallback
        self.kde_min_ks_pvalue = kde_min_ks_pvalue

        logger.info(
            f"Initialized MarginalDistributionEstimator with "
            f"{len(continuous_distributions)} distributions, "
            f"metric '{type(univariate_distribution_metric).__name__}', "
            f"categorical_encoding={categorical_encoding!r}, "
            f"use_zero_inflated={use_zero_inflated}, "
            f"kde_fallback={kde_fallback}."
        )

    @staticmethod
    def _process_single_column(
        col_name: str,
        column_data: np.ndarray,
        distributions: list[UnivariateDistribution],
        metric: UnivariateDistributionMetric,
        categorical_columns: set,
        categorical_encoding: str,
        use_zero_inflated: bool,
        kde_fallback: bool,
        kde_min_ks_pvalue: float,
    ) -> EstimatedMarginalDistribution:
        _logger = setup_logger()

        # --- Missing indicator columns: always uniform on [0, 1] ---
        if "missing" in col_name:
            return EstimatedMarginalDistribution(
                distribution=stats.uniform,
                marginal_distribution_info=MarginalDistributionInfo(
                    distribution_name=stats.uniform.name,
                    parameters=[0.0, 1.0],
                    column_name=col_name,
                ),
            )

        # --- Normal-scores-encoded categorical columns: use standard normal ---
        # The copula generates Z ~ N(0,1), so marginal = N(0,1) gives an identity
        # round-trip, and reverse preprocessing maps Z back to categories via Φ(z).
        if col_name in categorical_columns and categorical_encoding == _ENCODING_NORMAL_SCORES:
            return EstimatedMarginalDistribution(
                distribution=stats.norm,
                marginal_distribution_info=MarginalDistributionInfo(
                    distribution_name=stats.norm.name,
                    parameters=[0.0, 1.0],
                    column_name=col_name,
                ),
            )

        # --- Restrict distribution pool for peptide columns ---
        if "Peptide" in col_name:
            candidate_dists = [stats.lognorm, stats.beta, stats.gamma, stats.expon]
        else:
            candidate_dists = distributions

        # --- MLE fitting and metric evaluation (scipy distributions) ---
        distribution_parameters: dict = {}
        distribution_metrics: dict = {}

        for dist in candidate_dists:
            method = "MLE"
            try:
                if "Peptide" in col_name and "missing" not in col_name:
                    params = dist.fit(column_data, method=method, floc=0)
                else:
                    params = dist.fit(column_data, method=method)
            except Exception as e:
                _logger.error(f"Failed to fit '{dist.name}' to '{col_name}': {e}")
                continue

            metric_value = metric.evaluate(dist, params, column_data)
            distribution_parameters[dist] = params
            distribution_metrics[dist] = metric_value

        # --- Zero-inflated distributions (improvement 1.3) ---
        if use_zero_inflated and np.any(column_data == 0):
            for zi_cls in _ZERO_INFLATED_CANDIDATES:
                try:
                    zi_params = zi_cls.fit(column_data)
                    zi_metric = metric.evaluate(zi_cls, zi_params, column_data)
                    distribution_parameters[zi_cls] = zi_params
                    distribution_metrics[zi_cls] = zi_metric
                    _logger.debug(
                        f"'{col_name}': fitted {zi_cls.name} "
                        f"(params={zi_params}, metric={zi_metric:.4f})"
                    )
                except Exception as e:
                    _logger.warning(f"Failed to fit {zi_cls.name} to '{col_name}': {e}")

        if not distribution_parameters:
            raise ValueError(
                f"No valid distribution could be fitted for column '{col_name}'."
            )

        best_dist = max(distribution_metrics, key=distribution_metrics.get)
        best_params = distribution_parameters[best_dist]

        _logger.debug(
            f"'{col_name}': best distribution = "
            f"{getattr(best_dist, 'name', str(best_dist))} "
            f"(params={best_params})"
        )

        # --- KDE fallback (improvement 1.3) ---
        if kde_fallback:
            best_frozen = best_dist(*best_params)
            _, ks_pvalue = kstest(column_data, best_frozen.cdf)
            if ks_pvalue < kde_min_ks_pvalue:
                _logger.info(
                    f"'{col_name}': best parametric KS p-value={ks_pvalue:.4f} "
                    f"< {kde_min_ks_pvalue}; switching to KDE."
                )
                try:
                    kde_dist = KDEDistribution.fit(column_data)
                    kde_params = kde_dist.to_params()
                    return EstimatedMarginalDistribution(
                        distribution=KDEDistribution,
                        marginal_distribution_info=MarginalDistributionInfo(
                            distribution_name=KDEDistribution.name,
                            parameters=kde_params,
                            column_name=col_name,
                        ),
                    )
                except Exception as e:
                    _logger.warning(
                        f"'{col_name}': KDE fallback failed ({e}); "
                        "keeping best parametric model."
                    )

        dist_name = getattr(best_dist, "name", str(best_dist))
        return EstimatedMarginalDistribution(
            distribution=best_dist,
            marginal_distribution_info=MarginalDistributionInfo(
                distribution_name=dist_name,
                parameters=list(best_params),
                column_name=col_name,
            ),
        )

    def fit_marginal_distributions(
        self,
        preprocessed_dataset: pl.DataFrame,
        num_workers: int = 24,
        categorical_columns: Optional[list[str]] = None,
    ) -> list[EstimatedMarginalDistribution]:
        """Fit per-column marginal distributions in parallel.

        Args:
            preprocessed_dataset: Fully preprocessed (numeric) DataFrame.
            num_workers: Max parallel worker processes.
            categorical_columns: Override the estimator's categorical_columns for
                this call (useful when called from GaussianCopulaBasic._fit).
        """
        # Merge constructor-level and call-level categorical columns
        effective_cat_cols = self.categorical_columns | set(categorical_columns or [])

        column_names = preprocessed_dataset.columns
        col_data_list = [
            preprocessed_dataset.get_column(c).to_numpy() for c in column_names
        ]

        max_workers = min(os.cpu_count() // 2, num_workers)
        logger.info(
            f"Fitting marginals for {len(column_names)} columns "
            f"using {max_workers} worker processes."
        )

        worker_func = partial(
            self._process_single_column,
            distributions=self.continuous_distributions,
            metric=self.univariate_distribution_metric,
            categorical_columns=effective_cat_cols,
            categorical_encoding=self.categorical_encoding,
            use_zero_inflated=self.use_zero_inflated,
            kde_fallback=self.kde_fallback,
            kde_min_ks_pvalue=self.kde_min_ks_pvalue,
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(worker_func, column_names, col_data_list))

        distribution_counts = {}
        for r in results:
            name = r.marginal_distribution_info.distribution_name
            distribution_counts[name] = distribution_counts.get(name, 0) + 1
        logger.info(f"Distribution counts: {distribution_counts}")
        logger.success(f"Fitted marginals for {len(results)} columns.")

        return results

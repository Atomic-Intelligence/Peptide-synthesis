import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import polars as pl
from scipy import stats

from v0.src import setup_logger
from v0.src import (
    EstimatedMarginalDistribution,
    MarginalDistributionInfo,
)
from v0.src import (
    UnivariateDistributionMetric,
    UnivariateDistribution,
)

logger = setup_logger()


def get_distribution_class(distribution_name: str) -> UnivariateDistribution:
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
    ):
        self.continuous_distributions = [
            (
                get_distribution_class(distribution_name)
                if isinstance(distribution_name, str)
                else distribution_name
            )
            for distribution_name in continuous_distributions
        ]
        self.univariate_distribution_metric = univariate_distribution_metric

        logger.info(
            f"Initialized MarginalDistributionEstimator with "
            f"{len(continuous_distributions)} distributions and "
            f"metric '{type(univariate_distribution_metric).__name__}'."
        )

    @staticmethod
    def _process_single_column(
        col_name: str,
        column_data: np.ndarray,
        distributions: list[UnivariateDistribution],
        metric: UnivariateDistributionMetric,
    ) -> EstimatedMarginalDistribution:
        _logger = setup_logger()

        distribution_parameters: dict[UnivariateDistribution, tuple[float, ...]] = {}
        distribution_metrics: dict[UnivariateDistribution, float] = {}
        
        if 'missing' in col_name:
            distributions = [stats.uniform]
            
        elif 'Peptide' in col_name:
            distributions = [stats.lognorm, stats.beta, stats.gamma, stats.expon]

        for distribution in distributions:
            method = "MLE"

            try:
                if 'Peptide' in col_name:
                    parameters = distribution.fit(data=column_data, method=method, floc=0)
                else:
                    parameters = distribution.fit(data=column_data, method=method)
           
            except Exception as e:
                _logger.error(
                    f"Failed to fit distribution '{distribution.name}' with method '{method}': /n{e}"
                )
                continue

            metric_value = metric.evaluate(distribution, parameters, column_data)

            distribution_parameters[distribution] = parameters
            distribution_metrics[distribution] = metric_value

        if len(distribution_parameters) == 0:
            raise ValueError(
                f"No valid distribution could be fitted for column '{col_name}'."
            )

        best_distribution = max(distribution_metrics, key=distribution_metrics.get)

        _logger.debug(
            f"Best distribution for column '{col_name}': {best_distribution.name} with params {distribution_parameters[best_distribution]}"
        )

        if 'missing' in col_name:
            return EstimatedMarginalDistribution(
                distribution=stats.uniform,
                marginal_distribution_info=MarginalDistributionInfo(
                    distribution_name=stats.uniform.name,
                    parameters=[0.0, 1.0],
                    column_name=col_name,
                ),
            )

        return EstimatedMarginalDistribution(
            distribution=best_distribution,
            marginal_distribution_info=MarginalDistributionInfo(
                distribution_name=best_distribution.name,
                parameters=list(distribution_parameters[best_distribution]),
                column_name=col_name,
            ),
        )

    def fit_marginal_distributions(
        self,
        preprocessed_dataset: pl.DataFrame,
        num_workers: int = 24,
    ) -> list[EstimatedMarginalDistribution]:
        column_names = preprocessed_dataset.columns

        col_data_list = []

        for col_name in column_names:
            column_data = preprocessed_dataset.get_column(col_name).to_numpy()
            col_data_list.append(column_data)

        num_tasks = len(column_names)

        max_workers = min(os.cpu_count() // 2, num_workers)

        logger.info(
            f"Starting marginal distribution fitting for {num_tasks} columns using {max_workers} worker processes."
        )

        worker_func = partial(
            self._process_single_column,
            distributions=self.continuous_distributions,
            metric=self.univariate_distribution_metric,
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results_iterator = executor.map(worker_func, column_names, col_data_list)
            results = list(results_iterator)

        logger.success(
            "Parallel processing for marginals finished. Aggregating results and logging worker messages."
        )

        logger.success(
            f"Successfully fitted marginal distributions for {len(results)} columns."
        )

        result_distributions = [
            result.marginal_distribution_info.distribution_name for result in results
        ]
        distribution_counts = {
            distribution: result_distributions.count(distribution)
            for distribution in result_distributions
        }

        logger.info(f"Distribution counts: {distribution_counts}")

        return results

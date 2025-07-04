import mlflow
import numpy as np
from loguru import logger

from v0.src import safe_start_run, safe_log_figure
from v0.src import (
    split_into_quantiles,
    get_kaplan_meier_array,
)
from v0.src import (
    plot_kaplan_meier,
)


def survival_analysis(
    run_id: str,
    event_type: str,
    time_to_event_array: np.ndarray,
    survival_score_array: np.ndarray,
    num_quantiles: int = 5,
    num_timesteps: int = 500,
    dataset_type: str = "",
) -> None:
    """
    We first split the time_to_event_array into quantiles,
    based on some survival_score_array. Next we calculate the
    Kaplan-Meier array for  each quantile group, this array contains
    the proportion (percentage/100) of survivors in a given time step.
    """

    logger.info(f"Starting survival analysis for {event_type}")

    quantile_groups = split_into_quantiles(
        array_to_split=time_to_event_array.squeeze(),
        num_quantiles=num_quantiles,
        criterion_array=survival_score_array,
    )
    max_num_days = time_to_event_array.max()
    quantile_survival_arrays = [
        get_kaplan_meier_array(
            quantile, max_num_days=max_num_days, num_timesteps=num_timesteps
        )
        for quantile in quantile_groups
    ]
    time_steps = np.linspace(0, max_num_days, num_timesteps)
    figure = plot_kaplan_meier(quantile_survival_arrays, time_steps)
    if dataset_type != "":
        dataset_type = "_" + dataset_type

    with safe_start_run(run_id=run_id):
        mlflow.log_figure(
            figure, f"survival/{event_type}_kaplan_meier{dataset_type}.png"
        )

    logger.success(f"Finished survival analysis for {event_type}")
    return None

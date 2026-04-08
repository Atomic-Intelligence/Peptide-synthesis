import re

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import spearmanr, false_discovery_control
from sklearn.preprocessing import QuantileTransformer


def split_into_quantiles(
    array_to_split: np.ndarray,
    num_quantiles: int,
    criterion_array: np.ndarray | None = None,
) -> list[np.ndarray]:
    assert len(array_to_split.shape) == 1, "array_to_split needs to be a 1-D array!"
    quantile_transf = QuantileTransformer(n_quantiles=num_quantiles)
    if criterion_array is None:
        criterion_array = array_to_split
    else:
        assert (
            array_to_split.shape == criterion_array.shape
        ), "criterion_array and array_to_split should have the same shape!"
    quantiles = quantile_transf.fit_transform(criterion_array[:, None]).flatten()
    quantile_limits = np.linspace(0, 1, num_quantiles + 1)
    quantile_groups = []
    for i in range(num_quantiles):
        quantile_groups.append(
            array_to_split[
                np.logical_and(
                    quantiles > quantile_limits[i], quantiles < quantile_limits[i + 1]
                )
            ]
        )
    return quantile_groups


def get_kaplan_meier_array(
    time_to_event_array: np.ndarray, max_num_days: int, num_timesteps: int = 500
) -> np.ndarray:
    time_steps = np.linspace(0, max_num_days, num_timesteps)
    num_people = len(time_to_event_array)
    survived = np.ones_like(time_steps)
    for i, t in enumerate(time_steps):
        survived[i] -= (time_to_event_array <= t).astype(float).sum() / num_people
    return survived


def get_event_and_control(
    df: pl.DataFrame, event: str
) -> tuple[pl.DataFrame, pl.DataFrame]:

    event_df = df.filter(pl.col("event_type") == event).drop("event_type")
    control_df = df.filter(pl.col("event_type") == "no_event").drop("event_type")
    return event_df, control_df


def get_peptide_columns(df: pl.DataFrame) -> list[str]:
    pattern = re.compile("peptide", re.IGNORECASE)
    peptide_columns = [col for col in df.columns if re.search(pattern, col) is not None]
    return peptide_columns


def sparse_peptide_columns(
    df: pl.DataFrame, zero_fraction_threshold: float
) -> list[str]:
    """Return peptide columns whose zero-value fraction in *df* exceeds the threshold.

    Parameters
    ----------
    df : pl.DataFrame
        Reference DataFrame (typically the real data).
    zero_fraction_threshold : float
        Columns with a zero fraction strictly above this value are returned.
        E.g. 0.9 means columns where > 90 % of values are zero.
    """
    peptide_cols = get_peptide_columns(df)
    n = len(df)
    sparse = [col for col in peptide_cols if (df[col] == 0).sum() / n > zero_fraction_threshold]
    if sparse:
        logger.info(
            f"Excluding {len(sparse)} sparse peptide columns "
            f"(zero fraction > {zero_fraction_threshold})"
        )
    return sparse


def filer_peptide_by_zero_pecentage(
    df: pl.DataFrame, non_zero_threshold: int | float = 30.0
):
    # get only columns with at least 30% non-zero values
    peptide_columns = get_peptide_columns(df)
    filtered_columns = []
    for col in peptide_columns:
        non_zero_percentage = (df[col] != 0).sum() / len(df) * 100
        if non_zero_percentage >= non_zero_threshold:
            filtered_columns.append(col)

    logger.info(f"Working with {len(filtered_columns)} peptide columns!")

    return filtered_columns + ["event_type"]


def peptide_eGFR_correlation(
    df: pl.DataFrame, adjust_p_value: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    eGFR_peptides_array = df.to_numpy()

    result = spearmanr(eGFR_peptides_array)

    corr = result.statistic[-1, :-1]
    pvalues = result.pvalue[-1, :-1]

    if adjust_p_value:
        pvalues = false_discovery_control(pvalues)
    return corr, pvalues

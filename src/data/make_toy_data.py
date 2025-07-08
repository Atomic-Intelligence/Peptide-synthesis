import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.distributions.copula.api import GaussianCopula, CopulaDistribution

from src.data.utils import (
    CATEGORICAL_CLINICAL_COLUMNS,
    NUMERICAL_CLINICAL_COLUMNS,
    TIME_TO_EVENT_COLUMNS,
)


def make_toy_data(num_samples: int, num_peptide_cols: int = 30) -> pd.DataFrame:
    columns = (
        CATEGORICAL_CLINICAL_COLUMNS
        + NUMERICAL_CLINICAL_COLUMNS
        + TIME_TO_EVENT_COLUMNS
        + [f"Peptide_{i+1}" for i in range(num_peptide_cols)]
    )
    columns.remove("idAuswertung")

    marginals = np.random.choice(
        [stats.gamma(2), stats.lognorm(s=1), stats.expon],
        replace=True,
        size=num_peptide_cols,
    )
    num_numerical_columns = (
        len(NUMERICAL_CLINICAL_COLUMNS) + len(TIME_TO_EVENT_COLUMNS) + num_peptide_cols
    )
    corr = np.clip(
        0.5 * np.random.randn(num_numerical_columns, num_numerical_columns),
        a_min=-1,
        a_max=1,
    )
    corr = np.dot(corr, corr.T)
    corr = corr / np.linalg.norm(corr)

    corr[np.diag_indices(num_numerical_columns)] = np.ones(num_numerical_columns)
    numerical_copula = CopulaDistribution(
        copula=GaussianCopula(corr=corr), marginals=marginals
    )
    numerical_data = numerical_copula.rvs(nobs=num_samples)

    categorical_data = np.random.randint(
        low=0, high=2, size=(num_samples, len(CATEGORICAL_CLINICAL_COLUMNS) - 1)
    )
    data = np.concatenate([categorical_data, numerical_data], axis=1)
    dataframe = pd.DataFrame(data=data, columns=columns)
    return dataframe


if __name__ == "__main__":
    import os

    dataframe = make_toy_data(num_samples=2000, num_peptide_cols=30)
    print(dataframe)
    dataframe.to_csv(f"{os.getcwd()}/example_data/toy_data.csv")

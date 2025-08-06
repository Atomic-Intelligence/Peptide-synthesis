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
    numerical_columns = (
        NUMERICAL_CLINICAL_COLUMNS
        + TIME_TO_EVENT_COLUMNS
        + [f"Peptide_{i+1}" for i in range(num_peptide_cols)]
    )
    numerical_columns.remove("idAuswertung")
    categorical_columns = CATEGORICAL_CLINICAL_COLUMNS
    marginals = np.random.choice(
        [stats.gamma(2), stats.lognorm(s=1), stats.expon],
        replace=True,
        size=num_peptide_cols,
    )

    corr = np.clip(
        0.5 * np.random.randn(len(numerical_columns), len(numerical_columns)),
        a_min=-1,
        a_max=1,
    )
    corr = np.dot(corr, corr.T)
    corr = corr / np.linalg.norm(corr)

    corr[np.diag_indices(len(numerical_columns))] = np.ones(len(numerical_columns))
    numerical_copula = CopulaDistribution(
        copula=GaussianCopula(corr=corr), marginals=marginals
    )
    numerical_data = numerical_copula.rvs(nobs=num_samples)

    categorical_data = np.random.randint(
        low=0, high=2, size=(num_samples, len(CATEGORICAL_CLINICAL_COLUMNS))
    ).astype(int)
    cat_dataframe = pd.DataFrame(data=categorical_data, columns=categorical_columns)
    num_dataframe = pd.DataFrame(data=numerical_data, columns=numerical_columns)
    event_type = np.random.choice(
        ["no_event", "ckd", "hf"], p=[0.6, 0.15, 0.25], size=num_samples
    )
    dataframe = pd.concat([cat_dataframe, num_dataframe], axis=1)
    dataframe["event_type"] = event_type
    dataframe["idAuswertung"] = list(range(100, 100 + num_samples))
    return dataframe


if __name__ == "__main__":
    import os

    dataframe = make_toy_data(num_samples=2000, num_peptide_cols=30)
    print(dataframe)
    dataframe.to_csv(f"{os.getcwd()}/resources/toy_data.csv")

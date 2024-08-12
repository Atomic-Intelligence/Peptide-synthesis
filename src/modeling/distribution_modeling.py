import json
import random
from enum import Enum

import matplotlib.pyplot as plt
import polars as pl
from fitter import Fitter
from tqdm import tqdm


class FitMethod(str, Enum):
    sumsquare = 'sumsquare_error'
    aic = 'aic'
    bic = 'bic'


class Distributions(str, Enum):
    norm = 'norm'
    beta = 'beta'
    lognorm = 'lognorm'
    truncnorm = 'truncnorm'
    gamma = 'gamma'
    t = 't'
    gaussian_kde = 'gaussian_kde'


class DistributionEstimator:
    """`save_to` is the directory where distributions.json will be saved to
    or is loaded from if already exists. If it is None, distributions.json will
    be saved to the current working directory.
    If distributions.json is loaded (not computed), `original_peptides_df` can be set to None.
    """

    def __init__(
            self,
            distribution_list: list[str],
            original_peptides_data: pl.DataFrame | None = None,
            save_to: str | None = None,
            num_of_peptides_to_model: int = 5000,
            method: FitMethod = FitMethod.sumsquare
    ):
        self.results = None
        self.peptides_df = original_peptides_data
        self.save_to = save_to
        self.n = num_of_peptides_to_model
        self.top_n_peptides_df = self._top_n_columns_with_least_missing_values(
            self.peptides_df, self.n
        )
        self.distribution_list = distribution_list
        self.method = method

        # Check if the provided method is a valid FitMethod
        if method not in FitMethod.__members__.values():
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {', '.join(FitMethod.__members__.values())}.")

        # Check if the provided distribution list is valid
        for distribution in self.distribution_list:
            if distribution not in Distributions.__members__.values():
                raise ValueError(
                    f"Invalid distribution '{distribution}'. Must be one of: {', '.join(Distributions.__members__.values())}."
                )

    def estimate(
            self, peptides_to_model: pl.DataFrame
    ) -> dict[str, str]:
        self.results = {}

        for peptide in tqdm(
                self.top_n_peptides_df.columns, desc="Fitting Distributions"
        ):
            self.results[peptide] = self.estimate_single_column_distribution(
                self.top_n_peptides_df[peptide],
            )

        if self.save_to:
            with open(f"{self.save_to}/distributions.json", "w") as f:
                json.dump(self.results, f)
            print(f"Distributions saved to {self.save_to}.")

        distributions = {
            p: d for p, d in self.results.items() if p in peptides_to_model.columns
        }
        return distributions

    def estimate_single_column_distribution(
            self, column: pl.Series
    ) -> str:
        column = column.filter(column != 0)

        # Use the Fitter library to find the best distribution
        f = Fitter(column.to_list(), distributions=self.distribution_list)
        f.fit()

        # Get the name of the best fitting distribution
        best_distributions = f.get_best(method=self.method)
        best_dist_name = list(best_distributions.keys())[0]

        return best_dist_name

    def plot_single_column_distribution(
            self, column: pl.Series
    ):
        """Plot the data and the best fitting distribution."""
        column = column.filter(column != 0)

        # Use the Fitter library to find the best distribution
        f = Fitter(column.to_list(), distributions=self.distribution_list)
        f.fit()

        # Get the name of the best fitting distribution
        best_distributions = f.get_best(method=self.method)
        best_dist_name = list(best_distributions.keys())[0]

        # Plot the data and the best fitting distribution
        f.summary(lw=3, Nbest=1, method=self.method)
        plt.title(f"Best fitting distribution for {column.name}: {best_dist_name}")
        plt.show()

        return best_dist_name

    def demo_results(
            self, column: str | None = None
    ):
        """If `column` is None, random column is drawn."""
        if column is None:
            peptide = random.sample(list(self.results.keys()), 1)[0]

        else:
            peptide = column

        peptides_df = self.top_n_peptides_df[peptide].filter(
            self.top_n_peptides_df[peptide] != 0
        )

        # Use the Fitter library to find the best distribution
        f = Fitter(peptides_df, distributions=self.results[peptide])
        f.fit()

        # Plot the data and the best fitting distribution
        f.summary(lw=3, Nbest=1, method=self.method)
        plt.title(f"Best fitting distribution for peptide {peptide}")
        plt.show()

    def load_distributions(self, peptides_to_model: pl.DataFrame) -> dict:
        with open(f"{self.save_to}/distributions.json", "r") as f:
            self.results = json.load(f)

        distributions = {
            p: d for p, d in self.results.items() if p in peptides_to_model.columns
        }

        return distributions

    @staticmethod
    def _top_n_columns_with_least_missing_values(
            df: pl.DataFrame, n: int
    ) -> pl.DataFrame:
        # Replace 0 with None
        df = df.with_columns(pl.col(col).replace(0, None) for col in df.columns)

        # Calculate the number of missing values in each column
        missing_values = df.select(pl.all().null_count()).to_dicts()[0]

        # Sort the columns by the number of missing values
        sorted_columns = sorted(missing_values.items(), key=lambda x: x[1])

        # Select the top n columns
        top_n_columns = [col for col, _ in sorted_columns[:n]]

        # Select the top n columns from the DataFrame
        df_top_n_columns = df.select(top_n_columns)

        # Fill None values with 0
        df_top_n_columns = df_top_n_columns.fill_null(0)

        return df_top_n_columns

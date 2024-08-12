import json
import os
from enum import Enum

import polars as pl
from fitter import Fitter
from tqdm import tqdm


class FitMethod(str, Enum):
    sumsquare = "sumsquare_error"
    aic = "aic"
    bic = "bic"


class Distributions(str, Enum):
    norm = "norm"
    beta = "beta"
    lognorm = "lognorm"
    truncnorm = "truncnorm"
    gamma = "gamma"
    t = "t"
    gaussian_kde = "gaussian_kde"


class DistributionEstimator:
    """`save_to` is the directory where distributions.json file will be saved to
    or is loaded from if already exists. If it is None, distributions.json will
    not be saved. If distributions.json is loaded (not computed), `peptides_to_model`
    should be set to None.
    """

    def __init__(
        self,
        distribution_list: list[str],
        peptides_to_model: pl.DataFrame | None = None,
        save_to: str | None = None,
        method: FitMethod = FitMethod.sumsquare,
    ):
        self.distribution_list = distribution_list
        self.peptides_df = peptides_to_model
        self.save_to = save_to
        self.method = method

        # Check if the provided method is a valid FitMethod
        if method not in FitMethod.__members__.values():
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {', '.join(FitMethod.__members__.values())}."
            )

        # Check if the provided distribution list is valid
        for distribution in self.distribution_list:
            if distribution not in Distributions.__members__.values():
                raise ValueError(
                    f"Invalid distribution '{distribution}'. Must be one of: {', '.join(Distributions.__members__.values())}."
                )

    def estimate(self) -> dict[str, str]:
        distributions = {}

        for peptide in tqdm(self.peptides_df.columns, desc="Fitting Distributions"):
            distributions[peptide] = self.estimate_single_column_distribution(
                self.peptides_df[peptide],
            )

        if self.save_to:
            with open(os.path.join(self.save_to, "distributions.json"), "w") as f:
                json.dump(distributions, f)
            print(f"Distributions saved to {self.save_to}.")

        self.distributions = distributions
        return distributions

    def estimate_single_column_distribution(self, column: pl.Series) -> str:
        column = column.filter(column != 0)

        # Use the Fitter library to find the best distribution
        f = Fitter(column.to_list(), distributions=self.distribution_list)
        f.fit()

        # Get the name of the best fitting distribution
        best_distributions = f.get_best(method=self.method)
        best_dist_name = list(best_distributions.keys())[0]

        return best_dist_name

    def load_distributions(self) -> dict:
        with open(os.path.join(self.save_to, "distributions.json"), "r") as f:
            self.distributions = json.load(f)

        return self.distributions

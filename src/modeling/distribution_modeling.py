from enum import Enum

import polars as pl
from fitter import Fitter
from tqdm import tqdm


class FitMethod(str, Enum):
    """
    methods for estimation of goodness of fit
    """
    sumsquare = "sumsquare_error"
    aic = "aic"
    bic = "bic"


class Distributions(str, Enum):
    """
    allowed distributions for univariate distribution fitting
    """
    norm = "norm"
    beta = "beta"
    lognorm = "lognorm"
    truncnorm = "truncnorm"
    gamma = "gamma"
    t = "t"
    gaussian_kde = "gaussian_kde"


class DistributionEstimator:
    def __init__(
        self,
        distribution_list: list[str],
        method: str = FitMethod.sumsquare,
    ):
        """
        Used for estimating distributions to use in the gaussian copula for each peptide in the dataset.
        Args:
            distribution_list: list of allowed distributions
            peptides_to_model: dataframe containing the peptides for which optimal distributions should be
                               estimated
            method: metric which should be used to choose the best fitting distribution
        """
        self.distribution_list = distribution_list
        self.method = method

        self.distributions = None

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

    def estimate(self, peptides_df) -> dict[str, str]:
        """
        Estimate the best distributions for all peptides
        Returns: dictionary mapping the peptide column name to the distribution name
        """
        distributions = {}

        for peptide in tqdm(peptides_df.columns, desc="Fitting Distributions"):
            distributions[peptide] = self.estimate_single_column_distribution(
                peptides_df[peptide],
            )
        self.distributions = distributions
        return distributions

    def estimate_single_column_distribution(self, column: pl.Series) -> str:
        """
        Estimate the optimal distribution for a single column
        Args:
            column: column containing the peptide data
        Returns: name of best distribution
        """
        column = column.filter(column != 0)

        # Use the Fitter library to find the best distribution
        f = Fitter(column.to_list(), distributions=self.distribution_list)
        f.fit()

        # Get the name of the best fitting distribution
        best_distributions = f.get_best(method=self.method)
        best_dist_name = list(best_distributions.keys())[0]

        return best_dist_name

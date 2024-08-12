from copulas.univariate import (BetaUnivariate, GammaUnivariate, GaussianUnivariate,
                                GaussianKDE, TruncatedGaussian, UniformUnivariate)
from sdv.single_table import GaussianCopulaSynthesizer

from src.modeling.custom_univariate import LognormUnivariate


class CustomGaussianCopulaSynthesizer(GaussianCopulaSynthesizer):
    _DISTRIBUTIONS = {
        'norm': GaussianUnivariate,
        'beta': BetaUnivariate,
        'truncnorm': TruncatedGaussian,
        'gamma': GammaUnivariate,
        'uniform': UniformUnivariate,
        'gaussian_kde': GaussianKDE,
        'lognorm': LognormUnivariate,  # Include your custom distribution
    }

    def __init__(self, metadata, **kwargs):
        super().__init__(metadata, **kwargs)

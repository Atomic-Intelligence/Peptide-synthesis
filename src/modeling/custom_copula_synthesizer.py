from sdv.single_table import GaussianCopulaSynthesizer
import src.modeling.custom_univariate as custom_uv


class CustomGaussianCopulaSynthesizer(GaussianCopulaSynthesizer):
    _DISTRIBUTIONS = {
        'norm': custom_uv.GaussianUnivariate,
        'beta': custom_uv.BetaUnivariate,
        'truncnorm': custom_uv.TruncatedGaussian,
        'gamma': custom_uv.GammaUnivariate,
        'uniform': custom_uv.UniformUnivariate,
        'gaussian_kde': custom_uv.GaussianKDE,
        'lognorm': custom_uv.LognormUnivariate,  # Include your custom distribution
    }

    def __init__(self, metadata, **kwargs):
        super().__init__(metadata, **kwargs)
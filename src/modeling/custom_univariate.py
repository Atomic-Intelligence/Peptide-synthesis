import numpy as np
from scipy.stats import lognorm

from copulas.univariate.base import BoundedType, ParametricType, ScipyModel
from copulas.univariate import (BetaUnivariate, GammaUnivariate, GaussianUnivariate,
                                GaussianKDE, TruncatedGaussian, StudentTUnivariate,
                                UniformUnivariate, LogLaplace)


class LognormUnivariate(ScipyModel):
    """Wrapper around scipy.stats.lognorm.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
    """

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.BOUNDED
    MODEL_CLASS = lognorm

    def _fit_constant(self, X):
        self._params = {
            'loc': np.unique(X)[0],
            'scale': 0.0,
        }

    def _fit(self, X):
        loc = np.min(X)
        scale = np.max(X) - loc
        s, loc, scale = lognorm.fit(X, loc=loc, scale=scale)
        self._params = {
            'loc': loc,
            'scale': scale,
            's': s
        }

    def _is_constant(self):
        return self._params['scale'] == 0

    def _extract_constant(self):
        return self._params['loc']

# Import everything from copulas.univariate and add custom classes
__all__ = (
    'BetaUnivariate',
    'GammaUnivariate',
    'GaussianKDE',
    'GaussianUnivariate',
    'TruncatedGaussian',
    'StudentTUnivariate',
    'ParametricType',
    'BoundedType',
    'UniformUnivariate',
    'LogLaplace',
    'LognormUnivariate'
)
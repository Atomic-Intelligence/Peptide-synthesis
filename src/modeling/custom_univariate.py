import numpy as np
from scipy.stats import lognorm

from copulas.univariate.base import BoundedType, ParametricType, ScipyModel


class LognormUnivariate(ScipyModel):
    """Wrapper around scipy.stats.lognorm.
    This class wraps scipy.stats.lognorm using the interface from the Copulas library.
    This allows us to use the lognormal distribution in the gaussian copula method.
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


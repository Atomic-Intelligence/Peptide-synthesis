import abc
import numpy as np
from scipy.stats import t


class CopulaGeneratorFunction(abc.ABC):

    @abc.abstractmethod
    def fit(self, X: np.ndarray) -> None:
        return NotImplemented

    @abc.abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return NotImplemented

    @abc.abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return NotImplemented

    @abc.abstractmethod
    def inverse_transform(self, U: np.ndarray) -> np.ndarray:
        return NotImplemented

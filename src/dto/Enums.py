from enum import Enum


class EModelType(str, Enum):
    """
    Enum representing the different types of models that can be used for synthetic data generation.
    """
    DUMMY = "dummy"
    GAUSSIAN_COPULA = "gaussian_copula"
    CTGAN = "ctgan"


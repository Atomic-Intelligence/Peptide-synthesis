import numpy as np
import polars as pl
from sklearn.preprocessing import (
    OneHotEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
)
from typing import List, Optional, Union

Scaler = Union[QuantileTransformer, RobustScaler, StandardScaler, MinMaxScaler]


class FeatureProcessor:
    """Handles scaling of numerical features and one-hot encoding of categoricals.

    Shared by AuthenticityEstimator, DCR, MIA, and the two-sample classifier test.
    Fit on real data, then call transform on both real and synthetic.
    """

    def __init__(
        self,
        scaler: Scaler,
        categorical_columns: Optional[List[str]] = None,
    ):
        self.scaler = scaler
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="warn")
        self.categorical_columns: List[str] = categorical_columns or []
        self.fitted = False
        self.feature_names: Optional[np.ndarray] = None
        self.numerical_feature_names: Optional[List[str]] = None

    def fit(self, dataframe: pl.DataFrame) -> "FeatureProcessor":
        numerical_cols = [
            col for col in dataframe.columns if col not in self.categorical_columns
        ]
        self.scaler.fit(dataframe.select(numerical_cols).to_numpy())
        if self.categorical_columns:
            self.one_hot_encoder.fit(
                dataframe.select(self.categorical_columns).to_numpy()
            )
            categorical_feature_names = self.one_hot_encoder.get_feature_names_out(
                self.categorical_columns
            )
            self.feature_names = np.concatenate([numerical_cols, categorical_feature_names])
        else:
            self.feature_names = np.array(numerical_cols)
        self.numerical_feature_names = numerical_cols
        self.fitted = True
        return self

    def transform(self, dataframe: pl.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("FeatureProcessor must be fitted before transform.")
        numerical_cols = [
            col for col in dataframe.columns if col not in self.categorical_columns
        ]
        numerical = self.scaler.transform(dataframe.select(numerical_cols).to_numpy())
        if not self.categorical_columns:
            return numerical
        categorical = self.one_hot_encoder.transform(
            dataframe.select(self.categorical_columns).to_numpy()
        )
        return np.concatenate([numerical, categorical], axis=-1)

    def fit_transform(self, dataframe: pl.DataFrame) -> np.ndarray:
        self.fit(dataframe)
        return self.transform(dataframe)

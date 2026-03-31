import pickle
import tempfile
from functools import partial
from pathlib import Path
from typing import Union, Optional

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import polars as pl
from arfpy import arf
from loguru import logger
from omegaconf import DictConfig
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    RobustScaler,
    OneHotEncoder,
)

from src.data.utils import (
    NUMERICAL_CLINICAL_COLUMNS,
    TIME_TO_EVENT_COLUMNS,
    CATEGORICAL_CLINICAL_COLUMNS,
    get_peptide_columns,
)
from src.models.synthetization_model_interface import (
    SynthetizationModelInterface,
    MlFlowTrainingRunInfo,
    DatasetMetadata,
)


Scaler = Union[StandardScaler, RobustScaler, QuantileTransformer]


class ARFPipeline(SynthetizationModelInterface):
    def __init__(
        self,
        run_info: MlFlowTrainingRunInfo,
        arf_params: DictConfig,
        scaler: Scaler,
        description: Optional[dict[str, any]] = None,
    ):
        super().__init__(ml_flow_info=run_info, description=description)
        self.scaler = scaler
        self.oh_encoder = OneHotEncoder(sparse_output=False)
        self.model_factory = partial(
            arf.arf,
            min_node_size=arf_params.min_node_size,
            num_trees=arf_params.num_trees,
            max_features=arf_params.max_features,
        )
        self.model: arf.arf | None = None

    def _fit(
        self,
        data: pl.DataFrame,
        dataset_metadata: Optional[DatasetMetadata] = None,
    ):
        logger.info("Starting ARF model fitting...")
        data = data.fill_null(0.0).fill_nan(0.0)

        # Derive column groups from what is actually present in the data.
        # The pipeline already handles event filtering and drops the primary key,
        # so we just need to partition the remaining columns correctly.
        peptide_cols = get_peptide_columns(data)

        self.numerical_columns = peptide_cols + [
            col
            for col in NUMERICAL_CLINICAL_COLUMNS + TIME_TO_EVENT_COLUMNS
            if col in data.columns
        ]
        self.categorical_columns = [
            col
            for col in CATEGORICAL_CLINICAL_COLUMNS + ["event_type"]
            if col in data.columns
        ]

        numerical_data = data.select(self.numerical_columns).to_numpy()
        categorical_data = data.select(self.categorical_columns).to_numpy()

        numerical_features = self.scaler.fit_transform(numerical_data)
        categorical_features = self.oh_encoder.fit_transform(categorical_data)

        self.numerical_feature_names = self.scaler.get_feature_names_out().tolist()
        self.categorical_feature_names = (
            self.oh_encoder.get_feature_names_out().tolist()
        )

        # arfpy requires a pandas DataFrame
        arf_input = pd.DataFrame(
            np.concatenate([numerical_features, categorical_features], axis=-1),
            columns=self.numerical_feature_names + self.categorical_feature_names,
        )

        logger.info("Beginning adversarial training...")
        self.model = self.model_factory(arf_input)
        logger.success("Adversarial training complete!")
        logger.info("Beginning density estimation...")
        stats = self.model.forde()
        logger.success(f"Density estimation complete!\n{stats}")

    def _generate(self, n: int) -> pl.DataFrame:
        assert self.model is not None, "Model has not been fitted. Call .fit() first."

        raw_synthetic = self.model.forge(n=n)

        numerical_data = self.scaler.inverse_transform(
            raw_synthetic[self.numerical_feature_names].to_numpy()
        )
        categorical_data = self.oh_encoder.inverse_transform(
            raw_synthetic[self.categorical_feature_names].to_numpy()
        )

        synthetic_pd = pd.DataFrame(
            data=np.concatenate([numerical_data, categorical_data], axis=-1),
            columns=self.numerical_columns + self.categorical_columns,
        )
        return pl.from_pandas(synthetic_pd)

    @classmethod
    def _load_pretrained_from_mlflow_run(
        cls, run_id: str, ml_flow_info: MlFlowTrainingRunInfo
    ):
        model_uri = f"runs:/{run_id}/arf_model/arf_model.pkl"
        logger.info(f"Loading ARF model from {model_uri}")
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=temp_dir,
            )
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        return model

    def log_model(self):
        logger.info("Logging ARF model to MLflow...")
        with tempfile.TemporaryDirectory() as temp_dir:
            model_pickle_path = Path(temp_dir, "arf_model.pkl")
            with open(model_pickle_path, "wb") as f:
                pickle.dump(self, f)
            mlflow.log_artifact(model_pickle_path, "arf_model")


if __name__ == "__main__":
    arf_obj = ARFPipeline()
    arf_obj

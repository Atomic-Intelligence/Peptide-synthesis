import pickle
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from omegaconf import DictConfig
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.preprocessing import QuantileTransformer

from src.data.utils import CATEGORICAL_CLINICAL_COLUMNS
from src.models.AdversarialRandomForests.ARFPipeline import DataFrame
from src.models.synthetization_model_interface import SynthetizationModelInterface, MlFlowTrainingRunInfo


class CTGANSynthetizationModel(SynthetizationModelInterface):
    def __init__(
            self,
            ml_flow_info: MlFlowTrainingRunInfo,
            gan_params: DictConfig,
            description: Optional[dict[str, any]] = None,
    ):
        super().__init__(ml_flow_info=ml_flow_info, description=description)
        self.gan_params = gan_params
        self.ml_flow_info = ml_flow_info
        self.sdv_metadata = SingleTableMetadata()
        self.description = description
        self.scaler = QuantileTransformer()

    def _fit(self, data: DataFrame, *args, **kwargs):
        logger.info("Starting model fitting...")
        self.all_columns = data.columns
        data = data.fill_null(0.0)
        data = data.fill_nan(0.0)

        self.categorical_columns = CATEGORICAL_CLINICAL_COLUMNS + ["event_type"]
        self.numerical_columns = [col for col in data.columns if col not in self.categorical_columns]

        numerical_data = data.select(self.numerical_columns).to_numpy()
        numerical_features = self.scaler.fit_transform(numerical_data)
        self.numerical_feature_names = self.scaler.get_feature_names_out().tolist()

        self.processed_data = pd.DataFrame(numerical_features, columns=self.numerical_feature_names)
        self.processed_data[self.categorical_columns] = data[self.categorical_columns].to_pandas()

        self.sdv_metadata.detect_from_dataframe(self.processed_data)

        for col in self.categorical_columns:
            if col in self.processed_data.columns:
                self.sdv_metadata.update_column(column_name=col, sdtype='categorical')
        for col in self.numerical_feature_names:
            self.sdv_metadata.update_column(column_name=col, sdtype='numerical')

        self.model = CTGANSynthesizer(metadata=self.sdv_metadata, **self.gan_params)
        logger.info("Beginning CTGAN training...")
        self.model.fit(self.processed_data)
        logger.success("CTGAN training complete!")
        self.fitted = True

    @classmethod
    def _load_pretrained_from_mlflow_run(cls, run_id:str, ml_flow_info: MlFlowTrainingRunInfo):
        experiment = mlflow.get_experiment_by_name(ml_flow_info.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{ml_flow_info.run_name}'",
            output_format="pandas",
        )

        if runs.empty:
            raise ValueError(
                f"No run found with name {ml_flow_info.run_name} in experiment {ml_flow_info.experiment_name}"
            )

        run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{run_id}/ctgan_model/ctgan_model.pkl"
        logger.info(
            f"Artifacts found at uri: {model_uri}\t{mlflow.artifacts.list_artifacts(artifact_uri=model_uri)}"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=temp_dir,
            )

            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)

        return model

    def log_model(self):
        logger.info(f"Logging model to MLflow!")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_pickle_path = Path(temp_dir, "ctgan_model.pkl")
            with open(model_pickle_path, "wb") as model_save_file:
                pickle.dump(self, model_save_file)

            mlflow.log_artifact(model_pickle_path, "ctgan_model")

    def _generate(self, n_synthetic_patients: int) -> DataFrame:
        assert (
                self.model is not None
        ), f"The model has not been properly fitted to the real data! please call the .fit function first."

        if not self.fitted:
            raise RuntimeError("Model not fitted! Call .fit() first.")

        logger.info(f"Generating {n_synthetic_patients} synthetic samples...")
        self.model = self.load_pretrained_from_mlflow_run(self.ml_flow_info)
        synthetic_data = self.model.model.sample(num_rows=n_synthetic_patients)

        # Process numerical features
        numerical_features = synthetic_data[self.numerical_feature_names]
        numerical_data = self.scaler.inverse_transform(numerical_features.to_numpy())

        # Get categorical data
        categorical_data = synthetic_data[self.categorical_columns]

        # Combine all data
        synthetic_dataframe = pd.DataFrame(
            data=np.concatenate([numerical_data, categorical_data], axis=-1),
            columns=self.numerical_columns + self.categorical_columns
        )

        return pl.from_pandas(synthetic_dataframe)

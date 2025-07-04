import os
import pickle
import tempfile
from functools import partial
from pathlib import Path
from typing import Union, Optional, Literal

import hydra
import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import polars as pl
from arfpy import arf
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    RobustScaler,
    OneHotEncoder,
)

from v0.src import (
    NUMERICAL_CLINICAL_COLUMNS,
    TIME_TO_EVENT_COLUMNS,
    CATEGORICAL_CLINICAL_COLUMNS,
)
from v0.src import split_peptide_columns_by_zero_percentage
from v0.src import HistogramImputation
from v0.src import (
    SynthetizationModelInterface,
    MlFlowTrainingRunInfo,
)

# logger.add(sys.stdout, format="{time} - {level}: {message}", level="INFO")


Scaler = Union[StandardScaler, RobustScaler, QuantileTransformer]
DataFrame = Union[pl.DataFrame, pd.DataFrame]
Event = Literal["hf", "ckd", "cad", "no_event", "all"]


class ARFPipeline(SynthetizationModelInterface):
    def __init__(
        self,
        run_info: MlFlowTrainingRunInfo,
        arf_params: DictConfig,
        imputation_params: DictConfig,
        scaler: Scaler,
        event_type: Event = "all",
        zero_percentage_threshold: float = 0.3,
        description: Optional[dict[str, any]] = None,
    ):
        super().__init__(ml_flow_info=run_info, description=description)
        self.zero_percentage_threshold = zero_percentage_threshold
        self.scaler = scaler
        self.oh_encoder = OneHotEncoder(sparse_output=False)
        self.model_factory = partial(
            arf.arf,
            min_node_size=arf_params.min_node_size,
            num_trees=arf_params.num_trees,
            max_features=arf_params.max_features,
        )
        self.model: arf.arf | None = None
        self.imputation = partial(
            HistogramImputation, num_bins=imputation_params.num_bins
        )
        self.event_type = event_type

    def select_event(self, data: DataFrame) -> DataFrame:
        match self.event_type:
            case "hf":
                return data.filter(pl.col("event_type") == "hf")
            case "ckd":
                return data.filter(pl.col("event_type") == "ckd")
            case "cad":
                return data.filter(pl.col("event_type") == "cad")
            case "no_event":
                return data.filter(pl.col("event_type") == "no_event")
            case "all":
                return data

            case _:
                raise ValueError(f"{self.event_type} is not a supported event_type!")

    def _fit(self, data: DataFrame, *args, **kwargs):
        logger.info("Starting model fitting...")
        self.all_columns = data.columns
        # NOTE: below_zero_threshold are passed directly to the synthesis model
        # while the above_zero_threshold are passed to the data imputer (TODO)
        data = self.select_event(data)
        data = data.fill_null(0.0)
        data = data.fill_nan(0.0)

        self.below_zero_threshold, self.above_zero_threshold = (
            split_peptide_columns_by_zero_percentage(
                data, zero_perc_threshold=self.zero_percentage_threshold
            )
        )
        mlflow.log_table(
            {"arf": self.below_zero_threshold, "imputation": self.above_zero_threshold},
            "peptide_generation_source.json",
        )
        self.imputation = self.imputation(column_names=self.above_zero_threshold)
        self.imputation.fit(data.select(self.above_zero_threshold))
        logger.info(f"Peptide data split into low zero and high zero counts")

        # Save the original numerical and categorical columns
        self.numerical_columns = (
            self.below_zero_threshold
            + NUMERICAL_CLINICAL_COLUMNS
            + TIME_TO_EVENT_COLUMNS
        )

        self.categorical_columns = CATEGORICAL_CLINICAL_COLUMNS + ["event_type"]
        # Get numerical and categorical columns
        numerical_data = data.select(self.numerical_columns).to_numpy()
        categorical_data = data.select(self.categorical_columns).to_numpy()

        # Get scaled numerical features and one-hot encoded numerical features.
        numerical_features = self.scaler.fit_transform(numerical_data)
        categorical_features = self.oh_encoder.fit_transform(categorical_data)

        # Save transformed feature names
        self.numerical_feature_names = self.scaler.get_feature_names_out().tolist()
        self.categorical_feature_names = (
            self.oh_encoder.get_feature_names_out().tolist()
        )

        data_array = np.concatenate([numerical_features, categorical_features], axis=-1)

        # NOTE: arfpy model requires a pandas DataFrame as input
        df = pd.DataFrame(
            data_array,
            columns=self.numerical_feature_names + self.categorical_feature_names,
        )
        logger.info(f"Beginning adversarial training...")
        self.model = self.model_factory(df)
        logger.success(f"Adversarial training complete!")
        logger.info(f"Beinning density estimation!")
        stats = self.model.forde()
        logger.success(f"Density estimation complete!\n{stats}")

    @classmethod
    def _load_pretrained_from_mlflow_run(cls, run_id: str, ml_flow_info: MlFlowTrainingRunInfo):
        model_uri = f"runs:/{run_id}/arf_model/arf_model.pkl"
        logger.info(
            f"Artifacts found at uri: {model_uri}\t{mlflow.artifacts.list_artifacts(artifact_uri=model_uri)}"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = mlflow.artifacts.download_artifacts(
                # run_id=run_id, artifact_path="arf_model",
                artifact_uri=model_uri,
                dst_path=temp_dir,
            )

            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)

        return model

    def log_model(self):
        logger.info(f"Logging model to MLflow!")

        with tempfile.TemporaryDirectory("wb") as temp_dir:
            model_pickle_path = Path(temp_dir, "arf_model.pkl")
            with open(model_pickle_path, "wb") as model_save_file:
                pickle.dump(self, model_save_file)

            mlflow.log_artifact(model_pickle_path, "arf_model")

    def _generate(self, n: int) -> DataFrame:
        assert (
            self.model is not None
        ), f"The model has not been properly fitted to the real data! please call the .fit function first."

        raw_synthetic = self.model.forge(n=n)
        numerical_features = raw_synthetic[self.numerical_feature_names]
        categorical_features = raw_synthetic[self.categorical_feature_names]
        numerical_data = self.scaler.inverse_transform(numerical_features.to_numpy())
        categorical_data = self.oh_encoder.inverse_transform(
            categorical_features.to_numpy()
        )

        imputed_col_names, imputed_data = self.imputation.generate(n=n)
        synthetic_dataframe = pd.DataFrame(
            data=np.concatenate(
                [numerical_data, imputed_data, categorical_data], axis=-1
            ),
            columns=self.numerical_columns
            + imputed_col_names
            + self.categorical_columns,
        )
        return synthetic_dataframe


@hydra.main(
    version_base="1.1", config_path="../../../configs", config_name="arf_configs.yaml"
)
def main(cfg: DictConfig) -> None:
    # print(
    #    f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    # )

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    scaler = hydra.utils.instantiate(cfg.scaler)
    training_run_info = MlFlowTrainingRunInfo(
        experiment_name=cfg.mlflow.experiment_name
    )
    logger.info(cfg.description)
    arf_params_dict = OmegaConf.to_container(cfg.arf_params)
    imputation_param_dict = OmegaConf.to_container(cfg.imputation_params)
    description_dict = dict(
        **{
            "description": cfg.description,
            "zero_percentage_threshold": cfg.zero_percentage_threshold,
        },
        **arf_params_dict,
        **imputation_param_dict,
    )

    model = ARFPipeline(
        run_info=training_run_info,
        arf_params=cfg.arf_params,
        imputation_params=cfg.imputation_params,
        scaler=scaler,
        zero_percentage_threshold=float(cfg.zero_percentage_threshold),
        description=description_dict,
        event_type=cfg.event_type,
    )

    data = pl.read_csv(cfg.paths.real_data_path)

    mlflow_run_info = model.fit(data)
    logger.info(
        f"Completed experiment: {mlflow_run_info.experiment_name}\nCompleted run {mlflow_run_info.run_name}"
    )
    # model = ARFPipeline.load_pretrained_from_mlflow_run(mlflow_run_info)
    generated_data: DataFrame = model.generate(n_synthetic_patients=1700)
    save_dir = f"{cfg.paths.save_data_path}/arf_event_{cfg.event_type}_{mlflow_run_info.run_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = Path(save_dir, cfg.save_name)
    generated_data.to_csv(save_path)


if __name__ == "__main__":
    main()

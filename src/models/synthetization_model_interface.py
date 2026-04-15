import abc
from typing import Optional
from pathlib import Path

import json
import tempfile
import mlflow
import polars as pl
from pydantic import BaseModel

from src.logger import setup_logger

logger = setup_logger()


class DatasetMetadata(BaseModel):
    peptide_ids: list[str]
    patient_ids: list[str]
    full_dataset_path: str
    n_rows: int


class MlFlowTrainingRunInfo(BaseModel):
    """
    Represents an MlFlow training run
    """

    experiment_name: str
    run_name: Optional[str] = None


class SynthetizationModelInterface(abc.ABC):
    def __init__(
        self,
        ml_flow_info: MlFlowTrainingRunInfo,
        description: Optional[dict[str, any]] = None,
    ):
        self.fitted = False
        self.ml_flow_info = ml_flow_info
        self.description = description

    def fit(
        self,
        real_dataset: pl.DataFrame,
        dataset_metadata: Optional[DatasetMetadata] = None,
    ) -> MlFlowTrainingRunInfo:
        if self.fitted:
            raise RuntimeError("Model is already fitted")

        # get or create experiment id for given experiment name
        logger.info(f"{self.ml_flow_info}")
        experiment_id, _ = self.get_or_create_run(self.ml_flow_info)

        if "idAuswertung" in real_dataset.columns:
            real_dataset = real_dataset.drop("idAuswertung")

        # create mlflow run for the given experiment with the given run name
        # mlflow.set_experiment(self.ml_flow_info.experiment_name)
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=self.ml_flow_info.run_name
        ) as active_run:
            self.log_description()
            self.log_metadata(dataset_metadata)

            self._fit(real_dataset, dataset_metadata)  # internal fit method
            self.log_model()  # log the model

            self.fitted = True  # we are now fitted

            return MlFlowTrainingRunInfo(
                experiment_name=self.ml_flow_info.experiment_name,
                run_name=self.ml_flow_info.run_name,
            )

    def log_description(self):
        if self.description is None:
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the JSON file path
            json_path = Path(temp_dir, "description.json")

            # Save the config as JSON
            with open(json_path, "w") as f:
                json.dump(self.description, f)

            mlflow.log_artifact(json_path.as_posix(), "description")

    @staticmethod
    def log_metadata(metadata: DatasetMetadata):
        """Log the dataset metadata as a JSON artifact to MLflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define JSON file path
            json_path = Path(temp_dir) / "dataset_metadata.json"

            # Serialize DatasetMetadata to JSON
            with open(json_path, "w") as f:
                json.dump(metadata.model_dump(), f)

            # Log as MLflow artifact
            mlflow.log_artifact(json_path.as_posix(), artifact_path="metadata")

    @abc.abstractmethod
    def _fit(
        self,
        real_dataset: pl.DataFrame,
        dataset_metadata: Optional[DatasetMetadata] = None,
    ):
        """
        Internal fit method that should be implemented by the subclasses
        """
        pass

    def generate(self, n_synthetic_patients: int) -> pl.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before generating synthetic data")

        return self._generate(n_synthetic_patients)

    @abc.abstractmethod
    def _generate(self, n_synthetic_patients: int) -> pl.DataFrame:
        """
        Internal generate method that should be implemented by the subclasses
        """
        pass

    @abc.abstractmethod
    def log_model(self):
        """
        Internal log model method that should be implemented by the subclasses.
        Should save the model in the given mlflow run, allowing it to be loaded later.
        """
        pass

    @classmethod
    def load_pretrained_from_mlflow_run(cls, ml_flow_info: MlFlowTrainingRunInfo):
        """
        Load pretrained model from mlflow run.
        """
        run_id = cls.get_run_id(ml_flow_info)

        model = cls._load_pretrained_from_mlflow_run(
            run_id=run_id, ml_flow_info=ml_flow_info
        )
        model.fitted = True

        return model

    @classmethod
    def get_run_id(cls, ml_flow_info: MlFlowTrainingRunInfo):
        experiment = mlflow.get_experiment_by_name(ml_flow_info.experiment_name)
        if experiment is None:
            raise ValueError(
                f"MLflow experiment '{ml_flow_info.experiment_name}' not found."
            )

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{ml_flow_info.run_name}'",
            max_results=1,
            output_format="pandas",
        )

        if runs.empty:
            raise ValueError(
                f"MLflow run '{ml_flow_info.run_name}' not found in experiment "
                f"'{ml_flow_info.experiment_name}'."
            )
        run_id = runs.iloc[0].run_id

        return run_id

    @classmethod
    @abc.abstractmethod
    def _load_pretrained_from_mlflow_run(
        cls, run_id: str, ml_flow_info: MlFlowTrainingRunInfo
    ):
        """
        Internal load pretrained method that should be implemented by the subclasses.
        Should load the model from the given mlflow run, allowing it to be used for inference.
        """
        pass

    def get_or_create_run(
        self, ml_flow_info: MlFlowTrainingRunInfo
    ) -> tuple[str, str | None]:
        # get experiment by name
        experiment = mlflow.get_experiment_by_name(ml_flow_info.experiment_name)

        # if experiment is not found, create it
        if experiment is None:
            experiment_id = mlflow.create_experiment(ml_flow_info.experiment_name)

        # otherwise, get the experiment id
        else:
            experiment_id = experiment.experiment_id

        # search for runs with the given experiment id and run name
        run = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = '{ml_flow_info.run_name}'",
        )

        if not self.fitted:
            if not run.empty:
<<<<<<< HEAD
                # if the run already exists, raise an error, as we care about ensuring run names are unique
                raise RuntimeError(
                    f"Model {ml_flow_info.experiment_name}/{ml_flow_info.run_name} already exists!"
                )
=======
                # auto-version the run name if it already exists: name -> name_v1 -> name_v2 -> ...
                base_name = ml_flow_info.run_name
                version = 1
                while not run.empty:
                    versioned_name = f"{base_name}_v{version}"
                    run = mlflow.search_runs(
                        experiment_ids=[experiment_id],
                        filter_string=f"tags.mlflow.runName = '{versioned_name}'",
                    )
                    version += 1
                ml_flow_info.run_name = versioned_name
>>>>>>> troubleshooting
            return experiment_id, None

        else:
            if not run.empty:
                return experiment_id, run.iloc[0].run_id
            else:
                # if the model is fitted, there should be a run with the given experiment and run name
                raise RuntimeError(
                    f"Model {ml_flow_info.experiment_name}/{ml_flow_info.run_name} does not exist!"
                )

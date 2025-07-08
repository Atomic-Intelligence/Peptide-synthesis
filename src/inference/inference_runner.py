import uuid
from pathlib import Path

import mlflow
import polars as pl

from src.logger import setup_logger
from src.models.synthetization_model_interface import (
    MlFlowTrainingRunInfo,
    SynthetizationModelInterface,
)

logger = setup_logger()


class InferenceRunner:
    def __init__(
        self,
        ml_flow_info: MlFlowTrainingRunInfo,
        model: SynthetizationModelInterface,
        synthetic_dataset_path: Path | str,
        inference_run_name: str,
        n_synthetic_patients: int,
    ):
        self.ml_flow_info = ml_flow_info

        self.model = model

        self.synthetic_dataset_path = Path(synthetic_dataset_path)
        self.inference_run_name = (
            f"{inference_run_name}_{uuid.uuid4()}"  # ensure uniqueness
        )

        self.n_synthetic_patients = n_synthetic_patients

    def run(self, n_synthetic_patients: int) -> pl.DataFrame:
        # get the experiment id and run id for the given training mlflow run
        experiment_id, training_mlflow_run = self.model.get_or_create_run(
            self.ml_flow_info
        )

        # enter the original training mlflow run
        with mlflow.start_run(experiment_id=experiment_id, run_id=training_mlflow_run):
            # create an inference run with the training run as a parent
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=self.inference_run_name,
                nested=True,
            ):
                # generate a synthetic dataset and save it to the synthetic dataset path
                synthetic_patients = self.model.generate(n_synthetic_patients)
                self.save_synthetic_dataset(synthetic_patients)

                return synthetic_patients

    def save_synthetic_dataset(self, synthetic_patients: pl.DataFrame):
        sd_path = self.synthetic_dataset_path / f"{uuid.uuid4()}.csv"

        synthetic_patients.write_csv(sd_path)
        logger.success(f"Synthetic dataset saved to {self.synthetic_dataset_path}")

        mlflow.log_artifact(sd_path.as_posix(), artifact_path="synthetic_dataset")
        logger.success(f"Artifact synthetic_dataset saved to mlflow")

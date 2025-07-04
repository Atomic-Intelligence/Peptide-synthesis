import mlflow
import polars as pl

from v0.src import setup_logger
from v0.src import (
    SynthetizationModelInterface,
    DatasetMetadata,
    MlFlowTrainingRunInfo,
)

logger = setup_logger()

class DummySynthetizationModel(SynthetizationModelInterface):
    def __init__(self, ml_flow_info: MlFlowTrainingRunInfo):
        super().__init__(ml_flow_info=ml_flow_info)
        self.synthetic_dataset = None

    def _fit(self, real_dataset: pl.DataFrame, dataset_metadata: DatasetMetadata):
        mlflow.log_metric("a", 1)
        self.synthetic_dataset = real_dataset

        logger.info(
            f"Dummy model fitted on {len(self.synthetic_dataset)} synthetic patients"
        )

    def _generate(self, n_synthetic_patients: int) -> pl.DataFrame:
        mlflow.log_metric("b", 2)
        return self.synthetic_dataset.sample(
            n_synthetic_patients, with_replacement=True
        )

    def log_model(self):
        logger.info(
            f"Dummy logging model to MLFlow with experiment: {self.ml_flow_info.experiment_name}, "
            f"run: {self.ml_flow_info.run_name}"
        )

    @classmethod
    def _load_pretrained_from_mlflow_run(cls, ml_flow_info: MlFlowTrainingRunInfo):
        logger.info(
            f"Dummy loading model from MLFlow with experiment ID: {ml_flow_info.experiment_name}, "
            f"run ID: {ml_flow_info.run_name}"
        )
        model = cls(ml_flow_info=ml_flow_info)
        model.fitted = True
        model.synthetic_dataset = pl.DataFrame(
            {"patient_id": ["synthetic_patient_1", "synthetic_patient_2"]}
        )

        return model

import uuid
from pathlib import Path

import mlflow
import polars as pl

from v0.src import EModelType
from v0.src import InferenceRunner
from v0.src import DatasetMetadata, MlFlowTrainingRunInfo
from tests.mock.dummy_model import DummySynthetizationModel


# this represents a simple end-to-end integration test for the entire pipeline
# it sets up a dummy model, fits it on a synthetic dataset, and then runs inference on it
# this test is used to ensure the experiment tracking functionality is working as expected

def main():
    mlflow.set_tracking_uri("http://10.100.111.210:44555")

    real_dataset = pl.DataFrame(
        {
            "patient_id": ["patient_1", "patient_2", "patient_3"],
            "feature_1": [0.5, 1.0, 0.2],
            "feature_2": [0.1, 0.2, 0.3],
        }
    )
    dataset_metadata = DatasetMetadata(
        peptide_ids=["peptide_1", "peptide_2", "peptide_3"],
        patient_ids=["patient_1", "patient_2", "patient_3"],
        full_dataset_path=Path("path/to/dataset"),
    )

    ml_flow_info = MlFlowTrainingRunInfo(
        experiment_name=f"dummy_experiment_{uuid.uuid4()}",  # uuid to allow multiple runs
        run_name="dummy_run",
    )
    model = DummySynthetizationModel(ml_flow_info=ml_flow_info)

    model.fit(real_dataset, dataset_metadata)

    inference_runner = InferenceRunner(
        ml_flow_info,
        model_type=EModelType.DUMMY,
        synthetic_dataset_path=Path(
            "/home/stipe/atomic/peptide-synthesis-research/sd.csv"
        ),
        inference_run_name="small dataset",
    )

    inference_runner.run(n_synthetic_patients=20)

    inference_runner_2 = InferenceRunner(
        ml_flow_info,
        model_type=EModelType.DUMMY,
        synthetic_dataset_path=Path(
            "/home/stipe/atomic/peptide-synthesis-research/sd.csv"
        ),
        inference_run_name="large dataset",
    )
    inference_runner_2.run(n_synthetic_patients=200)


if __name__ == "__main__":
    main()

import uuid
from pathlib import Path

import mlflow
import polars as pl
from scipy import stats

from src.data.utils import (
    DataProcessor,
    CATEGORICAL_CLINICAL_COLUMNS,
    NUMERICAL_CLINICAL_COLUMNS,
)
from src.logger import setup_logger
from src.models.copulas.gaussian_copula.gaussian_copula_basic import GaussianCopulaBasic
from src.models.copulas.gaussian_copula.preprocessing import SDVPreprocessor
from src.models.copulas.marginal_distributions.marginal_distribution_estimator import (
    MarginalDistributionEstimator,
)
from src.models.copulas.marginal_distributions.marginal_distribution_metrics import (
    KullbackLeiblerDivergence, JensenShannonDivergence,
)
from src.models.synthetization_model_interface import (
    DatasetMetadata,
    MlFlowTrainingRunInfo,
)

logger = setup_logger()


def main():
    """Example usage of the GaussianCopulaBasic model"""
    data = pl.read_csv(
        "/data1/prostrat-ai/data/merged_peptide_and_clinical.csv",
    )

    processor = DataProcessor(
        data, clinical_columns=CATEGORICAL_CLINICAL_COLUMNS + NUMERICAL_CLINICAL_COLUMNS
    )

    processed_data, _ = (
        processor.filter_peptides(non_zero_threshold=40.0)
        .split_event_control(event="hf")
        .get_processed_data()
    )[0]

    logger.info(processed_data.shape)

    metadata = DatasetMetadata(
        peptide_ids=processed_data.columns,
        patient_ids=[],
        full_dataset_path=Path(
            "/data1/prostrat-ai/data/merged_peptide_and_clinical.csv"
        ).as_posix(),
        n_rows=len(processed_data),
    )

    copula = GaussianCopulaBasic(
        sdv_preprocessor=SDVPreprocessor(),
        marginal_distribution_estimator=MarginalDistributionEstimator(
            continuous_distributions=[
                stats.beta,
                stats.norm,
                stats.lognorm,
                stats.truncnorm,
                stats.gamma,
                stats.t,
                stats.expon,
                stats.uniform,
            ],
            univariate_distribution_metric=KullbackLeiblerDivergence(),
        ),
        ml_flow_info=MlFlowTrainingRunInfo(
            experiment_name="peptide-synthesis-research-test-3",
            run_name=f"test-gaussian-copula-{uuid.uuid4()}",
        ),
    )
    copula.fit(processed_data, metadata)
    synth = copula.generate(n_synthetic_patients=300)

    logger.info(f"Synthetic data:\n{synth}")

    synth.write_csv("synthetic_copula_student_t.csv")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://10.100.111.210:5002/")
    main()

import tempfile
import json
import hydra
from pathlib import Path
from omegaconf import DictConfig
import polars as pl
import mlflow
from src.evaluation.privacy.AuthenticityEstimator import AuthenticityEstimator
from src.data.utils import (
    split_peptide_columns_by_zero_percentage,
    CATEGORICAL_CLINICAL_COLUMNS,
    NUMERICAL_CLINICAL_COLUMNS,
)
from loguru import logger


@hydra.main(
    version_base="1.1",
    config_path="../../../configs/",
    config_name="evaluation/privacy_script.yaml",
)
def main(cfg: DictConfig):
    cfg = cfg.evaluation
    authenticity_estimator: AuthenticityEstimator = hydra.utils.instantiate(
        cfg.authenticity_estimator
    )
    logger.info(f"Loading Real Data")
    real_data = pl.read_csv(cfg.paths.real_data_path).fill_nan(0.0).fill_null(0.0)
    logger.success("Sucessfuly Loaded Real Data")
    logger.info(f"Loading Synthetic Data From {cfg.paths.synthetic_data_path}")
    synth_data = pl.read_csv(cfg.paths.synthetic_data_path).fill_nan(0.0).fill_null(0.0)
    logger.success("Sucessfuly Loaded Synthetic Data")
    peptides_to_keep, peptides_to_drop = split_peptide_columns_by_zero_percentage(
        real_data, zero_perc_threshold=cfg.zero_percentage
    )
    real_data = real_data.drop(peptides_to_drop)
    synth_data = synth_data.drop(peptides_to_drop)

    real_data = (
        real_data.select(
            NUMERICAL_CLINICAL_COLUMNS
            + peptides_to_keep
            + CATEGORICAL_CLINICAL_COLUMNS
            + ["event_type"]
        )
        .fill_null(0.0)
        .fill_nan(0.0)
    )
    synth_data = (
        synth_data.select(
            NUMERICAL_CLINICAL_COLUMNS
            + peptides_to_keep
            + CATEGORICAL_CLINICAL_COLUMNS
            + ["event_type"]
        )
        .fill_null(0.0)
        .fill_nan(0.0)
    )
    logger.info(f"Real: {real_data.shape}\tSynth {synth_data.shape}")

    logger.info(f"Fitting nearest neighbors and features...")
    authenticity_estimator.fit(real_dataframe=real_data)

    logger.info(f"Estimating Authenticity...")
    results = authenticity_estimator.estimate_authenticity(
        synthetic_dataframe=synth_data,
        real_dataframe=real_data,
        return_suspicious=True,
        feature_importance=True,
    )
    logger.success(f"Estimated Data Authenticity!")
    logger.info("Composing report...")
    figure = authenticity_estimator.visualize_authenticity(results=results)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run() as run:
        logger.info(f"Logging report to MLflow")
        mlflow.log_text(cfg.description, "description.txt")
        mlflow.log_figure(figure, "results.png")
        mlflow.log_dict(dict(results), "privacy_result_dict")

        with tempfile.TemporaryDirectory("wb") as temp_dir:
            authenticity_report = Path(temp_dir, "report.json")

        with open(authenticity_report, "wb") as model_save_file:
            json.dump(results.model_dump(), fp=model_save_file)

        mlflow.log_artifact(local_path=temp_dir)

    logger.success(f"All done, Goodbye!")


if __name__ == "__main__":
    main()

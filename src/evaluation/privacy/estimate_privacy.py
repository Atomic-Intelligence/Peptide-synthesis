import tempfile
import json
import hydra
from pathlib import Path
from omegaconf import DictConfig
import polars as pl
import mlflow
<<<<<<< HEAD
from src.evaluation.privacy.AuthenticityEstimator import AuthenticityEstimator
=======
from sklearn.preprocessing import RobustScaler

from src.evaluation.privacy.AuthenticityEstimator import AuthenticityEstimator
from src.evaluation.privacy.dcr import DCREstimator
from src.evaluation.privacy.reidentification_risk import ReidentificationRiskEstimator
>>>>>>> troubleshooting
from src.data.utils import (
    split_peptide_columns_by_zero_percentage,
    CATEGORICAL_CLINICAL_COLUMNS,
    NUMERICAL_CLINICAL_COLUMNS,
)
<<<<<<< HEAD
from src.mlflow import start_or_connect_mlflow_server
from loguru import logger


=======
from src.mlflow_utils import start_or_connect_mlflow_server
from loguru import logger


def _prepare_data(cfg):
    """Load, filter, and align real and synthetic datasets."""
    logger.info("Loading Real Data")
    real_data = pl.read_csv(cfg.paths.real_data_path).fill_nan(0.0).fill_null(0.0)
    logger.success("Loaded Real Data")
    logger.info(f"Loading Synthetic Data from {cfg.paths.synthetic_data_path}")
    synth_data = pl.read_csv(cfg.paths.synthetic_data_path).fill_nan(0.0).fill_null(0.0)
    logger.success("Loaded Synthetic Data")

    peptides_to_keep, peptides_to_drop = split_peptide_columns_by_zero_percentage(
        real_data, zero_perc_threshold=cfg.zero_percentage
    )
    real_data = real_data.drop(peptides_to_drop)
    synth_data = synth_data.drop(peptides_to_drop)

    feature_cols = (
        NUMERICAL_CLINICAL_COLUMNS
        + peptides_to_keep
        + CATEGORICAL_CLINICAL_COLUMNS
        + ["event_type"]
    )
    real_data = real_data.select(feature_cols).fill_null(0.0).fill_nan(0.0)
    synth_data = synth_data.select(feature_cols).fill_null(0.0).fill_nan(0.0)

    logger.info(f"Real: {real_data.shape}  Synth: {synth_data.shape}")
    return real_data, synth_data


>>>>>>> troubleshooting
@hydra.main(
    version_base="1.1",
    config_path="../../../configs/",
    config_name="evaluation/privacy_script.yaml",
)
def main(cfg: DictConfig):
    cfg = cfg.evaluation
    shutdown_hook = start_or_connect_mlflow_server(cfg.mlflow.tracking_uri)
<<<<<<< HEAD
    
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
=======

    real_data, synth_data = _prepare_data(cfg)

    # ------------------------------------------------------------------ #
    #  1. Authenticity (existing KNN-based estimator)                      #
    # ------------------------------------------------------------------ #
    authenticity_estimator: AuthenticityEstimator = hydra.utils.instantiate(
        cfg.authenticity_estimator
    )
    logger.info("Fitting AuthenticityEstimator...")
    authenticity_estimator.fit(real_dataframe=real_data)

    logger.info("Estimating Authenticity...")
    auth_results = authenticity_estimator.estimate_authenticity(
>>>>>>> troubleshooting
        synthetic_dataframe=synth_data,
        real_dataframe=real_data,
        return_suspicious=True,
        feature_importance=True,
    )
<<<<<<< HEAD
    logger.success(f"Estimated Data Authenticity!")
    logger.info("Composing report...")
    figure = authenticity_estimator.visualize_authenticity(results=results)
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
=======
    logger.success("Authenticity estimated.")

    auth_figure = authenticity_estimator.visualize_authenticity(results=auth_results)

    # ------------------------------------------------------------------ #
    #  2. DCR                                                              #
    # ------------------------------------------------------------------ #
    dcr_results = None
    dcr_figure = None
    run_dcr = getattr(cfg, "run_dcr", True)
    if run_dcr:
        logger.info("Running DCR estimation...")
        dcr_cfg = cfg.get("dcr", {}) if hasattr(cfg, "get") else {}
        dcr_estimator = DCREstimator(
            scaler=RobustScaler(),
            categorical_columns=CATEGORICAL_CLINICAL_COLUMNS + ["event_type"],
            holdout_fraction=dcr_cfg.get("holdout_fraction", 0.5)
            if isinstance(dcr_cfg, dict) else getattr(dcr_cfg, "holdout_fraction", 0.5),
            par_percentile=dcr_cfg.get("par_percentile", 5.0)
            if isinstance(dcr_cfg, dict) else getattr(dcr_cfg, "par_percentile", 5.0),
            distance_metric=dcr_cfg.get("distance_metric", "euclidean")
            if isinstance(dcr_cfg, dict) else getattr(dcr_cfg, "distance_metric", "euclidean"),
        )
        dcr_estimator.fit(real_data)
        dcr_results = dcr_estimator.estimate(synth_data)
        dcr_figure = dcr_estimator.plot(dcr_results)
        logger.success("DCR estimation complete.")

    # ------------------------------------------------------------------ #
    #  3. Re-identification Risk                                           #
    # ------------------------------------------------------------------ #
    reid_results = None
    reid_figure = None
    run_reid = getattr(cfg, "run_reidentification", True)
    if run_reid:
        logger.info("Running Re-identification Risk estimation...")
        reid_cfg = cfg.get("reidentification", {}) if hasattr(cfg, "get") else {}
        reid = ReidentificationRiskEstimator(
            scaler=RobustScaler(),
            categorical_columns=CATEGORICAL_CLINICAL_COLUMNS + ["event_type"],
            risk_threshold=reid_cfg.get("risk_threshold", 2.0)
            if isinstance(reid_cfg, dict) else getattr(reid_cfg, "risk_threshold", 2.0),
            distance_metric=reid_cfg.get("distance_metric", "euclidean")
            if isinstance(reid_cfg, dict) else getattr(reid_cfg, "distance_metric", "euclidean"),
        )
        reid.fit(real_data)
        reid_results = reid.estimate(synth_data)
        reid_figure = reid.plot(reid_results)
        logger.success("Re-identification Risk estimation complete.")

    # ------------------------------------------------------------------ #
    #  MLflow logging                                                      #
    # ------------------------------------------------------------------ #
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run():
        mlflow.log_text(cfg.description, "description.txt")

        # Authenticity
        mlflow.log_figure(auth_figure, "authenticity_results.png")
        mlflow.log_metrics(auth_results.summary()["feature_importance"] or {})
        mlflow.log_metric("auth_mean_binary_score", auth_results.mean_binary_score)
        mlflow.log_metric("auth_mean_ratio_score", auth_results.mean_ratio_score)

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "authenticity_report.json"
            with open(report_path, "w") as f:
                # Convert ndarray values to lists for JSON serialisation
                dump = {
                    k: v.tolist() if hasattr(v, "tolist") else v
                    for k, v in auth_results.model_dump().items()
                }
                json.dump(dump, f, indent=2)
            mlflow.log_artifact(str(report_path))

        # DCR
        if dcr_results is not None:
            mlflow.log_metrics(dcr_results.summary())
            mlflow.log_figure(dcr_figure, "dcr_results.png")
            logger.info(f"DCR summary: {dcr_results.summary()}")

        # Re-identification
        if reid_results is not None:
            mlflow.log_metrics(reid_results.summary())
            mlflow.log_figure(reid_figure, "reidentification_results.png")
            logger.info(f"Re-identification summary: {reid_results.summary()}")

    logger.success("All done, Goodbye!")
>>>>>>> troubleshooting


if __name__ == "__main__":
    main()

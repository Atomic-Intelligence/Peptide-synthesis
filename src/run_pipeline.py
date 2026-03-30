import hydra
import mlflow
import polars as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.utils import DataProcessor
from src.inference.inference_runner import InferenceRunner
from src.logger import setup_logger
from src.mlflow_utils import start_or_connect_mlflow_server
from src.models.Imputation.HistogramImputation import HistogramImputation
from src.models.synthetization_model_interface import (
    SynthetizationModelInterface,
    MlFlowTrainingRunInfo,
    DatasetMetadata,
)

logger = setup_logger()


def initialize_model(
    cfg: DictConfig,
) -> SynthetizationModelInterface:
    logger.info("Initializing model...")
    model = instantiate(cfg.model)
    logger.success(f"Model: {model} instantiated")

    return model


def train_model(
    cfg: DictConfig, model: SynthetizationModelInterface, sampled_patients_num: int = 0
) -> tuple[SynthetizationModelInterface, HistogramImputation, pl.DataFrame]:
    logger.info("Running training...")

    logger.info("Loading real dataset...")
    data_processor_partial = instantiate(cfg.data_processor, _partial_=True)
    df = pl.read_csv(cfg.paths.real_dataset_path)

    data_processor: DataProcessor = data_processor_partial(
        dfs=[df]  # pl.read_csv(cfg.paths.real_dataset_path)]
    )

    if sampled_patients_num > 0:
        data_processor = data_processor.sample_patients_with_preserved_ratios(
            n_patients=sampled_patients_num,
        )

    real_dataset, _ = (
        data_processor.filter_peptides(non_zero_threshold=cfg.non_zero_threshold)
        .split_event_control(event=cfg.event)
        .get_processed_data()
    )[0]

    imputation_data = data_processor.get_data_for_imputation()[0]
    logger.info(f"Imputation data shape: {imputation_data.shape}")

    histogram_imputation_model = instantiate(
        cfg.imputation, column_names=imputation_data.columns
    )

    patient_ids = (
        real_dataset[cfg.primary_key].unique().to_list()
        if cfg.primary_key in real_dataset.columns
        else []
    )
    patient_ids = [str(patient_id) for patient_id in patient_ids]

    real_dataset = real_dataset.drop(cfg.primary_key)

    logger.success(f"Loaded real dataset, shape: {real_dataset.shape}")

    peptide_ids = [col for col in real_dataset.columns if "peptide" in col.lower()]
    dataset_metadata = DatasetMetadata(
        peptide_ids=peptide_ids,
        patient_ids=patient_ids,
        full_dataset_path=cfg.paths.real_dataset_path,
        n_rows=len(patient_ids),
    )
    logger.info("Training model...")
    model.fit(real_dataset, dataset_metadata)
    histogram_imputation_model.fit(imputation_data)
    logger.success("Training completed!")

    return model, histogram_imputation_model, real_dataset


def run_inference(
    cfg: DictConfig,
    ml_flow_info: MlFlowTrainingRunInfo,
    model: SynthetizationModelInterface,
    histogram_imputation_model: HistogramImputation,
) -> pl.DataFrame:
    logger.info("Running inference...")
    inference_runner_partial = instantiate(cfg.inference, _partial_=True)
    inference_runner: InferenceRunner = inference_runner_partial(
        ml_flow_info=ml_flow_info, model=model
    )
    synthetic_data = inference_runner.run(cfg.inference.n_synthetic_patients)
    logger.success(f"Inference completed. Synthetic data shape: {synthetic_data.shape}")

    imputed_column_names, imputed_synthetic_data = histogram_imputation_model.generate(
        cfg.inference.n_synthetic_patients
    )
    if imputed_synthetic_data is not None:
        logger.success(
            f"Histogram imputation completed. Data shape: {imputed_synthetic_data.shape}"
        )
        df_imputed = pl.DataFrame(imputed_synthetic_data, schema=imputed_column_names)
        synthetic_data = pl.concat([synthetic_data, df_imputed], how="horizontal")

    return synthetic_data


def _load_real_data_for_eval(cfg: DictConfig) -> pl.DataFrame:
    """Re-derive the processed real dataset for evaluation when training was skipped."""
    data_processor_partial = instantiate(cfg.data_processor, _partial_=True)
    df = pl.read_csv(cfg.paths.real_dataset_path)
    data_processor = data_processor_partial(dfs=[df])
    real_dataset, _ = (
        data_processor.filter_peptides(non_zero_threshold=cfg.non_zero_threshold)
        .split_event_control(event=cfg.event)
        .get_processed_data()
    )[0]
    return real_dataset.drop(cfg.primary_key)


def run_evaluation(
    cfg: DictConfig,
    real_df: pl.DataFrame,
    synthetic_df: pl.DataFrame,
    model: "SynthetizationModelInterface | None" = None,
) -> None:
    """Run fidelity and privacy evaluation against a synthetic dataset.

    Fidelity metrics (marginal, correlation, joint, two-sample classifier,
    correlation uncertainty) are run via FidelityReport.  Privacy metrics
    (DCR and Authenticity) are run via PrivacyReport, which shares a single
    FeatureProcessor between both estimators to avoid redundant preprocessing.

    Results are logged to MLflow as a nested evaluation run under the training
    run when a model is provided.

    Parameters
    ----------
    cfg :
        Pipeline config.  An optional ``evaluation`` sub-key may carry
        ``dcr_holdout_fraction`` and ``dcr_par_percentile``.
    real_df :
        Processed real dataset (same representation used for training).
    synthetic_df :
        Generated synthetic dataset to evaluate.
    model :
        Fitted model whose MLflow run info is used to attach the evaluation
        run.  When None, metrics are only logged to the console.
    """
    from src.evaluation.fidelity.fidelity_report import FidelityReport
    from src.evaluation.privacy.privacy_report import PrivacyReport

    eval_cfg = getattr(cfg, "evaluation", None)
    dcr_holdout = getattr(eval_cfg, "dcr_holdout_fraction", 0.5) if eval_cfg else 0.5
    dcr_par = getattr(eval_cfg, "dcr_par_percentile", 5.0) if eval_cfg else 5.0

    # Drop string columns (e.g. event_type) — they are label/metadata columns that
    # are constant within each split and break the numerical preprocessing in both
    # fidelity and privacy estimators when the synthetic data is missing them or
    # contains unseen values.
    _string_dtypes = (pl.Utf8, pl.String, pl.Categorical)
    _str_cols = [c for c in real_df.columns if real_df[c].dtype in _string_dtypes]
    if _str_cols:
        logger.info(f"Dropping string columns before evaluation: {_str_cols}")
        real_df = real_df.drop(_str_cols)
        synthetic_df = synthetic_df.drop([c for c in _str_cols if c in synthetic_df.columns])

    logger.info("Running fidelity evaluation...")
    fidelity_report = FidelityReport()
    fidelity_results = fidelity_report.run(real_df, synthetic_df)
    logger.success("Fidelity evaluation complete.")
    for k, v in fidelity_results.summary().items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("Running privacy evaluation...")
    privacy_report = PrivacyReport(
        holdout_fraction=dcr_holdout,
        par_percentile=dcr_par,
    )
    privacy_results = privacy_report.run(real_df, synthetic_df)
    logger.success("Privacy evaluation complete.")
    for k, v in privacy_results.summary().items():
        logger.info(f"  {k}: {v}")

    if model is not None:
        try:
            experiment_id, training_run_id = model.get_or_create_run(model.ml_flow_info)
            with mlflow.start_run(
                experiment_id=experiment_id, run_id=training_run_id
            ):
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name="evaluation",
                    nested=True,
                ):
                    fidelity_metrics = fidelity_results.summary()
                    if fidelity_metrics:
                        mlflow.log_metrics(fidelity_metrics)
                    privacy_metrics = {
                        k: v
                        for k, v in privacy_results.summary().items()
                        if isinstance(v, (int, float))
                    }
                    if privacy_metrics:
                        mlflow.log_metrics(privacy_metrics)
            logger.success("Evaluation metrics logged to MLflow.")
        except Exception as exc:
            logger.error(f"Failed to log evaluation metrics to MLflow: {exc}")


@hydra.main(
    version_base="1.1",
    config_path="../configs/training_and_inference_pipeline",
    config_name="pipeline.yaml",
)
def main(cfg: DictConfig):
    shutdown_hook = start_or_connect_mlflow_server(cfg.ml_flow_tracking_uri)

    model = initialize_model(cfg)

    real_df = None
    histogram_imputation_model = None
    if cfg.run_training:
        model, histogram_imputation_model, real_df = train_model(
            cfg=cfg, model=model, sampled_patients_num=cfg.sampled_patients_num
        )

    synthetic_data = None
    if cfg.run_inference:
        synthetic_data = run_inference(
            cfg=cfg,
            ml_flow_info=model.ml_flow_info,
            model=model,
            histogram_imputation_model=histogram_imputation_model,
        )
        synthetic_data.write_csv(
            f"synthetic_{cfg.event}_sampled_{cfg.sampled_patients_num}.csv"
        )

    if cfg.run_evaluation:
        if real_df is None:
            real_df = _load_real_data_for_eval(cfg)
        if synthetic_data is None:
            raise ValueError(
                "run_evaluation requires synthetic data. Enable run_inference or "
                "set run_inference: true before run_evaluation: true."
            )
        run_evaluation(cfg=cfg, real_df=real_df, synthetic_df=synthetic_data, model=model)

    input("Press Enter to shut down the experiment viewing app...")
    shutdown_hook()
    logger.info("Shutting down the experiment viewing app...")
    logger.info("Done!")


if __name__ == "__main__":
    main()

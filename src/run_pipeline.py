import subprocess
import hydra
import mlflow
import polars as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.utils import DataProcessor
from src.inference.inference_runner import InferenceRunner
from src.logger import setup_logger
from src.mlflow import start_or_connect_mlflow_server
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
) -> SynthetizationModelInterface:
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
    logger.success("Training completed!")

    return model


def run_inference(
    cfg: DictConfig,
    ml_flow_info: MlFlowTrainingRunInfo,
    model: SynthetizationModelInterface,
) -> pl.DataFrame:
    logger.info("Running inference...")
    inference_runner_partial = instantiate(cfg.inference, _partial_=True)
    inference_runner: InferenceRunner = inference_runner_partial(
        ml_flow_info=ml_flow_info, model=model
    )
    synthetic_data = inference_runner.run(cfg.inference.n_synthetic_patients)
    logger.success(f"Inference completed. Synthetic data shape: {synthetic_data.shape}")

    return synthetic_data


@hydra.main(
    version_base="1.1",
    config_path="../configs/training_and_inference_pipeline",
    config_name="pipeline.yaml",
)
def main(cfg: DictConfig):
    shutdown_hook = start_or_connect_mlflow_server(cfg.ml_flow_tracking_uri)

    model = initialize_model(cfg)

    if cfg.run_training:
        train_model(cfg=cfg, model=model, sampled_patients_num=cfg.sampled_patients_num)

    if cfg.run_inference:
        synthetic_data = run_inference(
            cfg=cfg, ml_flow_info=model.ml_flow_info, model=model
        )
        synthetic_data.write_csv(
            f"synthetic_{cfg.event}_sampled_{cfg.sampled_patients_num}.csv"
        )
    
    input("Press Enter to shut down the experiment viewing app...")
    shutdown_hook()
    logger.info("Shutting down the experiment viewing app...")
    logger.info("Done!")


        

if __name__ == "__main__":
    main()

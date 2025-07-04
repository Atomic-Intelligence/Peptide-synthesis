import hydra
import mlflow
import polars as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from v0.src import DataProcessor
from v0.src import InferenceRunner
from v0.src import setup_logger
from v0.src import (
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

    def subsample_by_event(
        df: pl.DataFrame, event_counts: dict, event_col: str
    ) -> pl.DataFrame:
        """
        Subsamples a Polars DataFrame to keep a specified number of rows for each event type.

        Args:
            df: The input Polars DataFrame.
            event_counts: A dictionary where keys are event types and values are the
                        desired number of samples for each event.
            event_col: The name of the column containing the event type.

        Returns:
            A new Polars DataFrame with the subsampled data.
        """
        subsampled_dfs = []
        for event, count in event_counts.items():
            subset = df.filter(pl.col(event_col) == event).head(count)
            subsampled_dfs.append(subset)

        if not subsampled_dfs:
            return pl.DataFrame()  # Return an empty DataFrame if event_counts is empty

        return pl.concat(subsampled_dfs)

    # event_count = {"no_event": 254, "hf": 37, "ckd": 9}

    # df = subsample_by_event(df, event_counts=event_count, event_col="event_type")

    data_processor: DataProcessor = data_processor_partial(
        dfs=[df]  # pl.read_csv(cfg.paths.real_dataset_path)]
    )

    if sampled_patients_num > 0:
        data_processor = data_processor.sample_patients_with_preserved_ratios(
            n_patients=sampled_patients_num,
        )

    real_dataset, _ = (
        data_processor.filter_peptides(non_zero_threshold=30.0)
        .split_event_control(event=cfg.event)
        .get_processed_data()
    )[0]

    patient_ids = (
        real_dataset["idAuswertung"].unique().to_list()
        if "idAuswertung" in real_dataset.columns
        else []
    )
    patient_ids = [str(patient_id) for patient_id in patient_ids]
    print(real_dataset)

    real_dataset = real_dataset.drop("idAuswertung")

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
    logger.success(f"Training completed!")

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
    mlflow.set_tracking_uri("http://10.100.111.210:5002")

    model = initialize_model(cfg)

    if cfg.run_training:
        train_model(cfg=cfg, model=model, sampled_patients_num=cfg.sampled_patients_num)

    if cfg.run_inference:
        synthetic_data = run_inference(
            cfg=cfg, ml_flow_info=model.ml_flow_info, model=model
        )
        synthetic_data.write_csv(
            f"student_{cfg.event}_sampled_{cfg.sampled_patients_num}.csv"
        )


if __name__ == "__main__":
    main()

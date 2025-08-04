import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import hydra
import mlflow
import polars as pl
import numpy as np
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from src.mlflow import start_or_connect_mlflow_server

# Import all your existing modules
from src.data.PeptideDataset import (
    CATEGORICAL_CLINICAL_COLUMNS,
    NUMERICAL_CLINICAL_COLUMNS,
)
from src.evaluation.utils.eval_utils import get_event_and_control, get_peptide_columns
from src.evaluation.analysis.multi_peptide_analysis import survival_analysis
from src.evaluation.analysis.single_peptide_analysis import (
    eGFR_CKD_score_analysis,
    compare_eGFR,
    peptide_eGFR_analysis,
    mann_whitney_analysis,
    plot_single_peptide_distributions,
)
from src.evaluation.projections.pca_projections import (
    clinical_variable_pca,
    make_peptide_pca,
)
from src.evaluation.projections.umap_projections import make_peptide_umap
from src.evaluation.classifiers.machine_learning_efficiency import (
    train_on_real_estimate_on_synthetic,
    train_on_synthetic_test_on_real,
)

logging.basicConfig(level=logging.INFO)
import warnings

warnings.filterwarnings("ignore")


# Control CPU usage for libraries that use OpenMP or other parallelism
def limit_cpu_usage(max_threads):
    """Set environment variables to limit CPU usage in various libraries."""
    # Limit OpenMP threads (used by many scientific libraries)
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    # Limit OpenBLAS threads
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    # Limit MKL threads (used by Intel's Math Kernel Library)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    # Limit NUMEXPR threads
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
    # Limit VECLIB threads (Apple's vector library)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    # Limit TBB threads (Intel's Threading Building Blocks)
    os.environ["TBB_NUM_THREADS"] = str(max_threads)

    logger.info(f"CPU usage limited to {max_threads} threads per process")


def balance_event_types(
    df1: pl.DataFrame, df2: pl.DataFrame, column: str = "event_type"
) -> pl.DataFrame:
    # Calculate distribution from df1
    df1_dist = (
        df1.group_by(column)
        .agg(pl.count().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
    )

    # Get total number of rows we want in df2 (same as df1)
    target_total = df1.height

    # Calculate target counts for each category based on df1 proportions
    target_counts = df1_dist.select(
        [
            column,
            (pl.col("proportion") * target_total).cast(pl.Int64).alias("target_count"),
        ]
    )

    # Get current counts in df2
    df2_counts = df2.group_by(column).agg(pl.count().alias("current_count"))

    # Join target and current counts
    counts_comparison = target_counts.join(df2_counts, on=column, how="left")

    # For each category, sample the required number of rows
    sampled_dfs = []
    for row in counts_comparison.rows(named=True):
        category = row[column]
        target = row["target_count"]
        current = row["current_count"] or 0

        # Filter rows for this category
        category_df = df2.filter(pl.col(column) == category)

        if target == 0:
            continue
        elif current <= target:
            # If we have fewer than needed, take all and duplicate some
            sampled = category_df.sample(n=target, with_replacement=True)
        else:
            # If we have more than needed, downsample
            sampled = category_df.sample(n=target, with_replacement=False)

        sampled_dfs.append(sampled)

    # Combine all sampled dataframes
    result_df = pl.concat(sampled_dfs)

    return result_df


def prepare_event_data(real_dataset, synthetic_dataset, event_types):
    """Prepare event and control data for analysis based on a list of event types."""
    peptide_columns = get_peptide_columns(synthetic_dataset)
    real_peptides = real_dataset.select(peptide_columns + ["event_type"])
    synthetic_peptides = synthetic_dataset.select(peptide_columns + ["event_type"])

    event_data = {}
    for event in event_types:
        # Get event and control groups for real and synthetic data
        real_event, real_control = get_event_and_control(
            real_peptides, event=event.name
        )
        synthetic_event, synthetic_control = get_event_and_control(
            synthetic_peptides, event=event.name
        )

        # Extract time-to-event for this event type, including 'no_event' as control
        time_to_event_real = (
            real_dataset.filter(
                (pl.col("event_type") == event.name)
                | (pl.col("event_type") == "no_event")
            )
            .select(event.time_to_event_column)
            .to_numpy()
        )

        time_to_event_synthetic = (
            synthetic_dataset.filter(
                (pl.col("event_type") == event.name)
                | (pl.col("event_type") == "no_event")
            )
            .select(event.time_to_event_column)
            .cast(pl.Float32)
            .fill_null(0.0)
            .to_numpy()
        )
        logger.info(f"Time to event {event.display_name} {time_to_event_synthetic}")
        # Store data in a dictionary keyed by event name
        event_data[event.name] = {
            "real_event": real_event.to_numpy(),
            "real_control": real_control.to_numpy(),
            "synthetic_event": synthetic_event.to_numpy(),
            "synthetic_control": synthetic_control.to_numpy(),
            "time_to_event_real": time_to_event_real,
            "time_to_event_synthetic": time_to_event_synthetic,
        }

    # Add full dataset (independent of event types)
    event_data["full"] = {"real": real_peptides, "synthetic": synthetic_peptides}
    return event_data


# Define a worker initialization function to set thread limits for each worker
def init_worker(max_threads_per_worker):
    """Initialize each worker process with thread limits."""
    # Each worker process gets its own thread limit
    limit_cpu_usage(max_threads_per_worker)


def execute_tasks(executor, tasks):
    """Execute tasks using the provided executor and return results."""
    results = {}
    future_to_task = {executor.submit(task): name for name, task in tasks.items()}
    for future in as_completed(future_to_task):
        task_name = future_to_task[future]
        try:
            result = future.result()
            results[task_name] = result
            logger.info(f"Task '{task_name}' completed successfully")
        except Exception as exc:
            logger.error(f"Task '{task_name}' generated an exception: {exc}")

    return results


def run_evaluation_pipeline(
    cfg: DictConfig, real_dataset, synthetic_dataset, classifier_models, executor
):
    """Run all evaluation tasks in a streamlined pipeline for all event types using a single process pool."""
    logger.info("Running evaluation pipeline...")

    run = mlflow.start_run()
    run_id = run.info.run_id

    # Prepare event data for all event types
    event_data = prepare_event_data(real_dataset, synthetic_dataset, cfg.event_types)
    logger.info("EVENT DATA: " + " ".join(list(event_data.keys())))
    logger.success("Event data prepared!")

    # Define independent tasks
    tasks = {}
    for event in cfg.event_types:
        # Classifier tasks: real data training
        tasks[f"classifier_{event.name}_real"] = partial(
            train_on_real_estimate_on_synthetic,
            classifier=classifier_models[f"real_{event.name}_classifier"],
            event_type=event.name,
            real_df=event_data["full"]["real"],
            synthetic_df=event_data["full"]["synthetic"],
        )
        # Classifier tasks: synthetic data training
        tasks[f"classifier_{event.name}_synthetic"] = partial(
            train_on_synthetic_test_on_real,
            classifier=classifier_models[f"synthetic_{event.name}_classifier"],
            event_type=event.name,
            real_df=event_data["full"]["real"],
            synthetic_df=event_data["full"]["synthetic"],
        )
        # Mann-Whitney analysis
        tasks[f"mann_whitney_{event.name}"] = partial(
            mann_whitney_analysis,
            run_id=run_id,
            real_event=event_data[event.name]["real_event"],
            real_no_event=event_data[event.name]["real_control"],
            synthetic_event=event_data[event.name]["synthetic_event"],
            synthetic_no_event=event_data[event.name]["synthetic_control"],
            event_type=event.display_name,
            threshold_p=0.05,
            adjust_pvalue=True,
        )

    tasks["marginal_dist_comp"] = partial(
        plot_single_peptide_distributions,
        run_id=run_id,
        synthetic_dataset=event_data["full"]["synthetic"],
        real_dataset=event_data["full"]["real"],
    )

    # Visualization tasks (independent of event types)
    tasks["umap"] = partial(
        make_peptide_umap,
        run_id,
        event_data["full"]["real"],
        event_data["full"]["synthetic"],
    )
    tasks["pca"] = partial(
        make_peptide_pca,
        run_id,
        event_data["full"]["real"],
        event_data["full"]["synthetic"],
    )
    tasks["clinical_pca"] = partial(
        clinical_variable_pca,
        run_id=run_id,
        real_dataset=real_dataset,
        synthetic_dataset=synthetic_dataset,
        clinical_columns=CATEGORICAL_CLINICAL_COLUMNS,
        numerical_columns=NUMERICAL_CLINICAL_COLUMNS,
    )
    tasks["egfr_comparison"] = partial(
        compare_eGFR,
        run_id=run_id,
        synthetic_dataset=synthetic_dataset,
        real_dataset=real_dataset,
    )
    tasks["peptide_egfr"] = partial(
        peptide_eGFR_analysis,
        run_id=run_id,
        synthetic_dataset=synthetic_dataset,
        real_dataset=real_dataset,
    )

    # Execute all independent tasks using the shared executor
    reports = execute_tasks(executor, tasks)

    # Filter to keep only classifier reports
    classifier_reports = {k: v for k, v in reports.items() if "classifier" in k}

    # Define survival analysis tasks (dependent on classifier results)
    survival_tasks = {}
    for event in cfg.event_types:
        for dataset_type in ["real", "synthetic"]:
            report = classifier_reports.get(f"classifier_{event.name}_{dataset_type}")
            if report:
                survival_tasks[f"survival_{event.name}_{dataset_type}"] = partial(
                    survival_analysis,
                    run_id=run_id,
                    event_type=event.display_name,
                    time_to_event_array=event_data[event.name][
                        f"time_to_event_{dataset_type}"
                    ],
                    survival_score_array=report[dataset_type]["svm_score"],
                    num_quantiles=5,
                    dataset_type=dataset_type,
                )

    # Only add these tasks if the classifier reports exist
    if "classifier_ckd_real" in classifier_reports:
        survival_tasks[f"egfr_ckd_score_real"] = partial(
            eGFR_CKD_score_analysis,
            run_id=run_id,
            dataset_type="real",
            egfr_array=real_dataset.filter(
                (pl.col("event_type") == "ckd") | (pl.col("event_type") == "no_event")
            )
            .select("GFR_CKD_EPI_M")
            .to_numpy(),
            svm_ckd_score=classifier_reports[f"classifier_ckd_real"]["real"][
                "svm_score"
            ],
        )
        survival_tasks[f"egfr_ckd_score_synthetic"] = partial(
            eGFR_CKD_score_analysis,
            run_id=run_id,
            dataset_type="synthetic",
            egfr_array=synthetic_dataset.filter(
                (pl.col("event_type") == "ckd") | (pl.col("event_type") == "no_event")
            )
            .select("GFR_CKD_EPI_M")
            .to_numpy(),
            svm_ckd_score=classifier_reports[f"classifier_ckd_real"]["real"][
                "svm_score"
            ],
        )

    # Execute survival tasks using the same executor
    execute_tasks(executor, survival_tasks)

    # Log metrics for each event type
    for event in cfg.event_types:
        synth_report = classifier_reports.get(f"classifier_{event.name}_synthetic")
        if synth_report:
            mlflow.log_metric(
                f"{event.name}_synthetic_svm_f1_on_synthetic",
                synth_report["synthetic"]["f1"],
            )
            mlflow.log_metric(
                f"{event.name}_synthetic_svm_f1_on_real",
                synth_report["real"]["f1"],
            )

        real_report = classifier_reports.get(f"classifier_{event.name}_real")
        if real_report:
            mlflow.log_metric(
                f"{event.name}_real_svm_f1_on_synthetic",
                real_report["synthetic"]["f1"],
            )
            mlflow.log_metric(
                f"{event.name}_real_svm_f1_on_real",
                real_report["real"]["f1"],
            )

    logger.success("Evaluation pipeline completed!")
    return classifier_reports


def run_evaluation(cfg: DictConfig):
    """Optimized evaluation process with a single process pool and controlled CPU usage."""
    multiprocessing.set_start_method("spawn", force=True)
    num_cpus = multiprocessing.cpu_count()
    max_workers = min(num_cpus, cfg.num_threads)

    # Calculate threads per worker
    # For example, if you have 32 cores and 8 workers, each worker gets 1 thread
    # We use max(1, ...) to ensure at least 1 thread per worker
    threads_per_worker = max(1, int(num_cpus / max_workers / 2))

    logger.info(f"Running with {max_workers} workers (from {num_cpus} available CPUs)")
    logger.info(f"Each worker will use up to {threads_per_worker} threads")

    # Set thread limits for the main process
    limit_cpu_usage(threads_per_worker)

    # Load and prepare data
    logger.info("Loading data...")
    real_dataset = pl.read_csv(cfg.paths.real_data_path).head(100)
    synthetic_dataset = pl.read_csv(cfg.paths.synthetic_data_path).head(100)

    # synthetic_dataset = balance_event_types(real_dataset, synthetic_dataset)

    logger.info(f"Working with {len(real_dataset)} real patients")
    logger.info(f"Working with {len(synthetic_dataset)} synthetic patients")

    real_distribution = real_dataset.group_by("event_type").agg(
        pl.count().alias("count")
    )
    synthetic_distribution = synthetic_dataset.group_by("event_type").agg(
        pl.count().alias("count")
    )

    print(real_distribution)
    print(synthetic_distribution)

    logger.success("Loaded data!")

    columns = list(
        set(synthetic_dataset.columns).intersection(set(real_dataset.columns))
    )

    real_dataset = real_dataset.select(columns)
    real_dataset = real_dataset.fill_null(0.0)
    synthetic_dataset = synthetic_dataset.fill_null(0.0)

    # Initialize classifier models
    classifier_models = instantiate(cfg.classifier_models)

    # Create a single process pool that will be used for all parallel tasks
    # Use the initializer to set thread limits for each worker
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=init_worker, initargs=(threads_per_worker,)
    ) as executor:
        # Run the unified pipeline with the shared executor
        run_evaluation_pipeline(
            cfg, real_dataset, synthetic_dataset, classifier_models, executor
        )


@hydra.main(
    version_base="1.1",
    config_path="../../configs/evaluation",
    config_name="data_eval_config.yaml",
)
def main(cfg: DictConfig) -> None:
    shutdown_hook = start_or_connect_mlflow_server(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()

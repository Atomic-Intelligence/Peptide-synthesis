import numpy as np
import polars as pl
import mlflow
from loguru import logger
import umap
from sklearn.preprocessing import StandardScaler
from v0.src import safe_start_run
from v0.src import plot_reduced_data


def make_peptide_umap(
    run_id: str,
    real_dataset: pl.DataFrame,
    synthetic_dataset: pl.DataFrame,
    use_scaler: bool = False,
):
    logger.info("Starting UMAP analysis")

    real_dataset = real_dataset.drop("event_type").to_numpy()
    synthetic_dataset = synthetic_dataset.drop("event_type").to_numpy()

    n_real, n_synth = real_dataset.shape[0], synthetic_dataset.shape[0]
    all_peptides = np.concatenate([real_dataset, synthetic_dataset], axis=0)

    if use_scaler:
        scaler = StandardScaler()
        all_peptides = scaler.fit_transform(all_peptides)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(all_peptides)

    umap_fig = plot_reduced_data(
        embedding,
        num_real=n_real,
        title="Peptidomics UMAP",
        reduction_type="UMAP",
        real_first=True,
    )
    with safe_start_run(run_id=run_id):
        mlflow.log_figure(umap_fig, "projections/Peptidomics_UMAP.png")

    logger.success("Finished UMAP analysis")
    return

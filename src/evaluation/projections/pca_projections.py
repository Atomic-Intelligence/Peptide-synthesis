import mlflow
import numpy as np
import polars as pl
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.evaluation.mlflow_threadsafe import safe_start_run, safe_log_figure
from src.evaluation.utils.plotting import plot_reduced_data


def make_peptide_pca(
    run_id: str,
    real_dataset: pl.DataFrame,
    synthetic_dataset: pl.DataFrame,
    use_scaler: bool = True,
):
    logger.info("Starting PCA analysis")
    real_peptides = real_dataset.drop("event_type").to_numpy()
    synthetic_peptides = synthetic_dataset.drop("event_type").to_numpy()

    n_real, n_synth = real_peptides.shape[0], synthetic_peptides.shape[0]
    all_peptides = np.concatenate([real_peptides, synthetic_peptides], axis=0)

    if use_scaler:
        scaler = StandardScaler()
        all_peptides = scaler.fit_transform(all_peptides)
    reducer = PCA(n_components=2)
    embedding = reducer.fit_transform(all_peptides)

    umap_fig = plot_reduced_data(
        embedding,
        num_real=n_real,
        title="Peptidomics PCA",
        reduction_type="PCA",
        real_first=True,
    )
    with safe_start_run(run_id=run_id):
        mlflow.log_figure(umap_fig, "projections/Peptidomics_PCA.png")

    logger.success("Finished PCA analysis")
    return


def clinical_variable_pca(
    run_id: str,
    real_dataset: pl.DataFrame,
    synthetic_dataset: pl.DataFrame,
    clinical_columns: list[str],
    numerical_columns: list[str],
    use_scaler: bool = True,
):
    logger.info("Starting PCA analysis for clinical variables")

    # Process categorical columns with one-hot encoding
    # Get categorical data from both datasets
    real_categorical = real_dataset.select(clinical_columns)
    synthetic_categorical = synthetic_dataset.select(clinical_columns)

    # Convert to pandas for one-hot encoding
    real_categorical_pd = real_categorical.to_pandas()
    synthetic_categorical_pd = synthetic_categorical.to_pandas()

    # Apply one-hot encoding
    encoder = OneHotEncoder(
        sparse_output=False, drop="if_binary", handle_unknown="ignore"
    )
    real_categorical_encoded = encoder.fit_transform(real_categorical_pd)
    synthetic_categorical_encoded = encoder.transform(synthetic_categorical_pd)

    # Process numerical columns
    real_numerical = real_dataset.select(numerical_columns).to_numpy()
    synthetic_numerical = synthetic_dataset.select(numerical_columns).to_numpy()

    # Scale numerical data if requested
    if use_scaler:
        scaler = StandardScaler()
        real_numerical = scaler.fit_transform(real_numerical)
        synthetic_numerical = scaler.transform(synthetic_numerical)

    # Combine categorical and numerical features
    real_combined = np.hstack([real_categorical_encoded, real_numerical])
    synthetic_combined = np.hstack([synthetic_categorical_encoded, synthetic_numerical])

    # Get dimensions
    n_real, n_synth = real_combined.shape[0], synthetic_combined.shape[0]

    # Combine real and synthetic data for PCA
    all_data = np.vstack([real_combined, synthetic_combined])

    # Apply PCA
    reducer = PCA(n_components=2)
    embedding = reducer.fit_transform(all_data)

    # Plot and log results
    pca_fig = plot_reduced_data(
        embedding,
        num_real=n_real,
        title="Clinical and Numerical PCA",
        reduction_type="PCA",
        real_first=True,
    )

    with safe_start_run(run_id=run_id):
        mlflow.log_figure(pca_fig, "projections/Clinical_Numerical_PCA.png")

    logger.success("Finished PCA analysis")
    return

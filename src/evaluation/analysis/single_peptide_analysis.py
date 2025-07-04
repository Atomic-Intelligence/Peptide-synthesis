import logging
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger
from scipy.stats import mannwhitneyu, false_discovery_control

from src.evaluation.mlflow_threadsafe import safe_start_run
from src.evaluation.utils.eval_utils import (
    peptide_eGFR_correlation,
    get_peptide_columns,
)
from src.evaluation.utils.plotting import (
    real_synth_scatterplot,
    significant_peptide_venn,
)

logging.basicConfig(level=logging.INFO)


def plot_single_peptide_distributions(
    run_id: str,
    real_dataset: pl.DataFrame,
    synthetic_dataset: pl.DataFrame,
    num_peptides_to_plot: int = 20,
):
    peptide_cols = get_peptide_columns(synthetic_dataset)
    cols_to_plot = np.random.choice(
        peptide_cols, replace=False, size=num_peptides_to_plot
    ).tolist()
    figures = []
    for col in cols_to_plot:
        figure, ax = plt.subplots()
        ax.set_title(f"{col} marginal distributions", fontsize=18)
        real_array = real_dataset.select(col).to_numpy()
        ax.hist(
            real_array,
            weights=np.zeros_like(real_array) + 1.0 / len(real_array),
            color="tab:blue",
            ec="k",
            # alpha=0.7,
            label="Real",
            bins=50,
            density=False,
        )
        synth_array = (synthetic_dataset.select(col).to_numpy(),)
        ax.hist(
            synth_array,
            weights=np.zeros_like(synth_array) + 1.0 / len(synth_array),
            color="tab:orange",
            ec="k",
            alpha=0.7,
            label="Synth",
            bins=50,
            density=False,
        )
        plt.legend(fontsize=16)
        figures.append((col, figure))

    with safe_start_run(run_id=run_id):
        for col_name, figure in figures:
            mlflow.log_figure(
                figure, f"marginal_distribution_comparison/{col_name}.png"
            )
    return None


def eGFR_CKD_score_analysis(
    run_id: str, egfr_array: np.ndarray, svm_ckd_score: np.ndarray, dataset_type: str
) -> None:

    logger.info((f"Plotting eGFR to SVM CKD 273 score for {dataset_type} dataset"))
    figure, ax = plt.subplots()

    ax.set_title(f"Correlation analysis - {dataset_type} data", fontsize=19)

    sns.regplot(
        data=pl.DataFrame(data={"eGFR": egfr_array, "CKD273 score": svm_ckd_score}),
        x="eGFR",
        y="CKD273 score",
        ax=ax,
    )
    logger.success(
        f"Created correlation plot for eGFR-CKD273 score for {dataset_type} dataset."
    )
    with safe_start_run(run_id=run_id):
        mlflow.log_figure(
            figure, f"ckd_score_analysis/eGFR_CKD_score_{dataset_type}.png"
        )
    return None


def mann_whitney_analysis(
    run_id: str,
    real_event: np.ndarray,
    real_no_event: np.ndarray,
    synthetic_event: np.ndarray,
    synthetic_no_event: np.ndarray,
    event_type: str,
    threshold_p: float = 0.05,
    adjust_pvalue: bool = True,
):
    logger.info(f"Starting mann-whitney analysis for {event_type}")

    _, real_p_value = mannwhitneyu(real_event, real_no_event)
    _, synth_p_value = mannwhitneyu(synthetic_event, synthetic_no_event)
    if adjust_pvalue:
        real_p_value = false_discovery_control(real_p_value)
        synth_p_value = false_discovery_control(synth_p_value)

    real_significant = real_p_value < threshold_p
    synth_significant = synth_p_value < threshold_p
    venn_diagram_fig = significant_peptide_venn(
        real_significant=real_significant, synthetic_significant=synth_significant
    )

    common_significant = np.logical_and(real_significant, synth_significant)

    # Filter out non-significant peptides
    real_event = real_event[:, common_significant]
    real_no_event = real_no_event[:, common_significant]
    synthetic_event = synthetic_event[:, common_significant]
    synthetic_no_event = synthetic_no_event[:, common_significant]

    real_event_means = np.mean(real_event, axis=0)
    real_no_event_means = np.mean(real_no_event, axis=0)
    synthetic_event_means = np.mean(synthetic_event, axis=0)
    synthetic_no_event_means = np.mean(synthetic_no_event, axis=0)

    data_arrays = [
        real_event_means,
        real_no_event_means,
        synthetic_event_means,
        synthetic_no_event_means,
    ]
    df = pd.DataFrame()

    column_names = [
        f"Mean {event_type}\nReal training data",
        f"Mean {event_type}\nSynthetic data",
        "Mean NE\nReal training data",
        "Mean NE\nSynthetic data",
    ]

    data_arrays = [
        real_event_means,
        synthetic_event_means,
        real_no_event_means,
        synthetic_no_event_means,
    ]

    df = pd.DataFrame({col: data for col, data in zip(column_names, data_arrays)})
    g = sns.clustermap(
        df,
        cmap="coolwarm",
        center=0,
        col_cluster=False,
        cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1]},
        yticklabels=False,
        xticklabels=True,
        z_score=0,
        dendrogram_ratio=(0.1, 0.2),
    )
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0, ha="center")
    plt.subplots_adjust(left=0.1)
    with safe_start_run(run_id=run_id):
        mlflow.log_figure(
            venn_diagram_fig,
            f"significant_peptides/{event_type}_significant_peptides.png",
        )
        mlflow.log_figure(g.figure, f"heatmaps/peptide_mean_data_{event_type}.png")
    logger.success(f"Finished mann-whitney analysis for {event_type}")
    return None


def compare_eGFR(
    run_id: str,
    real_dataset: pl.DataFrame,
    synthetic_dataset: pl.DataFrame,
    eGFR_col: str = "GFR_CKD_EPI_M",
) -> None:
    logger.info("Starting EGFR comparison")

    real_eGFR = real_dataset.select(eGFR_col).with_columns(
        pl.lit("real").alias("source")
    )
    synth_eGFR = synthetic_dataset.select(eGFR_col).with_columns(
        pl.lit("synth").alias("source")
    )

    fig, ax = plt.subplots()
    ax.set_title("eGFR comparison", fontsize=18)
    # ax.set_xlabel(r"eGFR", fontsize=16)

    sns.histplot(
        pl.concat([real_eGFR, synth_eGFR]),
        kde=True,
        x=eGFR_col,
        hue="source",
        palette=["blue", "orange"],
        ax=ax,
        stat="density",
    )

    with safe_start_run(run_id=run_id):
        mlflow.log_figure(fig, "correlation_analysis/eGFR_comparison.png")

    logger.success("Finished EGFR comparison")
    return None


def peptide_eGFR_analysis(
    run_id: str, real_dataset: pl.DataFrame, synthetic_dataset: pl.DataFrame
) -> None:
    logger.info("Starting peptide eGFR analysis")

    peptide_cols = get_peptide_columns(synthetic_dataset)
    peptide_cols.remove("event_type")

    real_dataset = real_dataset.select(peptide_cols + ["GFR_CKD_EPI_M"])
    synthetic_dataset = synthetic_dataset.select(peptide_cols + ["GFR_CKD_EPI_M"])

    real_eGFR_peptide_corr, real_eGFR_peptide_pvalue = peptide_eGFR_correlation(
        real_dataset, adjust_p_value=False
    )

    synthetic_eGFR_peptide_corr, synthetic_eGFR_peptide_pvalue = (
        peptide_eGFR_correlation(synthetic_dataset, adjust_p_value=False)
    )

    correlation_df = pl.DataFrame(
        {
            "peptide": peptide_cols,
            "real_correlation": real_eGFR_peptide_corr,
            "real_pvalue": real_eGFR_peptide_pvalue,
            "synthetic_correlation": synthetic_eGFR_peptide_corr,
            "synthetic_pvalue": synthetic_eGFR_peptide_pvalue,
        }
    )

    temp_corr_csv = "peptide_eGFR_correlation.csv"

    correlation_df.write_csv(temp_corr_csv)

    corr_fig = real_synth_scatterplot(
        real=correlation_df.select("real_correlation").to_numpy(),
        synth=correlation_df.select("synthetic_correlation").to_numpy(),
    )
    with safe_start_run(run_id=run_id):
        mlflow.log_artifact(temp_corr_csv)
        mlflow.log_figure(
            corr_fig, "correlation_analysis/peptide_eGFR_cor_cor_plot.png"
        )

    if os.path.exists(temp_corr_csv):
        os.remove(temp_corr_csv)

    logger.success("Finished peptide eGFR analysis")
    return None

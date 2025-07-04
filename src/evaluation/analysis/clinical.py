import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl

from v0.src import NUMERICAL_CLINICAL_COLUMNS
from v0.src import safe_start_run


def clinical_marginal_distributions(
    run_id: str,
    real_dataset: pl.DataFrame,
    synthetic_dataset: pl.DataFrame,
):
    with safe_start_run(run_id=run_id):
        for col in NUMERICAL_CLINICAL_COLUMNS:
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
            mlflow.log_figure(figure, f"marginal_distribution_comparison/{col}.png")
    return None

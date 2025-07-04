import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib_venn import venn2, venn2_circles


def plot_kaplan_meier(
    survival_arrays: list[np.ndarray] | np.ndarray, time_steps: np.ndarray
) -> plt.Figure:
    fig, ax = plt.subplots()
    if isinstance(survival_arrays, np.ndarray):
        survival_arrays = [survival_arrays]

    for i, survival in enumerate(survival_arrays):
        if len(survival_arrays) > 1:
            label = f"Quantile: {i + 1}"
        else:
            label = None
        ax.plot(time_steps / 365, survival, label=label)
    ax.legend(fontsize=15)
    ax.set_title("Survival percentage", fontsize=18)
    ax.set_xlabel("Time [yrs.]", fontsize=16)
    ax.set_ylabel("Survival percentage", fontsize=16)
    return fig


def significant_peptide_venn(
    real_significant: np.ndarray, synthetic_significant: np.ndarray
) -> plt.Figure:
    only_real = np.logical_and(
        real_significant, np.logical_not(synthetic_significant)
    ).astype(float)
    only_synth = np.logical_and(
        np.logical_not(real_significant), synthetic_significant
    ).astype(float)
    overlap = np.logical_and(real_significant, synthetic_significant).astype(float)
    fig, ax = plt.subplots()
    with mpl.rc_context({"font.size": 16}):
        venn2(
            subsets=(only_real.sum(), only_synth.sum(), overlap.sum()),
            set_labels=("Real", "Synthetic"),
            ax=ax,
        )
        venn2_circles(
            subsets=(only_real.sum(), only_synth.sum(), overlap.sum()),
            ax=ax,
            linewidth=2,
        )

    return fig


def real_synth_scatterplot(real: np.ndarray, synth: np.ndarray) -> plt.Figure:
    corr_fig, corr_ax = plt.subplots()
    corr_ax.scatter(
        real,
        synth,
        color="tab:orange",
    )

    min_val = min(
        real.min(),
        synth.min(),
    )
    max_val = max(
        real.max(),
        synth.max(),
    )
    plt.plot([min_val, max_val], [min_val, max_val], color="tab:blue", alpha=0.5)

    plt.xlabel("Real Peptide Correlation")
    plt.ylabel("Synthetic Peptide Correlation")
    plt.title("Comparison of Real vs Synthetic Peptide Correlations with eGFR")
    plt.grid(True, alpha=0.3)
    return corr_fig


def plot_reduced_data(
    embedding: np.ndarray,
    num_real: int,
    title: str = None,
    reduction_type: str = "PCA",
    real_first: bool = True,
) -> plt.Figure:

    fig, ax = plt.subplots()
    num_synth = embedding.shape[0] - num_real
    assert (
        num_synth > 0
    ), f"Number of real data {num_real} must be strictly less than the number of total data {embedding.shape[0]}"
    real_umap = embedding[:num_real, :] if real_first else embedding[num_synth:, :]
    synthetic_umap = embedding[num_real:, :] if real_first else embedding[:num_synth, :]
    ax.scatter(
        real_umap[:, 0], real_umap[:, 1], color="tab:blue", alpha=0.7, label="Real"
    )
    ax.scatter(
        synthetic_umap[:, 0],
        synthetic_umap[:, 1],
        color="tab:orange",
        label="Synthetic",
    )
    if title is not None:
        ax.set_title(title, fontsize=18)
    ax.set_xlabel(f"{reduction_type} feature 1", fontsize=16)
    ax.set_ylabel(f"{reduction_type} feature 2", fontsize=16)
    ax.legend(fontsize=15)

    return fig

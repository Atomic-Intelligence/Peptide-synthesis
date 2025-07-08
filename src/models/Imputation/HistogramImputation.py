import numpy as np
from numpy.random import choice
import polars as pl
from tqdm import tqdm
from loguru import logger


class HistogramImputation:
    def __init__(self, column_names: list[str], num_bins: int):
        self.column_names = column_names
        self.num_bins = num_bins

    def fit(self, data: pl.DataFrame):
        self.col_densities = {col: [] for col in self.column_names}
        self.medians = {col: [] for col in self.column_names}

        for col in tqdm(self.column_names, desc="Calculating histogram imputation..."):
            col_values = data.select(col).to_numpy()
            hist, bin_edges = np.histogram(col_values, bins=self.num_bins)
            for i in range(len(bin_edges) - 1):
                up, low = bin_edges[i + 1], bin_edges[i]
                bin_idx = np.logical_and(col_values >= low, col_values <= up)
                if bin_idx.any():
                    self.medians[col].append(np.median(col_values[bin_idx]))
                else:
                    self.medians[col].append((up + low) / 2)

            self.col_densities[col] = hist / np.sum(hist)
        logger.success("Imputation done!")

    def generate(self, n: int) -> tuple[str, np.ndarray]:
        generated_data = [
            choice(self.medians[col], p=self.col_densities[col], size=n)[:, None]
            for col in self.column_names
        ]
        generated_data = np.concatenate(generated_data, axis=-1)
        return self.column_names, generated_data

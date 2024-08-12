from tqdm import tqdm
import polars as pl
import pandas as pd
import numpy as np

from src.data.data_models import Data, Processor


class HFProcessorForSynthetization(Processor):
    def __init__(
        self,
        primary_key: str,
    ):
        self.primary_key = primary_key

    def preprocess_data(self, data: Data) -> Data:
        data = self._preprocess_clinical_data(data)
        data = self._preprocess_peptide_data(data)
        return data

    def postprocess_data(
        self,
        data: Data,
        remaining_peptides: list[str],
        synthetic_data: pl.DataFrame,
    ) -> Data:
        remaining_columns = {}

        original_peptides = data.peptides.to_pandas()
        original_clinical = data.clinical.to_pandas()

        for peptide in tqdm(remaining_peptides, desc="Postprocessing"):
            if peptide not in original_peptides.columns:
                raise ValueError(
                    f"Peptide '{peptide}' not found in original data columns."
                )

            value = original_peptides[peptide].mean()

            missing_percentage = 1 - original_peptides[peptide].apply(
                lambda x: bool(x)
            ).sum() / len(original_peptides)
            missing_count = int(missing_percentage * len(synthetic_data))
            non_missing_count = len(synthetic_data) - missing_count

            values = [value] * non_missing_count + [0.0] * missing_count
            np.random.shuffle(values)

            remaining_columns[peptide] = values

        synthetic_data = pd.concat(
            [synthetic_data, (pd.DataFrame(remaining_columns))], axis=1
        )

        synthetic_data_clinical = synthetic_data[original_clinical.columns]
        synthetic_data_peptides = synthetic_data[
            [
                peptide if peptide != self.primary_key else peptide
                for peptide in original_peptides.columns
            ]
        ]

        print(f"Data postprocessed.")

        return Data(
            clinical=pl.from_pandas(synthetic_data_clinical),
            peptides=pl.from_pandas(synthetic_data_peptides),
        )

    @staticmethod
    def _preprocess_clinical_data(data: Data) -> Data:
        data.clinical = data.clinical.select(
            pl.col(col) for col in data.clinical.columns if col != ""
        )
        return data

    def _preprocess_peptide_data(self, data: Data) -> Data:
        data.peptides = data.peptides.rename({"": self.primary_key})
        data.peptides = data.peptides.with_columns(
            (pl.col(self.primary_key) / 1000).cast(pl.Int64)
        )
        return data

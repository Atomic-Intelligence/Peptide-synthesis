from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel
from tqdm import tqdm


class Data(BaseModel):
    """
    data structure containing both clinical and peptide data in polars dataframes
    """
    clinical: pl.DataFrame
    peptides: pl.DataFrame

    class Config:
        arbitrary_types_allowed = True


class Processor(ABC):
    def __init__(
            self,
            primary_key: str,
    ):
        """
        Abstract processor class for general clinical and peptide data manipulation
        Args:
            primary_key: column name of the primary key column
        """
        self.primary_key = primary_key

    @abstractmethod
    def preprocess_data(self, data: Data) -> Data:
        pass

    def postprocess_data(
            self,
            data: Data,
            remaining_peptides: list[str],
            synthetic_data: pl.DataFrame,
    ) -> Data:
        """
        postprocess synthesized data into final dataset
        Args:
            data: original data of real patients
            remaining_peptides: peptides which were not modeled using copulas due to not enough examples
            synthetic_data: result of the synthetic data generation using the copulas method

        Returns: synthetic dataset split into clinical and peptide tables

        """
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

    def get_peptides_for_modelling(
            self, data: pl.DataFrame, missing_threshold: float
    ) -> tuple[pl.DataFrame, list[str]]:
        """
        filters peptides which pass the frequency threshold needed to apply copulas to them
        Args:
            data: peptide data in a polars dataframe
            missing_threshold: if the missing data percentage goes above this value, copulas are not used

        Returns: polars dataframe of peptides for copula fitting and list of the remaining column names

        """
        # Replace 0 with None
        data = data.with_columns(
            [
                pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col)
                for col in data.columns
            ]
        )

        # Calculate the percentage of missing values in each column
        missing_values_percentages = data.select(
            pl.all().null_count() / pl.len()
        ).to_dict(as_series=False)

        # Identify columns with missing values below the threshold
        columns_to_model = [
            col
            for col, perc in missing_values_percentages.items()
            if perc[0] <= missing_threshold
        ]

        other_columns = set(data.columns) - set(columns_to_model)

        print(
            f"{len(columns_to_model) - 1} columns will be synthesized using advanced methods!"
        )
        print(
            f"{len(other_columns)} will be approximated using the mean as there is not enough data!"
        )

        # Select columns to model
        df = data.select(columns_to_model).with_columns(
            [
                (
                    pl.when(pl.col(col).dtype in [pl.Int64, pl.Float64])
                    .then(pl.col(col).cast(pl.Int64))
                    .when(pl.col(col).dtype == pl.Utf8)
                    .then(pl.col(col).cast(pl.Utf8))
                    .otherwise(pl.col(col).cast(pl.Float64))
                    .alias(col)
                    if col == self.primary_key
                    else pl.col(col).cast(pl.Float64)
                )
                for col in columns_to_model
            ]
        )

        return df, list(other_columns)

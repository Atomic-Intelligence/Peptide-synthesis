from abc import ABC, abstractmethod

import polars as pl
from pydantic import BaseModel


class Data(BaseModel):
    clinical: pl.DataFrame
    peptides: pl.DataFrame

    class Config:
        arbitrary_types_allowed = True


class Processor(ABC):
    @abstractmethod
    def preprocess_data(self, data: Data, *args) -> Data:
        pass

    @abstractmethod
    def postprocess_data(self, data: Data, *args) -> Data:
        pass

    def get_peptides_for_modelling(
        self, data: pl.DataFrame, missing_threshold: float
    ) -> tuple[pl.DataFrame, list[str]]:
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

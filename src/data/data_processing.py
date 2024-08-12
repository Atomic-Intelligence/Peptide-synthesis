import polars as pl

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

    def postprocess_data(self, data: Data) -> Data:
        # todo: check if this needs implementing
        return data

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

        print(f"{len(columns_to_model) - 1} columns will be synthesized using advanced methods!")
        print(f"{len(other_columns)} will be approximated using the mean as there is not enough data!")

        # Select columns to model
        df = data.select(columns_to_model).with_columns(
            [
                (
                    pl.col(col).cast(pl.Float64)
                    if col != self.primary_key
                    else pl.col(col).cast(pl.Int64)
                )
                for col in columns_to_model
            ]
        )

        return df, list(other_columns)

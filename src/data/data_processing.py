import polars as pl

from src.data.data_models import Data, Processor


class HFProcessorForSynthetization(Processor):
    def __init__(self, primary_key: str):
        super().__init__(primary_key)

    def preprocess_data(self, data: Data) -> Data:
        data = self._preprocess_clinical_data(data)
        data = self._preprocess_peptide_data(data)
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

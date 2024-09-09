import polars as pl

from src.data.data_models import Data, Processor


class HFProcessorForSynthetization(Processor):
    def __init__(self, primary_key: str):
        """
        Data processor for HF data
        Args:
            primary_key: primary key (patient id) column name
        """
        super().__init__(primary_key)

    def preprocess_data(self, data: Data) -> Data:
        """
        preprocess both peptides and clinical data
        Args:
            data: structure containing both peptides and clinical data
        Returns: preprocessed data
        """
        # First we preprocess data without null values
        data = self._preprocess_clinical_data(data)
        data = self._preprocess_peptide_data(data)

        # we exclude patients which contain null values
        valid_clinical_ids = data.clinical.drop_nulls()[self.primary_key].to_list()
        valid_peptides_ids = data.peptides.drop_nulls()[self.primary_key].to_list()
        valid_ids = set(valid_clinical_ids).intersection(set(valid_peptides_ids))
        print(len(valid_ids))
        data.clinical = data.clinical.filter(pl.col(self.primary_key).is_in(valid_ids))
        data.peptides = data.peptides.filter(pl.col(self.primary_key).is_in(valid_ids))
        return data

    def _preprocess_clinical_data(self, data: Data) -> Data:
        """
        clinical data preprocessing function
        Args:
            data: Data object containing clinical data as well as peptide data
        Returns: preprocessed data
        """
        data.clinical = data.clinical.select(
            pl.col(col) for col in data.clinical.columns if col != ""
        )
        if self.primary_key in data.clinical.columns:
            data.clinical = data.clinical.with_columns(
                pl.col(self.primary_key).cast(pl.Int64)
            )
        return data

    def _preprocess_peptide_data(self, data: Data) -> Data:
        """
        peptide data preprocessing function
        Args:
            data: Data object containing peptide data as well as clinical data
        Returns: preprocessed data
        """
        data.peptides = data.peptides.select(
            [pl.col(col) for col in data.peptides.columns if col != ""]
        )
        data.peptides = data.peptides.with_columns(
            (pl.col(self.primary_key) / 1000).cast(pl.Int64)
        )
        return data

from typing import Type, Any

import pandas as pd
import polars as pl
import sdv
import sdv.single_table.base

import numpy as np
import torch

from src.modeling.custom_copula_synthesizer import CustomGaussianCopulaSynthesizer


class Synthesizer:
    def __init__(
            self,
            original_data: pl.DataFrame,
            primary_key: str,
            peptides_to_model: list,
            constraints: list[dict[str, Any]],
            sdv_synthesizer: Type[
                sdv.single_table.base.BaseSingleTableSynthesizer
            ] = CustomGaussianCopulaSynthesizer,
            random_seed: int | None = None,
            *args,
            **kwargs,
    ):
        """
        Synthesizer for clinical+peptide data
        Args:
            original_data: dataframe containing the original real patient data
            primary_key: primary key column name
            peptides_to_model: peptides which should be modeled using the copula approach
            sdv_synthesizer: class which should be instantiated for the synthetic data model
            random_seed: seed for random number generator to be able to reproduce experiments
            constraints: deterministic constraints for columns
            *args: extra args for sdv_synthesizer
            **kwargs: extra kwargs for sdv_synthesizer
        """
        self.original_data = original_data
        self.primary_key = primary_key
        self.peptides_to_model = peptides_to_model
        self.random_seed = random_seed
        self.constraints = constraints

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        self.metadata = self._get_metadata(
            self.original_data.to_pandas(), self.peptides_to_model
        )

        self.sdv_synthesizer = sdv_synthesizer(metadata=self.metadata, *args, **kwargs)
        self._load_constraints()

    def sample(
            self,
            num_samples: int,
            batch_size: int | None = None,
    ) -> pd.DataFrame:
        """
        sample a synthetic dataset
        Args:
            num_samples: number of synthetic patients
            batch_size: generate data in batches of this size to speed up the process

        Returns: dataframe containing the synthetic dataset
        """
        return self.sdv_synthesizer.sample(num_samples, batch_size=batch_size)

    def fit(self):
        # Fit the model to the data
        self.sdv_synthesizer.fit(self.original_data.to_pandas())
        print("Model fitted.")

    def _get_metadata(
            self, dataset: pl.DataFrame, peptide_columns: list[str]
    ) -> sdv.metadata.SingleTableMetadata:
        """
        generate metadata for all columns so the synthesizer knows types of each variable
        Args:
            dataset: dataframe containing joined peptides and clinical data
            peptide_columns: list of peptide columns
        Returns: metadata object
        """
        metadata = sdv.metadata.SingleTableMetadata()
        metadata.detect_from_dataframe(dataset)

        for column in peptide_columns:
            if column == self.primary_key:
                metadata.update_column(column, sdtype="id")
            else:
                metadata.update_column(column, sdtype="numerical")

        return metadata

    def _load_constraints(self):
        """
        add constraints which enforce known rules data should adhere to
        i.e. diastolic blood pressure is always lower than systolic blood pressure
        """

        self.sdv_synthesizer.add_constraints(constraints=self.constraints)

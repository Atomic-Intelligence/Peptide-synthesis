from abc import ABC, abstractmethod
from typing import Type

import pandas as pd
import polars as pl
import sdv
import sdv.single_table.base

import numpy as np
import torch

from src.modeling.custom_copula_synthesizer import CustomGaussianCopulaSynthesizer


class Synthesizer(ABC):
    def __init__(
            self,
            original_data: pl.DataFrame,
            primary_key: str,
            peptides_to_model: list,
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
            *args: extra args for sdv_synthesizer
            **kwargs: extra kwargs for sdv_synthesizer
        """
        self.original_data = original_data
        self.primary_key = primary_key
        self.peptides_to_model = peptides_to_model
        self.random_seed = random_seed

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

    @abstractmethod
    def _load_constraints(self):
        pass


class HFSynthesizer(Synthesizer):
    def __init__(
            self,
            original_data: pl.DataFrame,
            primary_key: str,
            peptides_to_model: list,
            sdv_synthesizer: Type[
                sdv.single_table.base.BaseSingleTableSynthesizer
            ] = CustomGaussianCopulaSynthesizer,
            random_seed: int | None = None,
            *args,
            **kwargs,
    ):
        """
        Synthesizer for HF data with custom
        """
        super(HFSynthesizer, self).__init__(
            original_data=original_data,
            primary_key=primary_key,
            peptides_to_model=peptides_to_model,
            sdv_synthesizer=sdv_synthesizer,
            random_seed=random_seed,
            *args,
            **kwargs
        )

    def _load_constraints(self):
        """
        add constraints which enforce known rules data should adhere to
        i.e. diastolic blood pressure is always lower than systolic blood pressure
        """
        my_constraint = {
            "constraint_class": "Inequality",
            "constraint_parameters": {
                "low_column_name": "Blutdruck, diastolischM",
                "high_column_name": "Blutdruck, systolischM",
                "strict_boundaries": True,
            },
        }

        self.sdv_synthesizer.add_constraints(constraints=[my_constraint])

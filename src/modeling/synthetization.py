from abc import ABC, abstractmethod
from typing import Type

import polars as pl
import sdv
import sdv.single_table.base

from src.modeling.custom_copula_synthesizer import CustomGaussianCopulaSynthesizer


class Synthesizer(ABC):
    @abstractmethod
    def sample(self, num_samples: int, batch_size: int):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def _load_constraints(self):
        pass

    @abstractmethod
    def _get_metadata(self, dataset: pl.DataFrame, peptide_columns: pl.Series) -> sdv.metadata.SingleTableMetadata:
        pass


class HFSynthesizer(Synthesizer):
    """`save_to` is the directory where data will be saved to
    or is loaded from if already exists. If it is None, data will
    be saved to the current working directory."""

    def __init__(
        self,
        original_data: pl.DataFrame,
        primary_key: str,
        peptides_to_model: list,
        save_to: str | None = None,
        save_metadata: bool = False,
        save_peptides_over_threshold: bool = False,
        sdv_synthesizer: Type[sdv.single_table.base.BaseSingleTableSynthesizer] = CustomGaussianCopulaSynthesizer,
        *args,
        **kwargs,
    ):
        self.original_data = original_data
        self.save_to = save_to
        self.primary_key = primary_key
        self.peptides_to_model = peptides_to_model
        self.save_metadata = save_metadata
        self.save_peptides_over_threshold = save_peptides_over_threshold

        if save_to is not None:
            if self.save_peptides_over_threshold:
                # Save peptides that are used with copula approach to txt file
                with open(f"{save_to}/peptides_over_threshold.txt", "w") as f:
                    for peptide in self.peptides_to_model:
                        if peptide != self.primary_key:
                            f.write(f"{peptide}\n")

        self.metadata = self._get_metadata(
            self.original_data.to_pandas(), self.peptides_to_model
        )

        if save_to is not None:
            # create metadata and save it if necessary
            if self.save_metadata:
                self.metadata.save_to_json(f"{self.save_to}/metadata.json")

        self.sdv_synthesizer = sdv_synthesizer(metadata=self.metadata, *args, **kwargs)
        self._load_constraints()

    def sample(
        self,
        num_samples: int,
        batch_size: int | None = None,
    ):
        "The batch size defaults to `num_rows`, if None."
        return self.sdv_synthesizer.sample(num_samples, batch_size=batch_size)

    def fit(self):
        # Fit the model to the data
        self.sdv_synthesizer.fit(self.original_data.to_pandas())
        print("Model fitted.")

    def _get_metadata(
        self, dataset: pl.DataFrame, peptide_columns: list[str]
    ) -> sdv.metadata.SingleTableMetadata:

        metadata = sdv.metadata.SingleTableMetadata()
        metadata.detect_from_dataframe(dataset)

        for column in peptide_columns:
            if column == self.primary_key:
                metadata.update_column(column, sdtype="id")
            else:
                metadata.update_column(column, sdtype="numerical")

        return metadata

    def _load_constraints(self):
        """Loads given constraints defined in class
        from the src/generation/constraints.py."""

        # ----- For HF, systolic blood pressure must be greater than diastolic -----

        my_constraint = {
            "constraint_class": "Inequality",
            "constraint_parameters": {
                "low_column_name": "Blutdruck, diastolischM",
                "high_column_name": "Blutdruck, systolischM",
                "strict_boundaries": True,
            },
        }

        self.sdv_synthesizer.add_constraints(constraints=[my_constraint])

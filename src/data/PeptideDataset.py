from loguru import logger
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from functools import cached_property

import polars as pl
import torch

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch.utils.data import DataLoader, Dataset

from v0.src import (
    CATEGORICAL_CLINICAL_COLUMNS,
    NUMERICAL_CLINICAL_COLUMNS,
    select_top_n_peptides,
    split_zero_columns,
)


COLUMNS_TO_DROP = [
    "idAuswertung",
    "time-to-CKDevent(to event or last visit)",
    "FU duration_CAD_Hfevent (to event or last visit)",
]

SOURCE_COLUMN = [
    "event_type"  # name of the column that contains the source of the patient
]


class Identifiers(str, Enum):
    peptide = "Peptide"
    peptide_value = "value"
    peptide_present = "present"


def check_column_mapping(df: pl.DataFrame):
    clinical_columns = {
        col for col in df.columns if Identifiers.peptide.value not in col
    }

    mapped_columns = set(
        CATEGORICAL_CLINICAL_COLUMNS
        + NUMERICAL_CLINICAL_COLUMNS
        + [SOURCE_COLUMN]
        + COLUMNS_TO_DROP
        + ["event_type"]
    )

    assert (
        clinical_columns == mapped_columns
    ), f"Columns not mapped correctly: {clinical_columns - mapped_columns}"


@dataclass
class PeptideDataItem:
    encoded: torch.Tensor  # patient row encoded as a vector
    source: str  # class label of original source dataset of the patient
    column_names: list[str]  # list of column names of the encoded patient row


class PeptideDataset(Dataset):
    def __init__(
        self, dataset: pl.DataFrame, n_peptides: int, split_peptides: bool = True
    ):
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.quantile_transformer = QuantileTransformer(output_distribution="normal")

        # check if columns are mapped correctly
        check_column_mapping(dataset)

        self.dataset = dataset.fill_null(0.0)

        # print(list(self.dataset["Peptide_99901132"]))
        self.split_peptides = split_peptides

        # remove id columns
        self.dataset = self.dataset.drop(COLUMNS_TO_DROP, strict=False)

        # keep only n peptides which have the least amount of missing values
        self.dataset = select_top_n_peptides(
            df=self.dataset,
            n_peptides=n_peptides,
            peptide_columns=self.get_peptide_columns(self.dataset),
        )
        # start by decomposing peptides into a product of numerical and categorical variables
        if self.split_peptides:
            self.dataset = split_zero_columns(
                df=self.dataset, columns=self.get_peptide_columns(self.dataset)
            )
            logger.info("Split peptides into numerical and categorical!")

        logger.debug(self.dataset)

        # fit encoders for both numerical and categorical columns
        self.setup()

    def setup(self):
        # process categorical clinical variables and indicators of peptide activity
        self.onehot_encoder.fit(
            self.dataset.select(self.categorical_columns).to_numpy()
        )

        # process numerical clinical variables and peptide values
        self.quantile_transformer.fit(
            self.dataset.select(self.numerical_colums).to_numpy()
        )

    @cached_property
    def peptide_columns(self) -> list[str]:
        return self.get_peptide_columns(self.dataset)

    @cached_property
    def categorical_columns(self) -> list[str]:
        return self.get_categorical_columns(self.dataset)

    @cached_property
    def numerical_colums(self) -> list[str]:
        return self.get_numerical_columns(self.dataset, self.split_peptides)

    @staticmethod
    def get_peptide_columns(dataset: pl.DataFrame) -> list[str]:
        return [c for c in dataset.columns if Identifiers.peptide.value in c]

    @staticmethod
    def get_categorical_columns(dataset: pl.DataFrame) -> list[str]:
        return CATEGORICAL_CLINICAL_COLUMNS + [
            c for c in dataset.columns if Identifiers.peptide_present.value in c
        ]

    @staticmethod
    def get_numerical_columns(dataset: pl.DataFrame, split_peptides: bool) -> list[str]:
        # if we split the peptides, values are numerical, otherwise the original peptide column is numerical
        if split_peptides:
            return NUMERICAL_CLINICAL_COLUMNS + [
                c for c in dataset.columns if Identifiers.peptide_value.value in c
            ]
        else:
            return NUMERICAL_CLINICAL_COLUMNS + [
                c for c in dataset.columns if Identifiers.peptide.value in c
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> dict[str, torch.Tensor | str | list[str]]:
        peptide = self.dataset[idx, :]

        # extract source of the patient
        source = peptide[SOURCE_COLUMN][0]
        peptide = peptide.drop(SOURCE_COLUMN)

        # split into categorical and numerical columns
        categorical = peptide.select(self.categorical_columns)
        numerical = peptide.select(self.numerical_colums)

        # quantile transform numerical columns
        numerical = pl.DataFrame(
            self.quantile_transformer.transform(numerical.to_numpy()),
            schema=numerical.schema,
        )

        # one-hot encode categorical columns
        categorical_column_names = list(self.onehot_encoder.get_feature_names_out())
        oh_categorical = pl.DataFrame(
            self.onehot_encoder.transform(categorical.to_numpy().astype(float)),
            schema=categorical_column_names,
        )

        # combine all columns
        encoded = numerical.hstack(oh_categorical)
        logger.debug(encoded)

        data_item = PeptideDataItem(
            encoded=torch.tensor(encoded.to_numpy()),
            source=source,
            column_names=encoded.columns,
        )

        return asdict(data_item)


def get_dataloader(
    dataset_path: Path,
    n_peptides: int,
    batch_size: int,
    split_peptides: bool = True,  # if False, peptides are treated as a single numerical variable
    shuffle: bool = True,
    n_rows: int | None = None,  # None means all rows
) -> DataLoader:
    dataset = pl.read_csv(dataset_path, n_rows=n_rows)
    dataset = dataset.fill_null(0.0)

    return DataLoader(
        PeptideDataset(dataset, n_peptides=n_peptides, split_peptides=split_peptides),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def main(dataset_path: Path, n_peptides: int, n_rows: int | None = None):
    start = time.time()
    dataloader = get_dataloader(
        dataset_path=dataset_path,
        n_peptides=n_peptides,
        split_peptides=True,
        batch_size=1,
        n_rows=n_rows,
    )
    end = time.time()
    logger.info(f"Loaded data in {end - start:.2f} seconds")

    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    import sys

    logger.add(sys.stdout)
    main(
        n_peptides=100,
        n_rows=500,
    )

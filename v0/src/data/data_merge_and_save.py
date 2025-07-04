from pathlib import Path
import polars as pl
import os
import random


def merge_and_save(
    clinical_data_list: list[pl.DataFrame],
    peptides_data_list: list[pl.DataFrame],
    primary_key: str,
    save_to: Path | None = None,
) -> None:
    """
    function which merges multiple groups of synthetic patients into single tables for clinical
    and peptide data
    note: If `save_to` parameter is None, data will be saved to current working directory.
    Args:
        clinical_data_list: list of clinical synthetic dataframes
        peptides_data_list: list of peptide synthetic dataframes
        save_to: path to directory where data will be saved
    """

    if len(clinical_data_list) != len(peptides_data_list):
        raise ValueError("Length of clinical and peptides data lists must be equal.")

    clinical_data_merged = pl.concat(clinical_data_list)
    n = clinical_data_merged.shape[0]
    random_number = random.randint(0, 10000)
    clinical_data_merged = clinical_data_merged.with_columns(
        (pl.arange(1, n + 1) + pl.lit(random_number)).alias(primary_key)
    )
    peptides_data_merged = pl.concat(peptides_data_list).fill_null(0.0)
    peptides_data_merged = peptides_data_merged.with_columns(
        (pl.arange(1, n + 1) + pl.lit(random_number)).alias(primary_key)
    )

    Path.mkdir(save_to, exist_ok=True) if save_to is not None else os.getcwd()

    clinical_data_merged.write_csv(
        Path(save_to, "synthetic_data_clinical.csv"), include_header=True
    )
    peptides_data_merged.write_csv(
        Path(save_to, "synthetic_data_peptides.csv"), include_header=True
    )

    print(f"Data saved to: {save_to}.")

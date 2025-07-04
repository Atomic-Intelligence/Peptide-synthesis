from pathlib import Path

import polars as pl


def merge_synthetic_datasets(datasets: list[tuple[Path, Path, str]], save_to: Path):
    dfs = []

    for peptide_path, clinical_path, dataset_name in datasets:
        df_peptide = pl.read_csv(peptide_path)

        df_peptide = df_peptide.rename(
            mapping={
                c: f"Peptide_{int(float(c))}"
                for c in df_peptide.columns
                if c != "idAuswertung"
            },
        )

        df_clinical = pl.read_csv(clinical_path)

        df = df_peptide.join(df_clinical, on="idAuswertung")

        modified_df = df.with_columns(pl.lit(dataset_name).alias("event_type"))
        dfs.append(modified_df)

    all_columns = sorted(list(set.union(*[set(df.columns) for df in dfs])))
    column_dtypes = {}
    for df in dfs:
        for col in df.columns:
            if col not in column_dtypes:
                column_dtypes[col] = df[col].dtype

    # Standardize each DataFrame
    modified_dfs = []
    for df in dfs:
        new_dict = {}
        for col in all_columns:
            if col in df.columns:
                new_dict[col] = df[col]
            else:
                new_dict[col] = pl.Series(
                    col, [None] * len(df), dtype=column_dtypes[col]
                )
        new_df = pl.from_dict(new_dict)
        modified_dfs.append(new_df)

    concatenated_df = pl.concat(modified_dfs, how="vertical")

    concatenated_df.write_csv(save_to)


def main():
    datasets = [
        (
            Path("/data1/prostrat-ai/bootstraping_v0/hf/synthetic_data_peptides.csv"),
            Path("/data1/prostrat-ai/bootstraping_v0/hf/synthetic_data_clinical.csv"),
            "hf",
        ),
        (
            Path("/data1/prostrat-ai/bootstraping_v0/ckd/synthetic_data_peptides.csv"),
            Path("/data1/prostrat-ai/bootstraping_v0/ckd/synthetic_data_clinical.csv"),
            "ckd",
        ),
        (
            Path("/data1/prostrat-ai/bootstraping_v0/ne/synthetic_data_peptides.csv"),
            Path("/data1/prostrat-ai/bootstraping_v0/ne/synthetic_data_clinical.csv"),
            "no_event",
        ),
    ]

    merge_synthetic_datasets(
        datasets, save_to=Path("/data1/prostrat-ai/synthetic_datasets/paper.csv")
    )


if __name__ == "__main__":
    main()

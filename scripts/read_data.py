import polars as pl
import os


def concatenate_peptide_tables_from_files(
    path: str, filenames: list[str]
) -> pl.DataFrame:
    """
    concatenate multiple excel tables into a single resulting polars dataframe
    Args:
        path: path to the root directory containing the Excel tables
        filenames: filenames of the Excel tables
    Returns: dataframe containing the concatenated tables
    """
    tables = []
    for filename in filenames:
        print(f"Reading file {filename}")
        filename_path = os.path.join(path, f"{filename}.xlsx")
        if not os.path.exists(filename_path):
            print(f"File not found {filename_path}")
            continue

        data = pl.read_excel(
            source=filename_path,
            has_header=False,
            read_options={"skip_rows": 1},  # Skip the first row
        )
        print(f"File read {filename_path}")

        # Set the second row as the header
        headers = [str(value) for value in data.row(0)]  # Convert all values to strings
        data = data.slice(1)  # Drop the first row (header row)
        data.columns = headers  # Assign the extracted headers

        # Save the idMuster column values (first column) to use as headers later
        id_muster_values = data["idMuster"]

        # Drop the idMuster column before transposing
        data = data.drop("idMuster")

        # Transpose the data
        transpose_data = data.transpose(include_header=True, header_name="idAuswertung")
        transpose_data.columns = ["idAuswertung"] + [
            "Peptide_" + peptide for peptide in id_muster_values.to_list()
        ]

        transpose_data = transpose_data.with_columns(
            (
                pl.col("idAuswertung")
                .str.strip_chars(".0")  # Remove `.0` suffix
                .cast(pl.Int64)  # Convert to integer
            ).alias("idAuswertung")
        )
        tables.append(transpose_data)

    # Concatenate all tables vertically
    if tables:
        new_table = pl.concat(tables, how="vertical")
        return new_table
    return pl.DataFrame()


def merge_clinical_and_peptide_data(
    clinical_data: pl.DataFrame, peptide_data: pl.DataFrame
):
    # First, merge the dataframes on idAuswertung
    merged_df = peptide_data.join(
        clinical_data, left_on="idAuswertung", right_on="idAuswertung", how="right"
    )

    merged_df.write_csv("merged.csv")

    # Check if expected columns exist
    needed_columns = ["CKD.event", "CAD.event", "HF.event"]
    missing_columns = [col for col in needed_columns if col not in merged_df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    # # Create the event column using when-then logic
    merged_df = merged_df.with_columns(
        [
            pl.when(pl.col("CKD.event").cast(pl.Utf8) == "1")
            .then(pl.lit("ckd"))
            .when(pl.col("CAD.event").cast(pl.Utf8) == "1")
            .then(pl.lit("cad"))
            .when(pl.col("HF.event").cast(pl.Utf8) == "1")
            .then(pl.lit("hf"))
            .otherwise(pl.lit("no_event"))
            .alias("event_type")
        ]
    )
    #
    # # Drop the original event columns
    no_event = merged_df.filter(pl.col("event_type") == "no_event")
    to_keep = no_event.filter(
        (pl.col("CKD.event").cast(pl.Utf8) == "0")
        & (pl.col("CAD.event").cast(pl.Utf8) == "0")
        & (pl.col("HF.event").cast(pl.Utf8) == "0")
    )
    merged_df = merged_df.filter(pl.col("event_type") != "no_event")
    merged_df = pl.concat([merged_df, to_keep])
    merged_df = merged_df.drop(["CKD.event", "CAD.event", "HF.event"])
    return merged_df


def get_dataframe(root_path: str, clinical_data_path: str):
    # Load the clinical data
    clinical_data = pl.read_excel(clinical_data_path, sheet_name="clin.data")
    # load the list.datasets
    dataset_list = pl.read_excel(clinical_data_path, sheet_name="list.datasets")
    # get filenames
    filenames = dataset_list["File name"].unique().sort().to_list()
    # concatenate peptide data
    peptide_data = concatenate_peptide_tables_from_files(root_path, filenames)
    # merge
    final_dataframe = merge_clinical_and_peptide_data(clinical_data, peptide_data)
    return final_dataframe


def main():
    clinical_data_path = "/data1/prostrat-ai/data/AnalysisIDs_List_Synthetic_patients_final_INPUT file_V2.xlsx"
    root_path = "/data1/prostrat-ai/peptide_original_data"
    concatenated_table = get_dataframe(root_path, clinical_data_path)
    concatenated_table.write_csv("output3.csv")


if __name__ == "__main__":
    main()

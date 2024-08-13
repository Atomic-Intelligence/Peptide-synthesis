from pathlib import Path
import pandas as pd
import os


def concatenate_tables_from_files(path: str, filenames: list[str]) -> pd.DataFrame:
    """
    concatenate multiple excel tables into a single resulting pandas dataframe
    Args:
        path: path to the root directory containing the Excel tables
        filenames: filenames of the Excel tables

    Returns: dataframe containing the concatenated tables
    """
    tables = []
    for i in range(len(filenames)):
        filename_path = os.path.join(path, (filenames[i] + ".xlsx"))
        if not os.path.exists(filename_path):
            continue
        data = pd.read_excel(filename_path, header=1)
        transpose_data = data.T
        transpose_data.columns = transpose_data.iloc[0]
        transpose_data = transpose_data[1:]
        tables.append(transpose_data)
    new_table = pd.concat(tables)
    return new_table


def merge_hf_data(root_dir_path: str, save_dir_path: str):
    """
    This script is used to merge the heart failure data used in the research.
    This is a custom script and is not reproducible for data of a different type.
    :param root_dir_path: path to the directory where all excel tables used for research are located
    :param save_dir_path: path to the directory where the merged data will be saved
    """
    # Check if the directory exists and is not empty
    if Path(root_dir_path).exists() and any(Path(root_dir_path).iterdir()):
        # Load Excel file and sheets
        excel_data = pd.ExcelFile(
            root_dir_path + "AnalysisIDs_List_Synthetic_patients_final.xlsx"
        )
        filenames_data = pd.read_excel(excel_data, sheet_name="list.datasets")
        filenames_data = filenames_data.drop(
            filenames_data.columns[[4, 5, 6, 7, 8, 9]], axis=1
        )
        clin_data = pd.read_excel(excel_data, sheet_name="clin.data")

        # Process HF clinical data
        print("HF clinical data processing and saving")
        hf_clin_data = clin_data[clin_data["HF.event"] == 1].reset_index(drop=True)
        hf_clin_data = hf_clin_data.drop(["HF.event", "CAD.event"], axis=1)

        # save HF clinical data
        Path(save_dir_path).mkdir(exist_ok=True, parents=True)
        hf_save_path = os.path.join(save_dir_path, "hf_clinical_data.csv")
        hf_clin_data.to_csv(hf_save_path, index=False)

        # Process HF peptide data
        print("HF peptide data processing and saving")
        hf_filenames = filenames_data[filenames_data["Group"] == "HF"][
            "File name"
        ].unique()
        hf_peptides_data_merged = concatenate_tables_from_files(
            root_dir_path, hf_filenames
        )
        df_reset = hf_peptides_data_merged.reset_index()
        df_reset.rename(columns={"index": "idAuswertung"}, inplace=True)

        # save HF peptide data
        hf_peptides_path = os.path.join(save_dir_path, "hf_peptides_data.csv")
        df_reset.to_csv(hf_peptides_path, index=False)

        # Process no-event clinical data
        print("no-event clinical data processing and saving")
        no_event_clin_data = clin_data[
            (clin_data["CAD.event"] == 0) & (clin_data["HF.event"] == 0)
        ].reset_index(drop=True)
        no_event_clin_data = no_event_clin_data.drop(["HF.event", "CAD.event"], axis=1)

        # save no-event clinical data
        no_event_clinical_path = os.path.join(
            save_dir_path, "no_event_clinical_data.csv"
        )
        no_event_clin_data.to_csv(no_event_clinical_path, index=False)

        # Process no-event peptide data
        print("no-event peptide data processing and saving")
        no_event_filenames = filenames_data[
            filenames_data["Group"].isin(
                ["no_event_1", "no_event_2", "no_event_3", "no_event_4", "no_event_5"]
            )
        ]["File name"].unique()
        no_event_peptides_data = concatenate_tables_from_files(
            root_dir_path, no_event_filenames
        )
        df_ne_reset = no_event_peptides_data.reset_index()
        df_ne_reset.rename(columns={"index": "idAuswertung"}, inplace=True)

        # save no-event peptide data
        no_event_peptides_path = os.path.join(
            save_dir_path, "no_event_peptides_data.csv"
        )
        df_ne_reset.to_csv(no_event_peptides_path, index=False)
        return
    else:
        raise FileNotFoundError(
            f"Directory {root_dir_path} does not exist or is empty."
        )

import polars as pl
import re
from loguru import logger
from typing import Optional, List, Union, Tuple

CATEGORICAL_CLINICAL_COLUMNS = [
    "Geschlecht (1=female)",
    "Kidney disease",
    "diabetes",
    "CVD",
    "hypertension",
]
NUMERICAL_CLINICAL_COLUMNS = [
    "Blutdruck, diastolischM",
    "Blutdruck, systolischM",
    "GFR_CKD_EPI_M",
    "BMI_M",
    "Alter_M",
    "idAuswertung",
]

TIME_TO_EVENT_COLUMNS = [
    "time-to-CKDevent(to event or last visit)",
    "FU duration_CAD_Hfevent (to event or last visit)",
    # "FU duration (to event or last visit)",
]


def get_peptide_columns(df: pl.DataFrame) -> list[str]:
    pattern = re.compile("peptide", re.IGNORECASE)
    return [col for col in df.columns if re.search(pattern, col) is not None]


def split_peptide_columns_by_zero_percentage(
    df: pl.DataFrame, zero_perc_threshold: float = 0.3
):
    df = df.select(get_peptide_columns(df))
    total_rows = df.height

    # Calculate zero percentages for all columns at once
    zero_percentages = df.select(
        [(pl.col(col) == 0).sum() / total_rows for col in df.columns]
    ).to_dicts()[0]

    above_zero_threshold = [
        col for col, pct in zero_percentages.items() if pct > zero_perc_threshold
    ]
    below_zero_threshold = [
        col for col, pct in zero_percentages.items() if pct <= zero_perc_threshold
    ]
    return below_zero_threshold, above_zero_threshold


def select_top_n_peptides(
    df: pl.DataFrame, n_peptides: int, peptide_columns: list[str]
) -> pl.DataFrame:
    # Get missing value counts for each column
    zero_counts = (
        df[peptide_columns]
        .select(pl.all().eq(0).sum())
        .transpose(
            column_names=["zero_count"], include_header=True, header_name="column"
        )
    ).sort("zero_count", descending=False)

    # Get columns to keep (top N with the least zeros)
    columns_to_keep = zero_counts.head(n_peptides)["column"].to_list()

    # Get columns to drop (all peptides except the top N)
    columns_to_drop = [col for col in peptide_columns if col not in columns_to_keep]

    # Return DataFrame without the dropped columns
    return df.drop(columns_to_drop)


def split_zero_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    for col in columns:
        df = df.with_columns(
            [
                (~pl.col(col).eq(0)).alias(f"{col}_present"),
                pl.when(pl.col(col).eq(0))
                .then(pl.col(col).filter(~pl.col(col).eq(0)).mean())
                .otherwise(pl.col(col))
                .alias(f"{col}_value"),
            ]
        )
        df = df.drop(col)
    return df


class DataProcessor:
    def __init__(
        self,
        dfs: Union[pl.DataFrame, List[pl.DataFrame]],
        id_column: str,
        clinical_columns: Optional[List[str]],
    ):
        """Initialize the DataProcessor with a list of DataFrames and optional clinical columns."""
        self.dfs = [dfs] if isinstance(dfs, pl.DataFrame) else dfs
        self.id_column = id_column
        self.dfs = [df.filter(df["event_type"] != "cad") for df in self.dfs]

        self.clinical_columns = clinical_columns
        self.processed_dfs = self.dfs  # Start with the original datasets

        self.dfs_for_imputation = []

        for df in self.processed_dfs:
            print(len(df))

    def split_event_control(self, event: str = "event") -> "DataProcessor":
        """Split datasets into event and control groups based on the event type.

        If event is 'all' or 'all_events', returns each DataFrame as-is without splitting.
        """
        result_dfs = []

        if event in ("all", "all_events"):
            for df in self.processed_dfs:
                event_df = df
                result_dfs.append((event_df, None))
            self.processed_dfs = result_dfs
            return self

        for df in self.processed_dfs:
            event_df = df.filter(pl.col("event_type") == event)
            control_df = df.filter(pl.col("event_type") == "no_event")
            result_dfs.append((event_df, control_df))
        self.processed_dfs = result_dfs
        return self

    def filter_peptides(self, non_zero_threshold: float = 30.0) -> "DataProcessor":
        """Filter peptides based on a non-zero value threshold."""
        filtered_dfs = []
        for item in self.processed_dfs:
            # Handle both tuples (from split_event_control) and DataFrames
            if isinstance(item, tuple):
                event_df, control_df = item
                event_df = self._filter_peptides_in_df(event_df, non_zero_threshold)
                control_df = self._filter_peptides_in_df(control_df, non_zero_threshold)
                filtered_dfs.append((event_df, control_df))
            else:
                df = self._filter_peptides_in_df(item, non_zero_threshold)
                filtered_dfs.append(df)
        self.processed_dfs = filtered_dfs
        return self

    def get_data_for_imputation(self):
        return self.dfs_for_imputation

    def _filter_peptides_in_df(
        self, df: pl.DataFrame, non_zero_threshold: float
    ) -> pl.DataFrame:
        """Helper method to filter peptides in a single DataFrame."""
        peptide_columns = [
            col
            for col in df.columns
            if re.search("peptide", col, re.IGNORECASE) and "missing" not in col
        ]
        filtered_columns = [
            col
            for col in peptide_columns
            if (df[col] != 0).sum() / len(df) * 100 > non_zero_threshold
        ]

        logger.info(f"Working with {len(filtered_columns)} peptides")

        remaining_columns = set(peptide_columns) - set(filtered_columns)
        self.dfs_for_imputation.append(df.select(remaining_columns))
        logger.info(
            f"Remaining columns: {len(remaining_columns)} will be modeled using imputation."
        )

        if self.clinical_columns:
            valid_clinical_columns = [
                col for col in self.clinical_columns if col in df.columns
            ]
            missing_columns = set(self.clinical_columns) - set(valid_clinical_columns)
            if missing_columns:
                logger.warning(f"Clinical columns not found: {missing_columns}")
            filtered_columns.extend(valid_clinical_columns)
        return df.select(filtered_columns + ["event_type"] + [self.id_column])

    def select_clinical_columns(self) -> "DataProcessor":
        """Select only the clinical columns from the datasets."""
        clinical_dfs = []
        for item in self.processed_dfs:
            # Handle both tuples (from split_event_control) and DataFrames
            if isinstance(item, tuple):
                event_df, control_df = item
                event_df = self._select_clinical_columns_in_df(event_df)
                control_df = self._select_clinical_columns_in_df(control_df)
                clinical_dfs.append((event_df, control_df))
            else:
                df = self._select_clinical_columns_in_df(item)
                clinical_dfs.append(df)
        self.processed_dfs = clinical_dfs
        return self

    def _select_clinical_columns_in_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Helper method to select clinical columns in a single DataFrame."""
        valid_clinical_columns = [
            col for col in self.clinical_columns if col in df.columns
        ]
        return df.select(valid_clinical_columns + ["event_type"])

    def get_processed_data(
        self,
    ) -> List[Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]]:
        """Retrieve the processed data."""
        return self.processed_dfs

    def sample_patients_with_preserved_ratios(
        self,
        n_patients: int,
        event_col: str = "event_type",
        random_seed: int = None,
    ) -> "DataProcessor":
        logger.info(
            f"Sampling {n_patients} patients per dataset while preserving event ratios..."
        )

        sampled_dfs = []

        for i, df in enumerate(self.processed_dfs):
            logger.info(
                f"Processing DataFrame {i + 1}/{len(self.processed_dfs)} with shape {df.shape}"
            )

            df = df.filter(pl.col(event_col) != "cad")
            logger.info(f"Filtered out 'cad' events. Remaining shape: {df.shape}")

            # set different seed for each df
            if random_seed is not None:
                pl.random.seed(random_seed + i)

            patients_df = df.select([self.id_column, event_col]).unique(
                subset=[self.id_column]
            )
            event_counts = patients_df.group_by(event_col).agg(pl.len().alias("count"))

            print(event_counts)

            total_patients = patients_df.height

            if n_patients >= total_patients:
                logger.info(
                    f"Requested {n_patients} exceeds or equals available {total_patients}, using all patients."
                )
                sampled_dfs.append(df)
                continue

            target_counts = dict(
                zip(
                    event_counts[event_col].to_list(),
                    event_counts.with_columns(
                        ((pl.col("count") / total_patients) * n_patients)
                        .round(0)
                        .cast(pl.Int64)
                        .alias("target_count")
                    )["target_count"].to_list(),
                )
            )

            total_target = sum(target_counts.values())
            if total_target != n_patients:
                largest_group = max(target_counts.items(), key=lambda x: x[1])[0]
                target_counts[largest_group] += n_patients - total_target
                logger.info(
                    f"Adjusted count for event '{largest_group}' to correct rounding error."
                )

            sampled_patients_dfs = []
            for event_type, target_count in target_counts.items():
                if target_count <= 0:
                    logger.warning(
                        f"Target count for event '{event_type}' is <= 0 — skipping sampling for this event."
                    )
                    continue

                event_patients = patients_df.filter(pl.col(event_col) == event_type)
                available_count = event_patients.height
                actual_count = min(target_count, available_count)

                if actual_count <= 0:
                    logger.warning(
                        f"No patients available for event '{event_type}' "
                        f"(needed: {target_count}, available: {available_count}) — skipping."
                    )
                    continue

                sampled = event_patients.sample(n=actual_count, seed=random_seed)
                sampled_patients_dfs.append(sampled)

            if sampled_patients_dfs:
                selected_ids = pl.concat(sampled_patients_dfs)[self.id_column].to_list()
                sampled_df = df.filter(pl.col(self.id_column).is_in(selected_ids))
                sampled_dfs.append(sampled_df)
                logger.success(
                    f"Sampled {len(selected_ids)} unique patients for dataset {i + 1}"
                )
            else:
                sampled_dfs.append(pl.DataFrame())
                logger.warning(
                    f"No patients sampled for dataset {i + 1}. Resulting DataFrame is empty."
                )

        self.processed_dfs = sampled_dfs
        return self


if __name__ == "__main__":
    original_df = pl.read_csv("/data1/prostrat-ai/data/merged_peptide_and_clinical.csv")

    # Initialize the DataProcessor with the DataFrame and wanted clinical columns
    processor = DataProcessor(
        original_df,
        clinical_columns=CATEGORICAL_CLINICAL_COLUMNS
        + NUMERICAL_CLINICAL_COLUMNS
        + TIME_TO_EVENT_COLUMNS,
    )

    # Chain methods to perform multiple operations
    processed_data = (
        processor.filter_peptides(non_zero_threshold=30.0)
        # .select_clinical_columns()
        .split_event_control(event="hf").get_processed_data()
    )

    print(processed_data[0][0].shape)

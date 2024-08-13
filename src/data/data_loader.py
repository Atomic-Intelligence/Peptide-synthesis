import polars as pl

from src.data.data_models import Data, Processor


class DataLoader:
    def __init__(
            self,
            clinical_data_path: str,
            peptides_data_path: str,
            primary_key: str,
            number_of_samples: int | None = None,
            processor: Processor | None = None,
            filter_conditions: dict | None = None
    ):
        """
        dataloader class used to parse clinical and peptide data
        Args:
            clinical_data_path: path to clinical data
            peptides_data_path: path to peptide data
            primary_key: column name of primary key column
            number_of_samples: number of samples to use, if None all data is used
            processor: Processor object to use for data processing
            filter_conditions: conditions to use for filtering data
        """
        self.clinical_data_path = clinical_data_path
        self.peptides_data_path = peptides_data_path
        self.processor = processor
        self.primary_key = primary_key
        self.number_of_samples = number_of_samples
        self.filter_conditions = filter_conditions

    def get_data(self) -> Data:
        """If processor is provided, it will preprocess loaded data. Otherwise, raw data is loaded."""
        try:
            clinical = pl.read_csv(self.clinical_data_path)
            peptides = pl.read_csv(self.peptides_data_path)
            data = Data(clinical=clinical, peptides=peptides)

            if self.processor:
                data = self.processor.preprocess_data(data)

            if self.number_of_samples:
                data.clinical = data.clinical.sort(self.primary_key).slice(0, self.number_of_samples)
                data.peptides = data.peptides.sort(self.primary_key).slice(0, self.number_of_samples)

            if self.filter_conditions:
                data.clinical = self._apply_filter_conditions(data.clinical)

            return data

        except Exception as e:
            print(e, "\n")
            raise ValueError("Data is not loaded correctly!")

    def _apply_filter_conditions(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies the specified filter conditions to the DataFrame.

        Parameters:
        - df (pl.DataFrame): The DataFrame to apply filters to.

        Returns:
        - pl.DataFrame: The filtered DataFrame.
        """
        valid_operators = {"==", ">", "<", ">=", "<="}

        for column, condition in self.filter_conditions.items():
            # Check if the column exists in the DataFrame
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

            if isinstance(condition, tuple):
                # Validate tuple conditions like (">", 50)
                if len(condition) != 2:
                    raise ValueError(
                        f"Invalid condition format for column '{column}': {condition}. Expected a tuple of (operator, value).")
                operator, value = condition
                if operator not in valid_operators:
                    raise ValueError(
                        f"Invalid operator '{operator}' for column '{column}'. Expected one of {valid_operators}.")
                df = self._apply_operator(df, column, operator, value)
            else:
                # Default equality condition
                df = df.filter(pl.col(column) == condition)

        # Check if the resulting DataFrame is empty
        if df.is_empty():
            raise ValueError("The filtering operation resulted in an empty DataFrame.")

        return df

    @staticmethod
    def _apply_operator(df: pl.DataFrame, column: str, operator: str, value) -> pl.DataFrame:
        """
        Helper method to apply an operator-based condition to a DataFrame column.
        """
        if operator == "==":
            return df.filter(pl.col(column) == value)
        elif operator == ">":
            return df.filter(pl.col(column) > value)
        elif operator == "<":
            return df.filter(pl.col(column) < value)
        elif operator == ">=":
            return df.filter(pl.col(column) >= value)
        elif operator == "<=":
            return df.filter(pl.col(column) <= value)
        else:
            raise ValueError(f"Unsupported operator '{operator}' for column '{column}'.")

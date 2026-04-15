import random
from typing import Tuple, Dict, List, Optional, Union, Any

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import norm, rankdata

from src.logger import setup_logger

logger = setup_logger()

<<<<<<< HEAD

class SDVPreprocessor:
=======
_ENCODING_TRUNCATED_GAUSSIAN = "truncated_gaussian"
_ENCODING_NORMAL_SCORES = "normal_scores"


class SDVPreprocessor:
    """Preprocessor for the Gaussian Copula model.

    Handles missing-value imputation (with indicator columns) and categorical
    encoding.  Supports two categorical encoding strategies:

    - ``"truncated_gaussian"`` (default / current behaviour): each category is
      mapped to a random sample drawn from a truncated Gaussian on its cumulative
      probability interval ``[F_{i-1}, F_i)``.  Values live in ``[0, 1]``.

    - ``"normal_scores"`` (improvement 2.1): each category is deterministically
      mapped to ``Φ⁻¹((F_{i-1} + F_i) / 2)`` — the midpoint of its interval
      pushed through the probit transform.  Values live in ℝ and are directly
      interpretable as Gaussian copula scores, avoiding the false randomness of
      the truncated-Gaussian approach.  The inverse mapping uses
      ``Φ(z) → [0,1] → interval lookup``.
    """

    def __init__(self, categorical_encoding: str = _ENCODING_TRUNCATED_GAUSSIAN):
        if categorical_encoding not in (_ENCODING_TRUNCATED_GAUSSIAN, _ENCODING_NORMAL_SCORES):
            raise ValueError(
                f"categorical_encoding must be one of "
                f"{_ENCODING_TRUNCATED_GAUSSIAN!r}, {_ENCODING_NORMAL_SCORES!r}; "
                f"got {categorical_encoding!r}."
            )
        self.categorical_encoding = categorical_encoding

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------

>>>>>>> troubleshooting
    @staticmethod
    def handle_missing_values(
        df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, Dict[str, Dict[str, str]]]:
<<<<<<< HEAD
        """
        Process columns with missing values using Polars expressions by:
        1. Creating a binary indicator column (Yes/No) for original missingness.
        2. Filling missing values in the original column using random samples
           from non-missing values.

        Args:
            df: Input Polars DataFrame.

        Returns:
            Tuple of:
            - Processed Polars DataFrame.
            - Dictionary mapping original columns to created columns.
        """
        processed_df = df.clone()  # Use clone() for Polars DataFrames
        column_mapping: Dict[str, Dict[str, str]] = {}
        indicator_expressions: List[pl.Expr] = []
        fill_expressions: List[pl.Expr] = []
        columns_to_fill: List[str] = []  # Keep track of columns needing filling

        # --- Pass 1: Identify columns and prepare expressions ---
        for column in df.columns:
            # Check if the column actually has nulls in the original DataFrame
            if df.select(pl.col(column).is_null().any()).item():
                logger.debug(
                    f"Column '{column}' has missing values. Preparing expressions."
                )
                indicator_col_name = f"{column}_missing_indicator"
                column_mapping[column] = {
                    "filled_column": column,
                    "indicator_column": indicator_col_name,
                }
                columns_to_fill.append(column)  # Mark for filling later

                # --- Indicator Expression ---
                # This mask refers to the state *before* any filling happens
                is_null_mask = pl.col(column).is_null()
                indicator_expr = (
                    pl.when(is_null_mask)
                    .then(
                        pl.lit("No")
                    )  # Originally missing -> "No" (value wasn't present)
                    .otherwise(pl.lit("Yes"))  # Originally present -> "Yes"
                    .alias(indicator_col_name)
                )
                indicator_expressions.append(indicator_expr)

                # --- Fill Expression Preparation (get samples) ---
                non_missing_values_series = df.get_column(column).drop_nulls()

                if len(non_missing_values_series) > 0:
                    n_rows = len(df)
                    dtype = df.get_column(column).dtype  # Get original dtype

                    # Sample using Polars for potential efficiency and type handling
                    # Ensure sampling happens correctly even if few non-nulls exist
                    if len(non_missing_values_series) < n_rows:
                        # Sample with replacement if not enough unique values
                        sampled_values = non_missing_values_series.sample(
                            n=n_rows, with_replacement=True, shuffle=True
                        )
                    else:
                        # Just shuffle if enough unique values (or more)
                        sampled_values = non_missing_values_series.sample(
                            n=n_rows, with_replacement=False, shuffle=True
                        )

                    # Ensure the sampled series has the correct dtype
                    sampled_values = sampled_values.cast(dtype)

                    fill_expr = (
                        pl.when(
                            is_null_mask
                        )  # Use the same mask based on original state
                        .then(sampled_values)  # Use the prepared sampled series
                        .otherwise(pl.col(column))  # Keep original non-null value
                        .alias(column)  # Overwrite the original column
                    )
                    fill_expressions.append(fill_expr)
                else:
                    # Handle columns that are *entirely* null (optional: log warning/error)
                    logger.warning(
                        f"Column '{column}' is entirely null. Cannot sample for filling."
                    )
                    # Decide behavior: leave as null, fill with a constant, or raise error?
                    # Option: Leave as null (no fill_expr added for this column)
                    # Option: Fill with a default null representation if appropriate
                    # fill_expr = pl.lit(None, dtype=df.get_column(column).dtype).alias(column)
                    # fill_expressions.append(fill_expr)
                    pass  # Current behavior: leaves the all-null column as is.

        # --- Pass 2: Apply Indicator Expressions ---
        if indicator_expressions:
            logger.debug(f"Adding {len(indicator_expressions)} indicator columns.")
            processed_df = processed_df.with_columns(indicator_expressions)

        # --- Pass 3: Apply Fill Expressions ---
        if fill_expressions:
            logger.debug(f"Applying filling logic for {len(fill_expressions)} columns.")
=======
        """Add binary indicator columns for missingness and impute with random draws."""
        processed_df = df.clone()
        column_mapping: Dict[str, Dict[str, str]] = {}
        indicator_expressions: List[pl.Expr] = []
        fill_expressions: List[pl.Expr] = []

        for column in df.columns:
            if not df.select(pl.col(column).is_null().any()).item():
                continue

            logger.debug(f"Column '{column}' has missing values.")
            indicator_col_name = f"{column}_missing_indicator"
            column_mapping[column] = {
                "filled_column": column,
                "indicator_column": indicator_col_name,
            }

            is_null_mask = pl.col(column).is_null()
            indicator_expressions.append(
                pl.when(is_null_mask)
                .then(pl.lit("No"))
                .otherwise(pl.lit("Yes"))
                .alias(indicator_col_name)
            )

            non_missing = df.get_column(column).drop_nulls()
            if len(non_missing) > 0:
                n_rows = len(df)
                dtype = df.get_column(column).dtype
                with_replacement = len(non_missing) < n_rows
                sampled = non_missing.sample(
                    n=n_rows, with_replacement=with_replacement, shuffle=True
                ).cast(dtype)
                fill_expressions.append(
                    pl.when(is_null_mask)
                    .then(sampled)
                    .otherwise(pl.col(column))
                    .alias(column)
                )
            else:
                logger.warning(f"Column '{column}' is entirely null; cannot impute.")

        if indicator_expressions:
            processed_df = processed_df.with_columns(indicator_expressions)
        if fill_expressions:
>>>>>>> troubleshooting
            processed_df = processed_df.with_columns(fill_expressions)

        return processed_df, column_mapping

<<<<<<< HEAD
    @staticmethod
    def _truncated_gaussian_sample(
        lower: float, upper: float, size: int = 1, random_state: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        mean = (lower + upper) / 2
        std_dev = (upper - lower) / 6

        a = lower
        b = upper

        samples = stats.truncnorm.rvs(loc=mean, scale=std_dev, a=a, b=b, size=size)

        samples = stats.uniform.rvs(loc=a, scale=b - a, size=size)

=======
    # ------------------------------------------------------------------
    # Categorical encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncated_gaussian_sample(
        lower: float, upper: float, size: int = 1
    ) -> Union[float, np.ndarray]:
        mean = (lower + upper) / 2.0
        std_dev = (upper - lower) / 4.0
        if std_dev == 0:
            result = np.full(size, mean, dtype=np.float64)
            return result[0] if size == 1 else result
        a_std = (lower - mean) / std_dev
        b_std = (upper - mean) / std_dev
        samples = stats.truncnorm.rvs(a=a_std, b=b_std, loc=mean, scale=std_dev, size=size)
>>>>>>> troubleshooting
        return samples[0] if size == 1 else samples

    def convert_categorical(
        self, series: pl.Series
    ) -> Tuple[pl.Series, Dict[str, Any]]:
<<<<<<< HEAD
        if not (series.dtype == pl.Utf8 or series.dtype == pl.Categorical):
=======
        """Encode a categorical Series as floats using the configured encoding strategy."""
        if series.dtype not in (pl.Utf8, pl.Categorical):
>>>>>>> troubleshooting
            raise TypeError(f"Expected Utf8 or Categorical Series, got {series.dtype}")

        value_counts_df = (
            series.value_counts()
            .sort("count", descending=True)
            .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
        )

        if value_counts_df.is_empty():
            return pl.Series(name=series.name, values=[], dtype=pl.Float64), {
                "categories": [],
                "frequencies": [],
                "intervals": [],
<<<<<<< HEAD
            }

        # Column name in value_counts is the original series name
=======
                "encoding": self.categorical_encoding,
            }

>>>>>>> troubleshooting
        cat_col_name = series.name
        categories = value_counts_df[cat_col_name].to_list()
        frequencies = value_counts_df["proportion"].to_list()

<<<<<<< HEAD
        # Calculate cumulative probabilities to define intervals [lower, upper)
        cum_probs = np.cumsum(frequencies)
        intervals = list(zip([0.0] + cum_probs[:-1].tolist(), cum_probs.tolist()))

        # Create mapping from categories to intervals
        cat_to_interval = dict(zip(categories, intervals))

        # --- Efficient sampling using vectorized operations ---
        # 1. Create mapping from category to interval bounds
        map_df = pl.DataFrame(
            {
                cat_col_name: categories,
                "lower": [i[0] for i in intervals],
                "upper": [i[1] for i in intervals],
            }
        )

        # 2. Join the original series with the mapping
        # Convert series to frame for join
        series_df = series.to_frame()
        # Ensure correct dtypes for join keys if one is Categorical
        if series.dtype == pl.Categorical:
            map_df = map_df.with_columns(pl.col(cat_col_name).cast(pl.Categorical))

        joined_df = series_df.join(map_df, on=cat_col_name, how="left")

        # 3. Generate all samples needed using NumPy/SciPy based on joined intervals
        # Need to handle potential nulls in the original series -> resulting nulls in join
        num_rows = len(joined_df)
        numerical_values_np = np.full(
            num_rows, np.nan, dtype=np.float64
        )  # Initialize with NaN

        # Generate samples for each unique interval present in the data
        # Group by interval to sample efficiently
        unique_intervals = joined_df.select(["lower", "upper"]).drop_nulls().unique()

        # Create a temporary random state for this function call if needed

        for row in unique_intervals.iter_rows(named=True):
            lower, upper = row["lower"], row["upper"]
            # Find rows matching this interval (handle potential nulls in interval columns)
            mask = (joined_df["lower"] == lower) & (joined_df["upper"] == upper)
            # Filter out nulls explicitly if they weren't dropped earlier
            mask = mask.fill_null(False)

            n_samples = (
                mask.sum()
            )  # Count how many samples are needed for this interval
            if n_samples > 0:
                samples = self._truncated_gaussian_sample(
                    lower,
                    upper,
                    size=n_samples,
                )
                # Place samples into the correct positions using the mask
                numerical_values_np[mask.to_numpy()] = samples

        # Store conversion parameters
        conversion_params = {
            "categories": categories,
            "frequencies": frequencies,  # Already a list
            "intervals": intervals,  # Already a list of tuples
            "dtype": str(series.dtype),  # Store original dtype (Utf8 or Categorical)
        }

        # 4. Create the final Polars Series
        numerical_series = pl.Series(
            name=series.name, values=numerical_values_np, dtype=pl.Float64
        )

        return numerical_series, conversion_params
=======
        cum_probs = np.cumsum(frequencies)
        intervals: list[tuple[float, float]] = list(
            zip([0.0] + cum_probs[:-1].tolist(), cum_probs.tolist())
        )

        if self.categorical_encoding == _ENCODING_NORMAL_SCORES:
            numerical_values_np = self._apply_normal_scores(series, categories, intervals)
        else:
            numerical_values_np = self._apply_truncated_gaussian(series, categories, intervals, cat_col_name)

        conversion_params = {
            "categories": categories,
            "frequencies": frequencies,
            "intervals": intervals,
            "encoding": self.categorical_encoding,
            "dtype": str(series.dtype),
        }

        return (
            pl.Series(name=series.name, values=numerical_values_np, dtype=pl.Float64),
            conversion_params,
        )

    @staticmethod
    def _apply_normal_scores(
        series: pl.Series,
        categories: list,
        intervals: list[tuple[float, float]],
    ) -> np.ndarray:
        """Map each category to Φ⁻¹(midpoint of its cumulative interval)."""
        eps = 1e-8
        cat_to_score = {
            cat: float(norm.ppf(np.clip((lo + hi) / 2.0, eps, 1.0 - eps)))
            for cat, (lo, hi) in zip(categories, intervals)
        }
        return np.array(
            [cat_to_score.get(v, np.nan) for v in series.to_list()], dtype=np.float64
        )

    @staticmethod
    def _apply_truncated_gaussian(
        series: pl.Series,
        categories: list,
        intervals: list[tuple[float, float]],
        cat_col_name: str,
    ) -> np.ndarray:
        """Original approach: random sample from truncated Gaussian in each interval."""
        map_df = pl.DataFrame({
            cat_col_name: categories,
            "lower": [i[0] for i in intervals],
            "upper": [i[1] for i in intervals],
        })
        series_df = series.to_frame()
        if series.dtype == pl.Categorical:
            map_df = map_df.with_columns(pl.col(cat_col_name).cast(pl.Categorical))
        joined_df = series_df.join(map_df, on=cat_col_name, how="left")

        num_rows = len(joined_df)
        numerical_values_np = np.full(num_rows, np.nan, dtype=np.float64)
        unique_intervals = joined_df.select(["lower", "upper"]).drop_nulls().unique()

        for row in unique_intervals.iter_rows(named=True):
            lower, upper = row["lower"], row["upper"]
            mask = ((joined_df["lower"] == lower) & (joined_df["upper"] == upper)).fill_null(False)
            n_samples = mask.sum()
            if n_samples > 0:
                samples = SDVPreprocessor._truncated_gaussian_sample(lower, upper, size=n_samples)
                numerical_values_np[mask.to_numpy()] = samples

        return numerical_values_np
>>>>>>> troubleshooting

    def convert_categorical_columns(
        self, df: pl.DataFrame, categorical_columns: Optional[list[str]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Dict[str, Any]]]:
<<<<<<< HEAD
        """
        Process all categorical (Utf8, Categorical) columns in the DataFrame using Polars.

        Args:
            df: Input Polars DataFrame.

        Returns:
            Tuple of:
            - Processed DataFrame with categorical columns converted to numerical (Float64).
            - Dictionary with conversion parameters for each converted column.
        """
        processed_df = df.clone()
        conversion_params: Dict[str, Dict[str, Any]] = {}
        columns_to_convert: List[str] = []

        # Identify categorical columns (Utf8 or Categorical)
        if categorical_columns is None:
            for col_name, dtype in df.schema.items():
                if dtype in [pl.Utf8, pl.Categorical]:
                    columns_to_convert.append(col_name)
        else:
            columns_to_convert = categorical_columns
        # Apply conversion column by column
        # While Polars prefers whole-dataframe ops, this conversion is stateful per column
        for column in columns_to_convert:
            if (
                column in processed_df.columns
            ):  # Check if column still exists (e.g., removed?)
                series = processed_df.get_column(column)
                numerical_series, params = self.convert_categorical(series)
                # Overwrite the original column with the numerical version
=======
        """Convert all categorical columns to float using the configured encoding."""
        processed_df = df.clone()
        conversion_params: Dict[str, Dict[str, Any]] = {}

        if categorical_columns is None:
            columns_to_convert = [
                col for col, dtype in df.schema.items()
                if dtype in (pl.Utf8, pl.Categorical)
            ]
        else:
            # Also include any Utf8/Categorical columns not in the explicit list
            # (e.g. auto-generated _missing_indicator columns from handle_missing_values)
            extra_string_cols = [
                col for col, dtype in df.schema.items()
                if dtype in (pl.Utf8, pl.Categorical) and col not in categorical_columns
            ]
            columns_to_convert = list(categorical_columns) + extra_string_cols

        for column in columns_to_convert:
            if column in processed_df.columns:
                series = processed_df.get_column(column)
                numerical_series, params = self.convert_categorical(series)
>>>>>>> troubleshooting
                processed_df = processed_df.with_columns(numerical_series)
                conversion_params[column] = params

        return processed_df, conversion_params

    @staticmethod
    def inverse_convert_numerical_to_categorical(
        numerical_series: pl.Series, conversion_params: Dict[str, Any]
    ) -> pl.Series:
<<<<<<< HEAD
        """
        Convert numerical Polars Series (range [0,1]) back to categorical
        using the stored parameters.

        Args:
            numerical_series: Numerical Polars Series (Float64).
            conversion_params: Parameters used for the original conversion.

        Returns:
            Polars Series with original categories (Utf8 or Categorical).
        """
        categories = conversion_params["categories"]
        intervals = conversion_params["intervals"]

        # Create interval to category mapping
        interval_to_cat = dict(zip(intervals, categories))
        # Sort intervals by lower bound to ensure correct mapping
        sorted_intervals = sorted(intervals, key=lambda x: x[0])

        # Function to map a numerical value back to category
        # This often requires .apply in Polars for complex conditional logic per element
        def numerical_to_category(val: Optional[float]) -> Optional[str]:
            if val is None or np.isnan(val):
                return None

            for (lower, upper), category in interval_to_cat.items():
                # Handle the edge case for the interval containing 1.0
                is_last_interval = upper == 1.0
                if (lower <= val < upper) or (
                    is_last_interval and lower <= val <= upper
                ):
                    return category

            # Fallback: if value is outside [0, 1] or doesn't fit, map to most frequent
            # This might indicate an issue with the input data or process
            # Check bounds explicitly
            if val < 0.0:
                return categories[0]

            if val > 1.0:
                return categories[-1]

        # Use apply for the mapping logic
        # Ensure return_dtype matches the target categorical type (Utf8 or Categorical)
        categorical_series = numerical_series.map_elements(
            numerical_to_category, return_dtype=pl.Utf8  # Map to Utf8 first
        )

        return categorical_series.alias(numerical_series.name)
=======
        """Reverse categorical encoding back to the original category labels.

        Handles both ``truncated_gaussian`` (value in [0, 1] → interval lookup)
        and ``normal_scores`` (value in ℝ → Φ(z) → interval lookup).
        """
        categories = conversion_params["categories"]
        intervals = conversion_params["intervals"]
        encoding = conversion_params.get("encoding", _ENCODING_TRUNCATED_GAUSSIAN)

        interval_to_cat = dict(zip([tuple(i) for i in intervals], categories))

        def _find_category(val: Optional[float]) -> Optional[str]:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None

            # For normal_scores, map from ℝ → [0, 1] via Φ before lookup
            if encoding == _ENCODING_NORMAL_SCORES:
                u = float(norm.cdf(val))
            else:
                u = float(val)

            for (lower, upper), category in interval_to_cat.items():
                is_last = upper >= 1.0
                if (lower <= u < upper) or (is_last and lower <= u <= upper):
                    return category

            # Fallback for out-of-range values
            if not categories:
                return None
            if u < 0.0:
                return categories[0]
            return categories[-1]

        return numerical_series.map_elements(
            _find_category, return_dtype=pl.Utf8
        ).alias(numerical_series.name)

    # ------------------------------------------------------------------
    # Unified pipeline
    # ------------------------------------------------------------------
>>>>>>> troubleshooting

    def preprocess(
        self, df: pl.DataFrame, categorical_columns: Optional[list[str]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
<<<<<<< HEAD
        """
        Apply full preprocessing pipeline to the Polars DataFrame:
        1. Handle missing values
        2. Convert categorical variables

        Args:
            df: Input Polars DataFrame.

        Returns:
            Tuple of:
            - Fully processed Polars DataFrame.
            - Dictionary with all transformation parameters.
        """
        # 1. Handle missing values first
=======
        """Full preprocessing: handle missing values, then encode categoricals."""
>>>>>>> troubleshooting
        logger.info("Handling missing values...")
        df_missing_handled, missing_mapping = self.handle_missing_values(df)
        logger.success("Missing values handled.")

<<<<<<< HEAD
        # 2. Then convert categorical columns (including indicator columns)
        logger.info("Converting categorical columns...")
        df_processed, categorical_params = self.convert_categorical_columns(
            df_missing_handled, categorical_columns=categorical_columns
        )

        logger.success("Categorical columns converted.")

        # Combine transformation parameters
        transformations = {
=======
        logger.info(f"Converting categorical columns (encoding={self.categorical_encoding!r})...")
        df_processed, categorical_params = self.convert_categorical_columns(
            df_missing_handled, categorical_columns=categorical_columns
        )
        logger.success("Categorical columns converted.")

        return df_processed, {
>>>>>>> troubleshooting
            "missing_mappings": missing_mapping,
            "categorical_params": categorical_params,
        }

<<<<<<< HEAD
        return df_processed, transformations

    def reverse_preprocessing(
        self, df: pl.DataFrame, transformations: Dict[str, Any]
    ) -> pl.DataFrame:
        """
        Revert preprocessing transformations on a Polars DataFrame to recover
        original data format (useful when working with synthesized data).

        Args:
            df: Processed Polars DataFrame.
            transformations: Transformation parameters from preprocessing.

        Returns:
            Polars DataFrame with original format.
        """
        reverted_df = df.clone()

        # 1. Revert categorical transformations first
=======
    def reverse_preprocessing(
        self, df: pl.DataFrame, transformations: Dict[str, Any]
    ) -> pl.DataFrame:
        """Revert preprocessing to recover original data format."""
        reverted_df = df.clone()

        # 1. Revert categorical encoding
>>>>>>> troubleshooting
        cat_params = transformations.get("categorical_params", {})
        categorical_exprs = []
        for column, params in cat_params.items():
            if column in reverted_df.columns:
<<<<<<< HEAD
                numerical_series = reverted_df.get_column(column)
                # Generate expression for reversion
                reverted_series = self.inverse_convert_numerical_to_categorical(
                    numerical_series, params
                )
                categorical_exprs.append(reverted_series)  # Will overwrite existing col

        if categorical_exprs:
            reverted_df = reverted_df.with_columns(categorical_exprs)

        # 2. Revert missing value transformations
=======
                reverted_series = self.inverse_convert_numerical_to_categorical(
                    reverted_df.get_column(column), params
                )
                categorical_exprs.append(reverted_series)
        if categorical_exprs:
            reverted_df = reverted_df.with_columns(categorical_exprs)

        # 2. Revert missing value imputation
>>>>>>> troubleshooting
        missing_mappings = transformations.get("missing_mappings", {})
        cols_to_drop = []
        missing_exprs = []
        for orig_col, mapping in missing_mappings.items():
            filled_col = mapping["filled_column"]
            indicator_col = mapping["indicator_column"]
<<<<<<< HEAD

            if (
                filled_col in reverted_df.columns
                and indicator_col in reverted_df.columns
            ):
                # Create expression to reintroduce nulls based on indicator
                reintroduce_null_expr = (
                    pl.when(pl.col(indicator_col) == "No")
                    .then(
                        pl.lit(None, dtype=reverted_df[filled_col].dtype)
                    )  # Use None literal with correct dtype
                    .otherwise(pl.col(filled_col))
                    .alias(filled_col)  # Overwrite the original filled column name
                )
                missing_exprs.append(reintroduce_null_expr)
                cols_to_drop.append(indicator_col)

        # Apply null reintroduction expressions
        if missing_exprs:
            reverted_df = reverted_df.with_columns(missing_exprs)

        # Drop the indicator columns at the end
        if cols_to_drop:
            # Ensure columns to drop actually exist before dropping
            cols_to_drop = [col for col in cols_to_drop if col in reverted_df.columns]
=======
            if filled_col in reverted_df.columns and indicator_col in reverted_df.columns:
                missing_exprs.append(
                    pl.when(pl.col(indicator_col) == "No")
                    .then(pl.lit(None, dtype=reverted_df[filled_col].dtype))
                    .otherwise(pl.col(filled_col))
                    .alias(filled_col)
                )
                cols_to_drop.append(indicator_col)

        if missing_exprs:
            reverted_df = reverted_df.with_columns(missing_exprs)
        if cols_to_drop:
            cols_to_drop = [c for c in cols_to_drop if c in reverted_df.columns]
>>>>>>> troubleshooting
            if cols_to_drop:
                reverted_df = reverted_df.drop(cols_to_drop)

        return reverted_df

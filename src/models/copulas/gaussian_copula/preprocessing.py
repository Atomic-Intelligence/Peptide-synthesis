import random
from typing import Tuple, Dict, List, Optional, Union, Any

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import norm, rankdata

from src.logger import setup_logger

logger = setup_logger()

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
        if categorical_encoding not in (
            _ENCODING_TRUNCATED_GAUSSIAN,
            _ENCODING_NORMAL_SCORES,
        ):
            raise ValueError(
                f"categorical_encoding must be one of "
                f"{_ENCODING_TRUNCATED_GAUSSIAN!r}, {_ENCODING_NORMAL_SCORES!r}; "
                f"got {categorical_encoding!r}."
            )
        self.categorical_encoding = categorical_encoding

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------

    @staticmethod
    def handle_missing_values(
        df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, Dict[str, Dict[str, str]]]:
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
            processed_df = processed_df.with_columns(fill_expressions)

        return processed_df, column_mapping

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
        samples = stats.truncnorm.rvs(
            a=a_std, b=b_std, loc=mean, scale=std_dev, size=size
        )
        return samples[0] if size == 1 else samples

    def convert_categorical(
        self, series: pl.Series
    ) -> Tuple[pl.Series, Dict[str, Any]]:
        """Encode a categorical Series as floats using the configured encoding strategy."""
        if series.dtype not in (pl.Utf8, pl.Categorical):
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
                "encoding": self.categorical_encoding,
            }

        cat_col_name = series.name
        categories = value_counts_df[cat_col_name].to_list()
        frequencies = value_counts_df["proportion"].to_list()

        cum_probs = np.cumsum(frequencies)
        intervals: list[tuple[float, float]] = list(
            zip([0.0] + cum_probs[:-1].tolist(), cum_probs.tolist())
        )

        if self.categorical_encoding == _ENCODING_NORMAL_SCORES:
            numerical_values_np = self._apply_normal_scores(
                series, categories, intervals
            )
        else:
            numerical_values_np = self._apply_truncated_gaussian(
                series, categories, intervals, cat_col_name
            )

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
        map_df = pl.DataFrame(
            {
                cat_col_name: categories,
                "lower": [i[0] for i in intervals],
                "upper": [i[1] for i in intervals],
            }
        )
        series_df = series.to_frame()
        if series.dtype == pl.Categorical:
            map_df = map_df.with_columns(pl.col(cat_col_name).cast(pl.Categorical))
        joined_df = series_df.join(map_df, on=cat_col_name, how="left")

        num_rows = len(joined_df)
        numerical_values_np = np.full(num_rows, np.nan, dtype=np.float64)
        unique_intervals = joined_df.select(["lower", "upper"]).drop_nulls().unique()

        for row in unique_intervals.iter_rows(named=True):
            lower, upper = row["lower"], row["upper"]
            mask = (
                (joined_df["lower"] == lower) & (joined_df["upper"] == upper)
            ).fill_null(False)
            n_samples = mask.sum()
            if n_samples > 0:
                samples = SDVPreprocessor._truncated_gaussian_sample(
                    lower, upper, size=n_samples
                )
                numerical_values_np[mask.to_numpy()] = samples

        return numerical_values_np

    def convert_categorical_columns(
        self, df: pl.DataFrame, categorical_columns: Optional[list[str]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Dict[str, Any]]]:
        """Convert all categorical columns to float using the configured encoding."""
        processed_df = df.clone()
        conversion_params: Dict[str, Dict[str, Any]] = {}

        if categorical_columns is None:
            columns_to_convert = [
                col
                for col, dtype in df.schema.items()
                if dtype in (pl.Utf8, pl.Categorical)
            ]
        else:
            # Also include any Utf8/Categorical columns not in the explicit list
            # (e.g. auto-generated _missing_indicator columns from handle_missing_values)
            extra_string_cols = [
                col
                for col, dtype in df.schema.items()
                if dtype in (pl.Utf8, pl.Categorical) and col not in categorical_columns
            ]
            columns_to_convert = list(categorical_columns) + extra_string_cols

        for column in columns_to_convert:
            if column in processed_df.columns:
                series = processed_df.get_column(column)
                numerical_series, params = self.convert_categorical(series)
                processed_df = processed_df.with_columns(numerical_series)
                conversion_params[column] = params

        return processed_df, conversion_params

    @staticmethod
    def inverse_convert_numerical_to_categorical(
        numerical_series: pl.Series, conversion_params: Dict[str, Any]
    ) -> pl.Series:
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

    def preprocess(
        self, df: pl.DataFrame, categorical_columns: Optional[list[str]] = None
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """Full preprocessing: handle missing values, then encode categoricals."""
        logger.info("Handling missing values...")
        df_missing_handled, missing_mapping = self.handle_missing_values(df)
        logger.success("Missing values handled.")

        logger.info(
            f"Converting categorical columns (encoding={self.categorical_encoding!r})..."
        )
        df_processed, categorical_params = self.convert_categorical_columns(
            df_missing_handled, categorical_columns=categorical_columns
        )
        logger.success("Categorical columns converted.")

        return df_processed, {
            "missing_mappings": missing_mapping,
            "categorical_params": categorical_params,
        }

    def reverse_preprocessing(
        self, df: pl.DataFrame, transformations: Dict[str, Any]
    ) -> pl.DataFrame:
        """Revert preprocessing to recover original data format."""
        reverted_df = df.clone()

        # 1. Revert categorical encoding
        cat_params = transformations.get("categorical_params", {})
        categorical_exprs = []
        for column, params in cat_params.items():
            if column in reverted_df.columns:
                reverted_series = self.inverse_convert_numerical_to_categorical(
                    reverted_df.get_column(column), params
                )
                categorical_exprs.append(reverted_series)
        if categorical_exprs:
            reverted_df = reverted_df.with_columns(categorical_exprs)

        # 2. Revert missing value imputation
        missing_mappings = transformations.get("missing_mappings", {})
        cols_to_drop = []
        missing_exprs = []
        for orig_col, mapping in missing_mappings.items():
            filled_col = mapping["filled_column"]
            indicator_col = mapping["indicator_column"]
            if (
                filled_col in reverted_df.columns
                and indicator_col in reverted_df.columns
            ):
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
            if cols_to_drop:
                reverted_df = reverted_df.drop(cols_to_drop)

        return reverted_df

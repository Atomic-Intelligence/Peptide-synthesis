# Standard library imports
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# Third-party imports
import mlflow
import numpy as np
import polars as pl
import polars.selectors as cs
import statsmodels.stats.correlation_tools
from polars import DataFrame
from scipy import stats
from scipy.stats import rankdata
from statsmodels.distributions.copula.copulas import CopulaDistribution
from statsmodels.distributions.copula.elliptical import GaussianCopula, StudentTCopula

from src.logger import setup_logger
from src.models.copulas.gaussian_copula.data_models import ModelSignature
from src.models.copulas.gaussian_copula.preprocessing import SDVPreprocessor
from src.models.copulas.marginal_distributions.marginal_distribution_estimator import (
    MarginalDistributionEstimator,
)
from src.models.copulas.marginal_distributions.marginal_distribution_metrics import (
    MockUnivariateDistributionMetric,
)
from src.models.synthetization_model_interface import (
    SynthetizationModelInterface,
    DatasetMetadata,
    MlFlowTrainingRunInfo,
)


logger = setup_logger()

MODEL_ARTIFACT_DIR = "model"
SIGNATURE_FILENAME = "signature.json"


class ModelNotFittedError(RuntimeError):
    def __init__(self, message="Model must be fitted before this operation."):
        super().__init__(message)


class GaussianCopulaBasic(SynthetizationModelInterface):
    """Gaussian (or Student-T) copula model with configurable correlation estimation
    and post-generation constraints.

    Key improvements over the baseline:
    - **1.1** Correlation is estimated using fitted marginal CDFs (not a disconnected
      QuantileTransformer), making it theoretically consistent with CopulaDistribution.
    - **3.2** Rank-based correlation methods (kendall, spearman) available as alternatives
      to Pearson; rank-based methods handle non-linear monotone relationships better.
    - **1.2** Student-T degrees-of-freedom estimated from data instead of being hardcoded.
    - **1.4** Domain-aware post-generation constraints (e.g., clip peptides to [0, ∞)).
    - **2.1** Normal-scores categorical encoding (deterministic alternative to truncated-Gaussian).
    """

    def __init__(
        self,
        marginal_distribution_estimator: MarginalDistributionEstimator,
        sdv_preprocessor: SDVPreprocessor,
        ml_flow_info: MlFlowTrainingRunInfo,
        copula_type: str = "gaussian",
        corr_method: str = "pearson",
        student_t_df: Optional[float] = None,
        student_t_df_grid: Optional[list] = None,
        peptide_non_negative: bool = True,
        clip_columns: Optional[dict] = None,
        categorical_columns: Optional[list[str]] = None,
    ):
        """
        Args:
            marginal_distribution_estimator: Fits per-column marginal distributions.
            sdv_preprocessor: Handles missing value and categorical preprocessing.
            ml_flow_info: MLflow experiment / run metadata.
            copula_type: ``"gaussian"`` or ``"student_t"``.
            corr_method: Correlation estimation method.
                - ``"pearson"``  — Pearson correlation on marginal-CDF-transformed data
                  (theoretically consistent with the copula, fix 1.1).
                - ``"kendall"``  — Kendall τ converted to Gaussian copula parameter via
                  sin(π/2·τ).  Rank-based; robust to outliers and non-linear monotone
                  relationships (improvement 3.2).
                - ``"spearman"`` — Spearman ρ converted via 2·sin(π/6·ρ).  Rank-based;
                  faster than Kendall for large datasets (improvement 3.2).
            student_t_df: Degrees of freedom for Student-T copula.  ``None`` means
                estimate automatically from data (improvement 1.2); a positive float
                fixes the value.
            student_t_df_grid: Grid of df values to search when ``student_t_df=None``.
                Defaults to ``[2, 3, 5, 10, 20, 50]``.
            peptide_non_negative: If True, clip all columns matching ``(?i)peptide`` to
                ``[0, +∞)`` after generation (improvement 1.4).
            clip_columns: Dict mapping regex patterns to ``[min, max]`` bounds applied
                after generation.  E.g. ``{"Age": [0, 120], "BMI": [10, 70]}``.
            categorical_columns: Column names that should be treated as categorical.
        """
        self.sdv_preprocessor = sdv_preprocessor
        self.preprocessing_transformations: Optional[dict] = None
        self.marginal_distribution_estimator = marginal_distribution_estimator
        self.copula_type = copula_type
        self.corr_method = corr_method
        self.student_t_df = student_t_df
        self.student_t_df_grid = student_t_df_grid or [2, 3, 5, 10, 20, 50]
        self.peptide_non_negative = peptide_non_negative
        self.clip_columns = clip_columns or {}
        self.copula: Optional[CopulaDistribution] = None
        self._model_signature_to_save: Optional[ModelSignature] = None
        self.column_names: Optional[list[str]] = None
        self.categorical_columns = categorical_columns
        super().__init__(ml_flow_info)

    # ------------------------------------------------------------------
    # Static preprocessing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def convert_zeroes_to_nulls(data: DataFrame) -> DataFrame:
        target_col_names = data.select(cs.numeric() & cs.matches("(?i)peptide")).columns
        if not target_col_names:
            return data
        update_expressions = [
            pl.when(pl.col(col) == 0)
            .then(pl.lit(None))
            .otherwise(pl.col(col))
            .alias(col)
            for col in target_col_names
        ]
        return data.with_columns(update_expressions)

    @staticmethod
    def convert_int_to_categorical(
        data: DataFrame, categorical_columns: Optional[list[str]] = None
    ) -> DataFrame:
        """Cast explicitly listed integer columns to string for categorical treatment."""
        if not categorical_columns:
            return data
        categorical_set = set(categorical_columns)
        int_cols = [
            name
            for name, dtype in zip(data.columns, data.dtypes)
            if dtype == pl.Int64 and name in categorical_set
        ]
        if int_cols:
            data = data.with_columns([pl.col(col).cast(pl.Utf8) for col in int_cols])
        return data

    # ------------------------------------------------------------------
    # Copula space transformation (improvement 1.1)
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_data_to_copula_space(
        X: np.ndarray, marginals: list, eps: float = 1e-8
    ) -> np.ndarray:
        """Transform preprocessed data to Gaussian copula space via fitted marginal CDFs.

        This replaces the QuantileTransformer approach and ensures the correlation
        matrix is consistent with the marginals used in CopulaDistribution (fix 1.1).

        For each column j:
            U_j = F_j(X_j)         # CDF of the fitted marginal → uniform space
            Z_j = Φ⁻¹(U_j)        # probit transform → standard normal space
        """
        Z = np.empty_like(X, dtype=float)
        for j, marginal in enumerate(marginals):
            col = X[:, j]
            u = marginal.cdf(col)
            u = np.clip(u, eps, 1.0 - eps)
            Z[:, j] = stats.norm.ppf(u)
        return Z

    # ------------------------------------------------------------------
    # Correlation estimation (improvements 1.1 + 3.2)
    # ------------------------------------------------------------------

    def _estimate_correlation_matrix(
        self, X: np.ndarray, marginals: list
    ) -> np.ndarray:
        """Estimate the copula correlation matrix using the configured method.

        - ``pearson``:  Pearson correlation on marginal-CDF-transformed data.
                        Theoretically consistent with CopulaDistribution (fix 1.1).
        - ``kendall``:  Kendall τ → sin(π/2·τ).  Rank-based, handles non-linear
                        monotone dependence (improvement 3.2).
        - ``spearman``: Spearman ρ → 2·sin(π/6·ρ).  Rank-based, faster than Kendall.
        """
        match self.corr_method:
            case "pearson":
                Z = self._transform_data_to_copula_space(X, marginals)
                return np.corrcoef(Z, rowvar=False)

            case "kendall":
                n_cols = X.shape[1]
                if n_cols > 50:
                    logger.warning(
                        f"Kendall tau on {n_cols} columns is O(d²·n·log n) — this may be slow."
                    )
                tau = np.eye(n_cols)
                for i in range(n_cols):
                    for j in range(i + 1, n_cols):
                        t, _ = stats.kendalltau(X[:, i], X[:, j])
                        val = t if np.isfinite(t) else 0.0
                        tau[i, j] = tau[j, i] = val
                return np.sin(np.pi / 2.0 * tau)

            case "spearman":
                # Rank each column then compute Pearson on ranks (= Spearman rho)
                ranks = np.apply_along_axis(rankdata, 0, X)
                rho = np.corrcoef(ranks, rowvar=False)
                # Convert Spearman ρ to Gaussian copula parameter
                return 2.0 * np.sin(np.pi / 6.0 * rho)

            case _:
                raise ValueError(
                    f"Unknown corr_method '{self.corr_method}'. "
                    "Choose from: pearson, kendall, spearman."
                )

    # ------------------------------------------------------------------
    # Student-T df estimation (improvement 1.2)
    # ------------------------------------------------------------------

    def _estimate_student_t_df(self, Z: np.ndarray) -> float:
        """Estimate Student-T degrees of freedom from Gaussian-transformed data.

        Fits ``stats.t`` independently to each column of Z (already in Gaussian
        copula space), then takes the median df across columns.  This is fast,
        avoids expensive multivariate computations, and gives a robust estimate.
        """
        dfs = []
        for j in range(Z.shape[1]):
            try:
                df_fit, _, _ = stats.t.fit(Z[:, j], floc=0)
                if np.isfinite(df_fit) and df_fit > 2.0:
                    dfs.append(df_fit)
            except Exception:
                continue

        if not dfs:
            logger.warning(
                "Could not estimate Student-T df from data; defaulting to df=5."
            )
            return 5.0

        estimated = float(np.median(dfs))
        # Clip to a sensible range
        estimated = float(np.clip(estimated, 2.0, 100.0))
        logger.info(
            f"Estimated Student-T df = {estimated:.2f} (from {len(dfs)} columns)"
        )
        return estimated

    # ------------------------------------------------------------------
    # Post-generation domain constraints (improvement 1.4)
    # ------------------------------------------------------------------

    def _apply_post_generation_constraints(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply domain-aware constraints to generated data (improvement 1.4).

        - Clips columns matching ``(?i)peptide`` to [0, +∞) if ``peptide_non_negative=True``.
        - Applies user-specified ``clip_columns`` patterns.
        """
        expressions = []

        if self.peptide_non_negative:
            peptide_cols = [
                c
                for c in df.columns
                if re.search(r"(?i)peptide", c)
                and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]
            for col in peptide_cols:
                expressions.append(pl.col(col).clip(lower_bound=0.0).alias(col))

        for pattern, bounds in self.clip_columns.items():
            lo, hi = bounds[0], bounds[1]
            matched = [
                c
                for c in df.columns
                if re.search(pattern, c)
                and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            ]
            for col in matched:
                expressions.append(
                    pl.col(col).clip(lower_bound=lo, upper_bound=hi).alias(col)
                )

        if expressions:
            df = df.with_columns(expressions)

        return df

    # ------------------------------------------------------------------
    # MLflow persistence
    # ------------------------------------------------------------------

    @classmethod
    def read_model_signature(cls, run_id: str, temp_dir: str) -> ModelSignature:
        artifact_rel_path = Path(MODEL_ARTIFACT_DIR) / SIGNATURE_FILENAME
        local_artifact_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=str(artifact_rel_path),
                dst_path=temp_dir,
            )
        )
        with open(local_artifact_path, "r") as f:
            model_signature = ModelSignature.model_validate_json(f.read())
        shutil.rmtree(temp_dir, ignore_errors=True)
        return model_signature

    @classmethod
    def _load_pretrained_from_mlflow_run(
        cls, run_id: str, ml_flow_info: MlFlowTrainingRunInfo
    ) -> "GaussianCopulaBasic":
        temp_dir = tempfile.mkdtemp()
        try:
            model_signature = cls.read_model_signature(run_id=run_id, temp_dir=temp_dir)
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to download or read model artifact: {e}") from e
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        model = cls(
            marginal_distribution_estimator=MarginalDistributionEstimator(
                continuous_distributions=[
                    getattr(stats, m.distribution_name)
                    for m in model_signature.marginals
                ],
                univariate_distribution_metric=MockUnivariateDistributionMetric(),
            ),
            sdv_preprocessor=SDVPreprocessor(),
            ml_flow_info=ml_flow_info,
        )

        marginals = []
        for m in model_signature.marginals:
            try:
                dist_cls = getattr(stats, m.distribution_name)
            except AttributeError:
                from src.models.copulas.marginal_distributions.custom_distributions import (
                    CUSTOM_DISTRIBUTION_REGISTRY,
                )

                dist_cls = CUSTOM_DISTRIBUTION_REGISTRY.get(m.distribution_name)
                if dist_cls is None:
                    raise ValueError(
                        f"Unknown distribution '{m.distribution_name}' — not in scipy.stats "
                        f"or CUSTOM_DISTRIBUTION_REGISTRY."
                    )
            marginals.append(dist_cls(*m.parameters))

        corr = np.array(model_signature.correlation_matrix)
        fitted_cop = GaussianCopula(corr=corr)
        model.copula = CopulaDistribution(copula=fitted_cop, marginals=marginals)
        model.preprocessing_transformations = model_signature.preprocessing_info
        model.column_names = model_signature.column_names
        model.fitted = True

        logger.success("Successfully loaded and reconstructed model from MLflow.")
        return model

    def log_model(self) -> None:
        if self._model_signature_to_save is None:
            raise ModelNotFittedError("Model must be fitted before logging.")

        temp_dir = tempfile.mkdtemp()
        try:
            temp_file_path = Path(temp_dir) / SIGNATURE_FILENAME
            logger.info(f"Saving model signature to temporary file: {temp_file_path}")
            with open(temp_file_path, "w") as f:
                f.write(self._model_signature_to_save.model_dump_json(indent=2))
            mlflow.log_artifact(str(temp_file_path), artifact_path=MODEL_ARTIFACT_DIR)
            logger.success(
                f"Successfully logged model signature to MLflow: "
                f"{MODEL_ARTIFACT_DIR}/{SIGNATURE_FILENAME}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to log model artifact to MLflow: {e}") from e
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Core fit / generate
    # ------------------------------------------------------------------

    def _fit(
        self,
        real_dataset: DataFrame,
        dataset_metadata: Optional[DatasetMetadata] = None,
    ) -> None:
        logger.info("Starting model fitting process...")

        # --- Preprocessing ---
        real_dataset = self.convert_zeroes_to_nulls(real_dataset)
        real_dataset = self.convert_int_to_categorical(
            real_dataset, self.categorical_columns
        )

        preprocessed_dataset_pl, transformations = self.sdv_preprocessor.preprocess(
            real_dataset, categorical_columns=self.categorical_columns
        )
        self.preprocessing_transformations = transformations
        self.column_names = preprocessed_dataset_pl.columns
        logger.success("Preprocessing complete.")

        # --- Marginal fitting ---
        logger.info("Fitting marginal distributions...")
        fitted_marginals = (
            self.marginal_distribution_estimator.fit_marginal_distributions(
                preprocessed_dataset_pl,
                categorical_columns=self.categorical_columns or [],
            )
        )

        marginal_info_list = []
        reconstructed_marginals = []
        for est in fitted_marginals:
            marginal_info_list.append(est.marginal_distribution_info)
            reconstructed_marginals.append(
                est.distribution(*est.marginal_distribution_info.parameters)
            )
        logger.success("Marginal distribution fitting complete.")

        # --- Correlation estimation (improvements 1.1 + 3.2) ---
        logger.info(f"Estimating correlation matrix (method={self.corr_method!r})...")
        X = preprocessed_dataset_pl.to_numpy()
        estimated_correlation = self._estimate_correlation_matrix(
            X, reconstructed_marginals
        )

        logger.info("Projecting to nearest valid correlation matrix...")
        estimated_correlation = statsmodels.stats.correlation_tools.corr_clipped(
            estimated_correlation, threshold=1e-5
        )

        if np.isnan(estimated_correlation).any():
            logger.warning("NaN values remain in the estimated correlation matrix.")

        # --- Build copula ---
        match self.copula_type:
            case "gaussian":
                fitted_cop = GaussianCopula(
                    corr=estimated_correlation, allow_singular=True
                )
                logger.success("Gaussian copula built.")

            case "student_t":
                if self.student_t_df is None:
                    # Estimate df from data (improvement 1.2)
                    Z = self._transform_data_to_copula_space(X, reconstructed_marginals)
                    effective_df = self._estimate_student_t_df(Z)
                else:
                    effective_df = float(self.student_t_df)
                logger.info(f"Building Student-T copula with df={effective_df:.2f}")
                fitted_cop = StudentTCopula(corr=estimated_correlation, df=effective_df)

            case _:
                raise ValueError(f"Unknown copula_type: {self.copula_type!r}")

        # --- Save model signature ---
        self._model_signature_to_save = ModelSignature(
            correlation_matrix=estimated_correlation.tolist(),
            marginals=marginal_info_list,
            preprocessing_info=self.preprocessing_transformations,
            column_names=self.column_names,
        )

        self.copula = CopulaDistribution(
            copula=fitted_cop, marginals=reconstructed_marginals
        )
        self._corr_matrix = estimated_correlation
        self._fitted_marginals = reconstructed_marginals
        self.fitted = True
        logger.success("Model fitting complete.")

    def _generate(self, n_synthetic_samples: int) -> pl.DataFrame:
        if not self.fitted:
            raise ModelNotFittedError()

        logger.info(f"Generating {n_synthetic_samples} synthetic samples...")
        generated_np = self.copula.rvs(nobs=n_synthetic_samples)
        logger.success(f"Generated synthetic data with shape {generated_np.shape}")

        generated_df = pl.DataFrame(generated_np, schema=self.column_names)

        logger.info("Applying reverse preprocessing transformations...")
        reversed_df = self.sdv_preprocessor.reverse_preprocessing(
            generated_df, self.preprocessing_transformations
        )

        # Replace remaining nulls with 0 then apply domain constraints (improvement 1.4)
        reversed_df = reversed_df.fill_null(0.0)
        reversed_df = self._apply_post_generation_constraints(reversed_df)

        logger.success("Synthetic data generation complete.")
        return reversed_df

    def generate_conditional(
        self,
        conditions: dict,
        n_samples: int,
    ) -> pl.DataFrame:
        """Generate samples conditioned on fixed values for a subset of columns.

        Uses the exact Gaussian copula conditional distribution via the Schur
        complement of the correlation matrix.  Only supported for
        ``copula_type='gaussian'``.

        Args:
            conditions: Mapping ``{column_name: value}`` in the **original** data
                space (strings for categorical columns, numbers for numeric).
                Column names must exist in ``self.column_names`` (preprocessed
                space).  Categorical columns are looked up via stored preprocessing
                params to find the encoded float value.
            n_samples: Number of synthetic rows to generate.

        Returns:
            pl.DataFrame with ``n_samples`` rows.  Conditioned columns will have
            values very close to the specified conditions (up to floating-point
            round-trips through CDF / PPF).

        Raises:
            ModelNotFittedError: if called before fitting.
            NotImplementedError: if ``copula_type`` is not ``'gaussian'``.
            ValueError: if a condition column is not found in the model.
        """
        if not self.fitted:
            raise ModelNotFittedError()
        if self.copula_type != "gaussian":
            raise NotImplementedError(
                "generate_conditional is only supported for copula_type='gaussian'."
            )

        all_cols = self.column_names
        col_to_idx = {c: i for i, c in enumerate(all_cols)}
        marginals = self._fitted_marginals

        obs_indices: list[int] = []
        obs_z_values: list[float] = []

        cat_params = self.preprocessing_transformations.get("categorical_params", {})

        for col_name, raw_value in conditions.items():
            if col_name not in col_to_idx:
                raise ValueError(
                    f"Condition column '{col_name}' not found in model columns. "
                    f"Available: {all_cols}"
                )
            idx = col_to_idx[col_name]

            # Transform raw value to preprocessed float
            if col_name in cat_params:
                encoding = cat_params[col_name].get("encoding", "truncated_gaussian")
                categories = cat_params[col_name]["categories"]
                intervals = cat_params[col_name]["intervals"]
                if raw_value not in categories:
                    raise ValueError(
                        f"Value {raw_value!r} is not a known category for '{col_name}'. "
                        f"Known: {categories}"
                    )
                cat_idx = categories.index(raw_value)
                lo, hi = intervals[cat_idx]
                mid = (lo + hi) / 2.0
                if encoding == "normal_scores":
                    preprocessed_value = float(
                        stats.norm.ppf(np.clip(mid, 1e-8, 1 - 1e-8))
                    )
                else:
                    preprocessed_value = float(mid)
            else:
                preprocessed_value = float(raw_value)

            # Transform to Gaussian copula space
            u = float(np.clip(marginals[idx].cdf(preprocessed_value), 1e-8, 1 - 1e-8))
            z = float(stats.norm.ppf(u))

            obs_indices.append(idx)
            obs_z_values.append(z)

        unobs_indices = [i for i in range(len(all_cols)) if i not in set(obs_indices)]

        # Partition correlation matrix
        corr = self._corr_matrix
        obs = np.array(obs_indices)
        unobs = np.array(unobs_indices)
        z1 = np.array(obs_z_values)

        Sigma_11 = corr[np.ix_(obs, obs)]
        Sigma_12 = corr[np.ix_(obs, unobs)]
        Sigma_21 = corr[np.ix_(unobs, obs)]
        Sigma_22 = corr[np.ix_(unobs, unobs)]

        # Conditional mean and covariance (Schur complement)
        try:
            Sigma_11_inv = np.linalg.inv(Sigma_11 + 1e-8 * np.eye(len(obs)))
        except np.linalg.LinAlgError:
            Sigma_11_inv = np.linalg.pinv(Sigma_11)

        mu_2_given_1 = Sigma_21 @ Sigma_11_inv @ z1
        Sigma_2_given_1 = Sigma_22 - Sigma_21 @ Sigma_11_inv @ Sigma_12

        # Ensure PSD
        Sigma_2_given_1 = statsmodels.stats.correlation_tools.corr_clipped(
            Sigma_2_given_1, threshold=1e-5
        )

        # Sample from conditional normal
        Z2_samples = np.random.multivariate_normal(
            mean=mu_2_given_1, cov=Sigma_2_given_1, size=n_samples
        )

        # Reconstruct full Z matrix
        Z_full = np.empty((n_samples, len(all_cols)))
        Z_full[:, unobs] = Z2_samples
        for k, z_val in zip(obs_indices, obs_z_values):
            Z_full[:, k] = z_val

        # Transform back through marginal PPFs
        X_full = np.empty_like(Z_full)
        for j, marginal in enumerate(marginals):
            u_j = np.clip(stats.norm.cdf(Z_full[:, j]), 1e-8, 1 - 1e-8)
            X_full[:, j] = marginal.ppf(u_j)

        generated_df = pl.DataFrame(X_full, schema=all_cols)
        reversed_df = self.sdv_preprocessor.reverse_preprocessing(
            generated_df, self.preprocessing_transformations
        )
        reversed_df = reversed_df.fill_null(0.0)
        reversed_df = self._apply_post_generation_constraints(reversed_df)

        logger.success(
            f"Conditional generation complete: {n_samples} samples with "
            f"{len(obs_indices)} fixed column(s)."
        )
        return reversed_df

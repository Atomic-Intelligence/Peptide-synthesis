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
from statsmodels.distributions.copula.copulas import CopulaDistribution
from statsmodels.distributions.copula.elliptical import GaussianCopula, StudentTCopula

from src.logger import setup_logger
from src.models.copulas.gaussian_copula.data_models import (
    ModelSignature,
)
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
from sklearn.preprocessing import QuantileTransformer


logger = setup_logger()

# --- Constants ---
MODEL_ARTIFACT_DIR = "model"
SIGNATURE_FILENAME = "signature.json"


class ModelNotFittedError(RuntimeError):
    def __init__(self, message="Model must be fitted before this operation."):
        super().__init__(message)


class GaussianCopulaBasic(SynthetizationModelInterface):
    def __init__(
        self,
        marginal_distribution_estimator: MarginalDistributionEstimator,
        sdv_preprocessor: SDVPreprocessor,
        ml_flow_info: MlFlowTrainingRunInfo,
        copula_type: str = "gaussian",
<<<<<<< HEAD
=======
        corr_method: str = "pearson",
        student_t_df: Optional[float] = None,
        student_t_df_grid: Optional[list] = None,
        peptide_non_negative: bool = True,
        clip_columns: Optional[dict] = None,
>>>>>>> troubleshooting
        categorical_columns: Optional[list[str]] = None,
    ):
        self.sdv_preprocessor = sdv_preprocessor
        self.preprocessing_transformations: Optional[dict] = None

        self.marginal_distribution_estimator = marginal_distribution_estimator

        self.copula_type = copula_type
<<<<<<< HEAD
=======
        self.corr_method = corr_method
        self.student_t_df = student_t_df
        self.student_t_df_grid = student_t_df_grid or [2, 3, 5, 10, 20, 50]
        self.peptide_non_negative = peptide_non_negative
        self.clip_columns = clip_columns or {}
>>>>>>> troubleshooting

        self.copula: Optional[CopulaDistribution] = None
        self._model_signature_to_save: Optional[ModelSignature] = None

        self.column_names: Optional[list[str]] = None
        self.categorical_columns = categorical_columns

        super().__init__(ml_flow_info)

    @staticmethod
    def convert_zeroes_to_nulls(data: DataFrame) -> DataFrame:
        target_col_names = data.select(cs.numeric() & cs.matches("Peptide")).columns

        update_expressions = []
        if target_col_names:  # Proceed only if there are matching columns
            update_expressions = [
                pl.when(pl.col(col_name) == 0)
                .then(
                    pl.lit(None)
                )  # Using pl.lit(None) is slightly more explicit than just None
                .otherwise(pl.col(col_name))
                .alias(
                    col_name
                )  # Explicitly ensure the output column name matches the input
                for col_name in target_col_names
            ]

        if update_expressions:
            data = data.with_columns(update_expressions)

        return data

    @staticmethod
    def convert_int_to_categorical(data: DataFrame) -> DataFrame:
        int_cols = [
            name for name, dtype in zip(data.columns, data.dtypes) if dtype == pl.Int64
        ]

        # Cast all integer columns to boolean
        data = data.with_columns(
            [pl.col(col_name).cast(pl.Utf8) for col_name in int_cols]
        )

        return data

    @classmethod
    def read_model_signature(cls, run_id: str, temp_dir: str):
        artifact_rel_path = Path(MODEL_ARTIFACT_DIR) / SIGNATURE_FILENAME
        local_artifact_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=str(artifact_rel_path),  # Ensure path is string
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
        # Download the model signature artifact
        temp_dir = tempfile.mkdtemp()

        try:
            model_signature = cls.read_model_signature(run_id=run_id, temp_dir=temp_dir)
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)  # Ensure cleanup on error
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
            ml_flow_info=MlFlowTrainingRunInfo(
                experiment_name=model_signature.experiment_name,
                run_name=model_signature.run_name,
            ),
        )

        marginals = []
        for marginal_info in model_signature.marginals:
            dist_class = getattr(stats, marginal_info.distribution_name)

            # Create a frozen distribution instance with loaded parameters
            dist_instance = dist_class(*marginal_info.parameters)
            marginals.append(dist_instance)

        correlation_matrix = np.array(model_signature.correlation_matrix)
        fitted_cop = GaussianCopula(corr=correlation_matrix)
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
                f"Successfully logged model signature artifact to MLflow path: "
                f"{MODEL_ARTIFACT_DIR}/{SIGNATURE_FILENAME}"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to log model artifact to MLflow: {e}") from e
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _generate(self, n_synthetic_samples: int) -> pl.DataFrame:
        if not self.fitted:
            raise ModelNotFittedError("Model must be fitted before generating data.")

        logger.info(f"Generating {n_synthetic_samples} synthetic samples...")
        generated_data_np = self.copula.rvs(nobs=n_synthetic_samples)
        logger.success(f"Generated synthetic data with shape {generated_data_np.shape}")

        generated_df_pl = pl.DataFrame(generated_data_np, schema=self.column_names)

        logger.info("Applying reverse preprocessing transformations...")
        reversed_data_pd = self.sdv_preprocessor.reverse_preprocessing(
            generated_df_pl, self.preprocessing_transformations
        )

        reversed_data_pd = reversed_data_pd.fill_null(0.0)

<<<<<<< HEAD
        logger.success("Synthetic data generation complete.")
        return reversed_data_pd

=======
        logger.info("Applying domain constraints...")
        reversed_data_pd = self._apply_domain_constraints(reversed_data_pd)

        logger.success("Synthetic data generation complete.")
        return reversed_data_pd

    def _estimate_corr_matrix(self, data: np.ndarray) -> np.ndarray:
        """Estimate correlation matrix using the configured corr_method."""
        match self.corr_method:
            case "pearson":
                return np.corrcoef(data, rowvar=False)
            case "kendall":
                n_cols = data.shape[1]
                tau_mat = np.eye(n_cols)
                for i in range(n_cols):
                    for j in range(i + 1, n_cols):
                        tau, _ = stats.kendalltau(data[:, i], data[:, j])
                        tau_mat[i, j] = tau_mat[j, i] = tau
                return np.sin(np.pi / 2 * tau_mat)
            case "spearman":
                rho_result, _ = stats.spearmanr(data)
                if data.shape[1] == 1:
                    rho_mat = np.array([[1.0]])
                elif not isinstance(rho_result, np.ndarray):
                    rho_mat = np.array([[1.0, rho_result], [rho_result, 1.0]])
                else:
                    rho_mat = rho_result
                return 2 * np.sin(np.pi / 6 * rho_mat)
            case _:
                raise ValueError(f"Unknown corr_method: {self.corr_method!r}")

    def _estimate_student_t_df(self, data: np.ndarray) -> float:
        """Estimate Student-T df by fitting stats.t per column and taking the median."""
        dfs = []
        for col_idx in range(data.shape[1]):
            try:
                df, _, _ = stats.t.fit(data[:, col_idx], floc=0, fscale=1)
                if np.isfinite(df) and df > 0:
                    dfs.append(df)
            except Exception:
                pass
        return float(np.median(dfs)) if dfs else 5.0

    def _apply_domain_constraints(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clip columns according to peptide_non_negative and clip_columns config."""
        expressions = []

        if self.peptide_non_negative:
            for col in df.columns:
                if re.search(r"(?i)peptide", col):
                    expressions.append(pl.col(col).clip(lower_bound=0.0).alias(col))

        for pattern, bounds in self.clip_columns.items():
            lo, hi = bounds[0], bounds[1]
            for col in df.columns:
                if re.search(pattern, col):
                    expressions.append(
                        pl.col(col).clip(lower_bound=lo, upper_bound=hi).alias(col)
                    )

        if expressions:
            df = df.with_columns(expressions)
        return df

>>>>>>> troubleshooting
    def transform_to_corr_space(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            match self.copula_type:
                case "gaussian":
                    self.copula_generator_func = QuantileTransformer(
                        output_distribution="normal"
                    )
                    return self.copula_generator_func.fit_transform(X)
                case "student":
                    raise ValueError(
                        f"Generator fucntion not Implemented for Student-T copula."
                    )
                case _:
                    raise ValueError(f"{self.copula_type} unknowns copula type!")
        else:
            return self.copula_generator_func.transform(X)

    def _fit(
        self,
        real_dataset: DataFrame,
        dataset_metadata: Optional[DatasetMetadata] = None,
    ) -> None:
        logger.info("Starting model fitting process...")

        logger.info("Preprocessing real dataset...")
        real_dataset = self.convert_zeroes_to_nulls(real_dataset)
        real_dataset = self.convert_int_to_categorical(real_dataset)

        preprocessed_dataset_pl, transformations = self.sdv_preprocessor.preprocess(
            real_dataset, categorical_columns=self.categorical_columns
        )
        self.preprocessing_transformations = transformations
        logger.success("Preprocessing complete.")

        self.column_names = preprocessed_dataset_pl.columns

        logger.info("Fitting marginal distributions...")
        fitted_marginals = (
            self.marginal_distribution_estimator.fit_marginal_distributions(
                preprocessed_dataset_pl,
            )
        )
        logger.success("Marginal distribution fitting complete.")

<<<<<<< HEAD
        logger.info("Fitting Gaussian Copula correlation matrix...")
=======
        logger.info(f"Fitting copula correlation matrix (method={self.corr_method})...")
>>>>>>> troubleshooting
        preprocessed_data_np = self.transform_to_corr_space(
            preprocessed_dataset_pl.to_numpy()
        )

<<<<<<< HEAD
        estimated_correlation = np.corrcoef(preprocessed_data_np, rowvar=False)
=======
        estimated_correlation = self._estimate_corr_matrix(preprocessed_data_np)
>>>>>>> troubleshooting

        logger.info("Nearest correlation matrix fitting...")
        estimated_correlation = statsmodels.stats.correlation_tools.corr_clipped(
            estimated_correlation, threshold=1e-5
        )

        logger.success("Nearest correlation matrix fitting complete.")

        if np.isnan(estimated_correlation).any():
            logger.warning("NaN values found in estimated correlation matrix.")

        match self.copula_type:
            case "gaussian":
                fitted_cop = GaussianCopula(
                    corr=estimated_correlation, allow_singular=True
                )
            case "student_t":
<<<<<<< HEAD
                fitted_cop = StudentTCopula(corr=estimated_correlation, df=1)
=======
                if self.student_t_df is None:
                    df = self._estimate_student_t_df(preprocessed_data_np)
                    logger.info(f"Estimated Student-T df from data: {df:.2f}")
                else:
                    df = float(self.student_t_df)
                    logger.info(f"Using fixed Student-T df: {df}")
                fitted_cop = StudentTCopula(corr=estimated_correlation, df=df)
>>>>>>> troubleshooting
            case _:
                raise ValueError(f"Unknown copula type: {self.copula_type}")

        self.copula = fitted_cop
        logger.success("Copula correlation fitting complete.")

        logger.info("Creating model signature...")
        marginal_info_list = []
        reconstructed_marginal_instances = []

        for estimated_marginal_distribution in fitted_marginals:
            marginal_info_list.append(
                estimated_marginal_distribution.marginal_distribution_info
            )

            reconstructed_marginal_instances.append(
                estimated_marginal_distribution.distribution(
                    *estimated_marginal_distribution.marginal_distribution_info.parameters
                )
            )

        self._model_signature_to_save = ModelSignature(
            correlation_matrix=estimated_correlation.tolist(),
            marginals=marginal_info_list,
            preprocessing_info=self.preprocessing_transformations,
            column_names=self.column_names,
        )

        logger.success("Model signature created.")

        self.copula = CopulaDistribution(
            copula=fitted_cop, marginals=reconstructed_marginal_instances
        )

        self.fitted = True
        logger.success(
            "Model fitting process completed successfully. Model is now fitted."
        )

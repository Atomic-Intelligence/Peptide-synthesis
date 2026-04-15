import numpy as np
import polars as pl
from loguru import logger
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import (
    QuantileTransformer,
    RobustScaler,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)
from typing import Union, Optional, Tuple, Dict, Any, List, Callable
from pydantic import BaseModel, ConfigDict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

<<<<<<< HEAD
=======
from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler as _Scaler  # shared utility

>>>>>>> troubleshooting
# Define types for clarity
Scaler = Union[QuantileTransformer, RobustScaler, StandardScaler, MinMaxScaler]


class AuthenticityResults(BaseModel):
    """Data class to store and organize authenticity evaluation results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    binary_authenticity: np.ndarray
    authenticity_ratio: np.ndarray
    mean_binary_score: float
    mean_ratio_score: float
    suspicious_indices: np.ndarray = None
    feature_importance: Dict[str, float] = None

    def summary(self) -> Dict[str, Any]:
        """Return a summary of authenticity results."""
        return {
            "mean_binary_score": self.mean_binary_score,
            "mean_ratio_score": self.mean_ratio_score,
            "num_suspicious_samples": (
                len(self.suspicious_indices)
                if self.suspicious_indices is not None
                else 0
            ),
            "feature_importance": self.feature_importance,
        }


<<<<<<< HEAD
class FeatureProcessor:
    """Handles transformation of features - separate from main estimation logic."""

    def __init__(
        self,
        scaler: Scaler,
        categorical_columns: List[str] = None,
    ):
        self.scaler = scaler
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="warn")
        self.categorical_columns = categorical_columns or []
        self.fitted = False
        self.feature_names = None

    def fit(self, dataframe: pl.DataFrame) -> None:
        """Fit the feature processor to the training data."""
        numerical_cols = [
            col for col in dataframe.columns if col not in self.categorical_columns
        ]

        # Fit the scaler on numerical features
        self.scaler.fit(dataframe.select(numerical_cols).to_numpy())

        # Fit the encoder on categorical features if any
        if self.categorical_columns:
            self.one_hot_encoder.fit(
                dataframe.select(self.categorical_columns).to_numpy()
            )

        # Store the transformed feature names for interpretability
        self.numerical_feature_names = numerical_cols
        if self.categorical_columns:
            self.categorical_feature_names = self.one_hot_encoder.get_feature_names_out(
                self.categorical_columns
            )
            self.feature_names = np.concatenate(
                [self.numerical_feature_names, self.categorical_feature_names]
            )
        else:
            self.feature_names = np.array(self.numerical_feature_names)

        self.fitted = True

    def transform(self, dataframe: pl.DataFrame) -> np.ndarray:
        """Transform input data using fitted scalers and encoders."""
        if not self.fitted:
            raise ValueError("Feature processor must be fitted before transform")

        numerical_cols = [
            col for col in dataframe.columns if col not in self.categorical_columns
        ]
        numerical = self.scaler.transform(dataframe.select(numerical_cols).to_numpy())

        if not self.categorical_columns:
            return numerical

        categorical = self.one_hot_encoder.transform(
            dataframe.select(self.categorical_columns).to_numpy()
        )
        return np.concatenate([numerical, categorical], axis=-1)

    def fit_transform(self, dataframe: pl.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(dataframe)
        return self.transform(dataframe)


=======
>>>>>>> troubleshooting
class AuthenticityEstimator:
    """Evaluates the authenticity of synthetic data compared to real data."""

    def __init__(
        self,
        scaler: Scaler,
        categorical_columns: List[str] = None,
        algorithm: str = "ball_tree",
        authenticity_threshold: float = 1.0,
        verbose: bool = True,
        metric: str | Callable = "minkowski",
<<<<<<< HEAD
    ):
        self.feature_processor = FeatureProcessor(scaler, categorical_columns)
=======
        fitted_feature_processor: Optional[FeatureProcessor] = None,
    ):
        if fitted_feature_processor is not None:
            self.feature_processor = fitted_feature_processor
        else:
            self.feature_processor = FeatureProcessor(scaler, categorical_columns)
>>>>>>> troubleshooting
        self.knn = NearestNeighbors(algorithm=algorithm, metric=metric)
        self.authenticity_threshold = authenticity_threshold
        self.verbose = verbose
        self.real_data = None
        self.real_dataframe = None

    def fit(self, real_dataframe: pl.DataFrame) -> "AuthenticityEstimator":
        """Fit the model on real data."""
        logger.info("Preparing real data for modeling")
        self.real_dataframe = real_dataframe
<<<<<<< HEAD
        self.real_data = self.feature_processor.fit_transform(real_dataframe)
=======
        if self.feature_processor.fitted:
            self.real_data = self.feature_processor.transform(real_dataframe)
        else:
            self.real_data = self.feature_processor.fit_transform(real_dataframe)
>>>>>>> troubleshooting

        logger.info("Fitting nearest neighbors model")
        self.knn.fit(self.real_data)
        logger.success("Model fitted successfully")

        return self

    def _calculate_neighbor_distances(
        self, synthetic_data: np.ndarray, real_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate distances between synthetic samples and their nearest real neighbors."""
        # Calculate distances from synthetic points to nearest real points
        logger.warning(f"Synthetic has NaN {np.isnan(synthetic_data).any()}")
        synth_to_real_dist, synth_to_real_idx = self.knn.kneighbors(
            synthetic_data, n_neighbors=1
        )

        # For each real neighbor of a synthetic point, find its nearest real neighbors
        # We use n_neighbors=2 because the first neighbor will be the point itself
        real_to_real_dist, real_to_real_idx = self.knn.kneighbors(
            real_data[synth_to_real_idx.flatten()], n_neighbors=2
        )

        return (
            synth_to_real_dist,
            synth_to_real_idx,
            real_to_real_dist,
            real_to_real_idx,
        )

    def get_instance_authenticity(
        self,
        synthetic_dataframe: pl.DataFrame,
        real_dataframe: pl.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate authenticity metrics for each synthetic instance."""
        # Use provided real data or fall back to the fitted data
        if real_dataframe is not None:
            logger.info("Using newly provided real data")
            real_data = self.feature_processor.transform(real_dataframe)
        else:
            if self.real_data is None:
                raise ValueError(
                    "Model not fitted with real data and no real data provided"
                )
            real_data = self.real_data

        logger.info("Transforming synthetic data")
        synthetic_data = self.feature_processor.transform(synthetic_dataframe)

        # Calculate distances
        synth_to_real_dist, synth_to_real_idx, real_to_real_dist, real_to_real_idx = (
            self._calculate_neighbor_distances(synthetic_data, real_data)
        )

        # Calculate authenticity metrics
        binary_authenticity = synth_to_real_dist[:, 0] > real_to_real_dist[:, 1]
        authenticity_ratio = (
            synth_to_real_dist.flatten() / real_to_real_dist[:, 1].flatten()
        )

        if self.verbose:
            avg_distance = np.concatenate(
                [real_to_real_dist[:, 1], synth_to_real_dist[:, 0]], axis=-1
            ).mean()
            logger.info(f"Average distance to closest neighbor: {avg_distance:.4f}")

        return binary_authenticity, authenticity_ratio

    def estimate_authenticity(
        self,
        synthetic_dataframe: pl.DataFrame,
        real_dataframe: Optional[pl.DataFrame] = None,
        return_suspicious: bool = False,
        feature_importance: bool = False,
    ) -> AuthenticityResults:
        """Calculate overall authenticity scores and optionally identify suspicious samples."""
        binary_authenticity, authenticity_ratio = self.get_instance_authenticity(
            synthetic_dataframe, real_dataframe
        )

        # Calculate mean scores (excluding infinite values)
        mean_binary = binary_authenticity.mean().item()
        valid_ratios = authenticity_ratio[authenticity_ratio != np.inf]
        mean_ratio = valid_ratios.mean().item() if len(valid_ratios) > 0 else np.inf

        results = AuthenticityResults(
            binary_authenticity=binary_authenticity,
            authenticity_ratio=authenticity_ratio,
            mean_binary_score=mean_binary,
            mean_ratio_score=mean_ratio,
        )

        # Identify suspicious samples
        if return_suspicious:
            results.suspicious_indices = np.where(
                authenticity_ratio > self.authenticity_threshold
            )[0]
            logger.info(
                f"Identified {len(results.suspicious_indices)} suspicious samples"
            )

        # Calculate feature importance
        if feature_importance:
            results.feature_importance = self._calculate_feature_importance(
                synthetic_dataframe, binary_authenticity
            )

        return results

    def _calculate_feature_importance(
        self, synthetic_dataframe: pl.DataFrame, binary_authenticity: np.ndarray
    ) -> Dict[str, float]:
        """Calculate which features most contribute to inauthenticity."""
        if not hasattr(self.feature_processor, "feature_names"):
            return None

        # Transform synthetic data
        synthetic_data = self.feature_processor.transform(synthetic_dataframe)

        # Split data into authentic and inauthentic
        authentic_samples = synthetic_data[~binary_authenticity]
        inauthentic_samples = synthetic_data[binary_authenticity]

        if len(authentic_samples) == 0 or len(inauthentic_samples) == 0:
            logger.info(
                "Cannot calculate feature importance: need both authentic and inauthentic samples"
            )
            return None

        # Calculate differences in feature distributions
        authentic_mean = np.median(authentic_samples, axis=0)
        inauthentic_mean = np.median(inauthentic_samples, axis=0)

        # Calculate absolute differences
        feature_diffs = np.abs(authentic_mean - inauthentic_mean)

        # Create feature importance dictionary
        importance_dict = {
            feature: diff
            for feature, diff in zip(
                self.feature_processor.feature_names, feature_diffs
            )
        }

        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def visualize_authenticity(
        self,
        results: AuthenticityResults,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize authenticity results with various plots."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot 1: Histogram of authenticity ratios
        lower = np.percentile(
            results.authenticity_ratio[results.authenticity_ratio != np.inf], 0
        )
        upper = np.percentile(
            results.authenticity_ratio[results.authenticity_ratio != np.inf], 95
        )
        sns.histplot(
            results.authenticity_ratio[results.authenticity_ratio != np.inf],
            bins=30,
            binrange=(lower, upper),
            ax=axes[0],
        )

        axes[0].set_title("Distribution of Authenticity Ratios")
        axes[0].set_xlabel("Authenticity Ratio")
        axes[0].axvline(
            self.authenticity_threshold, color="red", linestyle="--", label="Threshold"
        )
        axes[0].legend()

        # Plot 2: Binary authenticity
        labels = ["Authentic", "Inauthentic"]
        counts = [
            results.binary_authenticity.sum(),
            (~results.binary_authenticity).sum(),
        ]
        axes[1].pie(counts, labels=labels, autopct="%1.1f%%")
        axes[1].set_title("Binary Authenticity Classification")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig

    def audit_synthetic_data(
        self,
        synthetic_dataframe: pl.DataFrame,
        real_dataframe: Optional[pl.DataFrame] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform a comprehensive audit of synthetic data quality."""
        results = self.estimate_authenticity(
            synthetic_dataframe=synthetic_dataframe,
            real_dataframe=real_dataframe,
            return_suspicious=True,
            feature_importance=True,
        )

        # Visualize results
        self.visualize_authenticity(
            synthetic_dataframe=synthetic_dataframe,
            results=results,
            save_path=output_path,
        )

        # Create audit report
        audit_report = {
            "authenticity_scores": results.summary(),
            "sample_count": len(synthetic_dataframe),
            "authentic_percentage": (~results.binary_authenticity).mean() * 100,
            "suspicious_samples_count": (
                len(results.suspicious_indices)
                if results.suspicious_indices is not None
                else 0
            ),
        }

        if (
            results.suspicious_indices is not None
            and len(results.suspicious_indices) > 0
        ):
            # Extract top suspicious samples
            top_suspicious = results.suspicious_indices[
                np.argsort(results.authenticity_ratio[results.suspicious_indices])[-5:]
            ]
            audit_report["top_suspicious_samples"] = synthetic_dataframe.rows(
                top_suspicious
            )

        return audit_report

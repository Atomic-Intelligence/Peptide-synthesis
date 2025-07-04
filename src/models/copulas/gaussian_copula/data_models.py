# --- Pydantic Models ---
from pydantic import BaseModel, Field

from src.models.copulas.marginal_distributions.marginal_distribution_metrics import (
    UnivariateDistribution,
)


class MarginalDistributionInfo(BaseModel):
    distribution_name: str = Field(
        ..., description="Name of the distribution (e.g., 'gamma', 'norm')"
    )
    parameters: list[float] = Field(
        ..., description="Fitted parameters for the distribution."
    )
    column_name: str = Field(
        ..., description="Name of the column this distribution was fitted for."
    )


class ModelSignature(BaseModel):
    correlation_matrix: list[list[float]] = Field(
        ..., description="Estimated correlation matrix."
    )
    marginals: list[MarginalDistributionInfo] = Field(
        ..., description="List of fitted marginal distributions."
    )
    preprocessing_info: dict = Field(
        ..., description="Preprocessing transformations applied by SDVPreprocessor."
    )

    column_names: list[str] = Field(
        ..., description="Column names of the original dataset."
    )


class EstimatedMarginalDistribution(BaseModel):
    distribution: UnivariateDistribution = Field(
        ..., description="Estimated distribution."
    )

    marginal_distribution_info: MarginalDistributionInfo = Field(
        ..., description="Marginal distribution info."
    )

    class Config:
        arbitrary_types_allowed = True

from pathlib import Path
from typing import Any

import polars as pl
from src.data.data_loader import DataLoader
from src.data.data_models import Processor
from src.modeling.distribution_modeling import DistributionEstimator
from src.modeling.synthetization import Synthesizer
from src.modeling.custom_copula_synthesizer import CustomGaussianCopulaSynthesizer
from src.data.data_merge_and_save import merge_and_save


def data_synthesis(
    peptide_data_path: str,
    clinical_data_path: str,
    save_path: str,
    missing_threshold: float,
    primary_key: str,
    distribution_list: list[str],
    fit_distribution_method: str,
    filters: list[dict],
    number_of_synth_samples: int,
    batch_size: int,
    constraints: list[dict[str, Any]],
    processor: Processor | None = None,
    random_seed: int | None = None,
    clinical_columns_to_estimate: list[str] | None = None,
    number_of_original_samples: int | None = None,
):
    distribution_estimator = DistributionEstimator(
        primary_key, distribution_list, fit_distribution_method
    )
    synth_df = []
    for filter_dict in filters:
        loader = DataLoader(
            clinical_data_path,
            peptide_data_path,
            primary_key,
            number_of_original_samples,
            processor,
            filter_dict,
        )
        print("Loading data...")
        data = loader.get_data()
        # find peptides that have at least 30% non-zero values
        print("Getting peptides for modelling...")
        peptides_to_model, low_count_peptides = processor.get_peptides_for_modelling(
            data.peptides, missing_threshold
        )

        # estimate marginal distributions
        print("Estimating marginal distributions...")
        # peptides_to_model = peptides_to_model.select(peptides_to_model.columns[:5])
        distributions = distribution_estimator.estimate(peptides_to_model)
        for clinical_column in clinical_columns_to_estimate:
            distributions[clinical_column] = (
                distribution_estimator.estimate_single_column_distribution(
                    data.clinical[clinical_column]
                )
            )

        # merge clinical data with peptides_to_model
        original_data: pl.DataFrame = data.clinical.join(
            peptides_to_model, on=primary_key
        )

        # initialize synthesizer
        peptides_to_model_names = [
            col for col in peptides_to_model.columns if col != primary_key
        ]

        synthesizer = Synthesizer(
            original_data=original_data,
            primary_key=primary_key,
            peptides_to_model=peptides_to_model_names,
            sdv_synthesizer=CustomGaussianCopulaSynthesizer,
            random_seed=random_seed,
            numerical_distributions=distributions,
            constraints=constraints,
        )

        synthesizer.fit()

        # sample
        synthetic_data = synthesizer.sample(number_of_synth_samples, batch_size)

        synth_df.append(
            processor.postprocess_data(data, low_count_peptides, synthetic_data)
        )

    clinical_data_list = [data.clinical for data in synth_df]
    peptides_data_list = [data.peptides for data in synth_df]
    merge_and_save(clinical_data_list, peptides_data_list, primary_key, Path(save_path))

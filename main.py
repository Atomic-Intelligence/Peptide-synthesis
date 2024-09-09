import yaml
from src.data.hf_data_merging import merge_hf_data
from data_synthesis import data_synthesis
from src.data.data_processing import HFProcessorForSynthetization
from src.modeling.bootstrapping_results import bootstrapping_data
from pathlib import Path
import pandas as pd

def load_config(config_file="configuration.yaml"):
    """
    Load configuration from a YAML file.

    :param config_file: Path to the YAML configuration file.
    :return: Dictionary containing the configuration.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Load the configuration
    config = load_config()

    # Access configuration sections
    initial_data_handling = config.get("initial_data_handling", {})
    synthesis = config.get("synthesis", {})

    if initial_data_handling.get("inital_data_merging"):
        merge_hf_data(
            initial_data_handling.get("root_dir_path"),
            initial_data_handling.get("save_dir_path"),
        )

    filtering_ = synthesis.get("filtering")
    filters = []
    for item in filtering_:
        for key, value in item.items():
            filters.append({key: tuple(value)})

    peptide_data_paths = synthesis.get("peptide_data_paths")
    clinical_data_paths = synthesis.get("clinical_data_paths")
    save_paths = synthesis.get("save_paths")
    bootstrapping = synthesis.get("bootstrapping")
    bootstrapping_nonzero_threshold = synthesis.get("bootstrapping_nonzero_threshold")
    bootstrapping_sample_sizes = synthesis.get("bootstrapping_sample_sizes")
    bootstrapping_iteration_number = synthesis.get("bootstrapping_iteration_number")
    missing_threshold = synthesis.get("missing_threshold")
    primary_key = synthesis.get("primary_key")
    n_of_original_samples = synthesis.get("number_of_original_samples")
    distribution_list = synthesis.get("distribution_list")
    fit_distr_method = synthesis.get("fit_distribution_method")
    random_seed = synthesis.get("random_seed")
    batch_size = synthesis.get("batch_size")
    n_of_synth_samples = synthesis.get("number_of_synth_samples")
    clinical_columns_to_estimate = synthesis.get("clinical_columns_to_estimate")
    constraints = synthesis.get("constraints")

    processor = HFProcessorForSynthetization(primary_key=primary_key)

    for i in range(len(peptide_data_paths)):
        data_synthesis(
            peptide_data_paths[i],
            clinical_data_paths[i],
            save_paths[i],
            missing_threshold,
            primary_key,
            distribution_list,
            fit_distr_method,
            filters,
            n_of_synth_samples[i],
            batch_size,
            constraints,
            processor,
            random_seed,
            clinical_columns_to_estimate,
            n_of_original_samples,
        )
        if bootstrapping:
            print(bootstrapping_nonzero_threshold)
            print("#### Bootstrapping started ####")
            path_to_synth_table = Path(save_paths[i], "synthetic_data_peptides.csv")
            best_seed, statistic = bootstrapping_data(
                path_to_synth_table,
                peptide_data_paths[i],
                bootstrapping_nonzero_threshold,
                bootstrapping_sample_sizes[i],
                bootstrapping_iteration_number
            )
            data = pd.read_csv(path_to_synth_table).sample(bootstrapping_sample_sizes[i], random_state=best_seed)

            # Saving bootstrapped peptides data
            print("#### Saving bootstrapped data ####")
            data.to_csv(
                Path(save_paths[i], "synthetic_data_peptides_bootstrapped.csv"),
                header=True,
                index=False
            )

            # select sample ids
            sample_ids = data[primary_key]  # Adjust 'id' to match your actual column name for IDs

            # load clinical data
            path_to_clin_table = Path(save_paths[i], "synthetic_data_clinical.csv")
            clinical = pd.read_csv(path_to_clin_table)

            # Filter clinical table
            clinical_sample = clinical[clinical[primary_key].isin(sample_ids)]

            # Saving bootstrapped clinical data
            clinical_sample.to_csv(
                Path(save_paths[i], "synthetic_data_clinical_bootstrapped.csv"),
                header=True,
                index=False
            )

            # save statistic
            print("#### Saving statistic ####")
            stat = [
                {'Peptide_id': peptide_id,
                 'kl_divergence': values['kl_divergence'],
                 'ks_p-value': values['ks_p-value']}
                for peptide_id, values in statistic.items()
            ]
            # Convert the list of dictionaries into a pandas DataFrame
            df = pd.DataFrame(stat)

            # Save the DataFrame to a CSV file
            df.to_csv(
                Path(save_paths[i], "synthetic_data_peptides_statistic.csv"),
                header=True,
                index=False
            )




if __name__ == "__main__":
    main()

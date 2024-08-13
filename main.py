import yaml
from src.data.hf_data_merging import merge_hf_data
from data_synthesis import data_synthesis
from src.data.data_processing import HFProcessorForSynthetization


def load_config(config_file='configuration.yaml'):
    """
    Load configuration from a YAML file.

    :param config_file: Path to the YAML configuration file.
    :return: Dictionary containing the configuration.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Load the configuration
    config = load_config()

    # Access configuration sections
    initial_data_handling = config.get('initial_data_handling', {})
    synthesis = config.get('synthesis', {})

    if initial_data_handling.get('inital_data_merging'):
        merge_hf_data(
            initial_data_handling.get('root_dir_path'),
            initial_data_handling.get('save_dir_path'),
        )

    filtering_ = synthesis.get('filtering')
    filters = []
    for item in filtering_:
        for key, value in item.items():
            filters.append({key: tuple(value)})

    peptide_data_paths = synthesis.get('peptide_data_paths')
    clinical_data_paths = synthesis.get('clinical_data_paths')
    save_paths = synthesis.get('save_paths')
    missing_threshold = synthesis.get('missing_threshold')
    primary_key = synthesis.get('primary_key')
    n_of_original_samples = synthesis.get('number_of_original_samples')
    distribution_list = synthesis.get('distribution_list')
    fit_distr_method = synthesis.get('fit_distribution_method')
    random_seed = synthesis.get('random_seed')
    batch_size = synthesis.get('batch_size')
    n_of_synth_samples = synthesis.get('number_of_synth_samples')
    clinical_columns_to_estimate = synthesis.get('clinical_columns_to_estimate')

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
            n_of_synth_samples,
            batch_size,
            processor,
            random_seed,
            clinical_columns_to_estimate,
            n_of_original_samples
        )


if __name__ == "__main__":
    main()

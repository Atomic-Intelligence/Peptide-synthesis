# Peptide-synthesis

## Overview
This repository contains the code for generation of synthetic clinical and peptide data using
the gaussian copula approach. It can be used for generation of any number of synthetic patients
for any dataset containing clinical and peptide data.
## Features
- custom filtering of data allowing generation of synthetic data for any number of diseases
- extendable list of probability distributions that can be made to fit various datasets
- able to generate an unlimited number of synthetic patients
- customizable number of peptides and original patients to be used, allowing for rapid prototyping

## Installation

A prerequisite to running the code is a working `python 3.11` installation. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```
Installation should take around 3 minutes to complete.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Atomic-Intelligence/Peptide-synthesis.git
   cd Peptide-synthesis
   ```
2. Set up a dataset containing peptide and clinical data. The `resources` directory contains an example of a valid dataset on which the pipeline can be run.
3. Set up the configuration file `configuration.yaml` to fit your experiment. This is an example configuration for reproducing our research.
   ```yaml
   initial_data_handling:
      inital_data_merging: True  #  True if merging of multiple Excel sheets is needed
      root_dir_path: ""  #  directory containing the real dataset
      save_dir_path: ""  #  directory where synthetic data will be saved
   
   synthesis:
      filtering:  #  filters for each group of patients which should be modeed separately
         - Kidney_disease:  #  first group are patients which have kidney disease
             - "="
             - 1
         - Kidney_disease:  # second group are patients which do not have kidney disease
             - "="
             - 0
      peptide_data_paths:   # paths to peptide data for each group
        - ""
        - ""
      clinical_data_paths:  # paths to clinical data for each groups
        - ""
        - ""
      save_paths:  #  paths where results will be saved
        - ""
        - ""
      missing_threshold: 0.7  #  peptides with a percentage of missing values over this are not modeled 
      primary_key: 'idAuswertung'  #  primary key for the dataset
      number_of_original_samples: None  #  if you want to use just a subset of original patients, specify the number here
      distribution_list:  #  available distributions
        - 'lognorm'
        - 'norm'
      fit_distribution_method: "sumsquare_error"  #  method to choose best fitting distribution for each variable
      random_seed: 42  #  random seed if you want to fix the experiment
      batch_size: 100  #  batch size for faster sampling
      number_of_synth_samples:  #  number of synthetic patients to generate
        - [250, 250] # each number in a list defines the number of samples for different filters in filtering
      clinical_columns_to_estimate:  #  clinical variables for which distribution should be estimated
        - "GFR_CKD_EPI_M"
      constraints:  # list of rule based constraints for your data
        - constraint_class: "Inequality"
          constraint_parameters:
            low_column_name: "Blutdruck, diastolischM"
            high_column_name: "Blutdruck, systolischM"
            strict_boundaries: True
    ```
3. Run the main script using the Python environment where the requirements have been installed:
   ```bash
    python3 main.py
   ```

## Bootstraping:

If bootstraping is enabled in `configuration.yaml`, script will evaluate the similarity between synthetic and original peptide datasets through a bootstrapping procedure:

1. **Data Preparation**: Loads the datasets and selects columns with sufficient non-zero values for analysis.
2. **Bootstrapping**: Iteratively samples subsets of synthetic data, comparing them to the original dataset using two metrics:
   - **Kolmogorov-Smirnov (KS) p-value**: Measures distributional similarity.
   - **Kullback-Leibler (KL) Divergence**: Quantifies differences in probability distributions.
3. **Best Subset Selection**: Identifies the subset with the lowest KS results and improved KL divergence.
4. **Output**:
   - Saves the refined synthetic peptide subset, corresponding clinical data, and statistical metrics for documentation.

This process ensures the synthetic dataset closely mirrors the original in key statistical properties.

## Demo data

In `/resources` directory, demo data can be seen. This data can be used for testing the provided code. When running the main script with the default configuration, the code should run for around 20 seconds and after that, your `/output` directory should contain the following files:

- `synthetic_data_clinical_bootstrapped.csv`
- `synthetic_data_clinical.csv`
- `synthetic_data_peptides_bootstrapped.csv`
- `synthetic_data_peptides_statistic.csv`
- `synthetic_data_peptides.csv`.

File with statistical properties contains peptide ids, kl divergences and ks p-values for corresponding peptides while other files contain synthesized data with the following columns:

#### Clinical data:
- Patient ID
- Hospitalization duration
- Sex (0-male)
- Kidney disease
- Diabetes
- Hypertension
- Blutdruck, diastolischM
- Blutdruck, systolischM
- GFR_CKD_EPI_M
- BMI
- Age

#### Peptide data:
- Patient ID
- p1
- p2
- p3
- p4
- p5
- p6
- p7
- p8
- p9
- p10
- p11
- p12
- p13
- p14
- p15.
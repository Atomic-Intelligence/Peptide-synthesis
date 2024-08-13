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

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Atomic-Intelligence/Peptide-synthesis.git
   cd Peptide-synthesis
   ```
2. Set up the configuration file `configuration.yaml` to fit your experiment. This is an example configuration for reproducing our research.
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
      number_of_synth_samples: 500  #  number of synthetic patients to generate
      clinical_columns_to_estimate:  #  clinical variables for which distribution should be estimated
        - "GFR_CKD_EPI_M"
    ```
3. Run the main script using the Python environment where the requirements have been installed:
   ```bash
    python3 synthesize.py
   ```

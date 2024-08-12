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
2. Set up the configuration file `configuraition.yaml` to fit your experiment
   ```yaml
   a: b tu cemo dodat primjer kako izgleda yamlica
    ```
3. Run the main script using the Python environment where the requirements have been installed:
   ```bash
    python3 synthesize.py
   ```

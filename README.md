# Peptide Synthesis Research

A comprehensive pipeline for synthetic peptide data generation and evaluation using machine learning models, particularly focusing on heart failure and chronic kidney disease biomarker analysis with advanced privacy-preserving capabilities.

## Overview

This project provides a complete workflow for:
- Training synthetic data generation models on real peptide datasets
- Generating synthetic peptide data that preserves clinical characteristics
- Evaluating the quality of synthetic data through multiple analysis methods
- Privacy evaluation and authenticity assessment of synthetic data
- Supporting multiple generative models (Gaussian Copula, GAN, Adversarial Random Forest)

## Installation

### Prerequisites
- Python 3.12+
- Poetry for dependency management

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd Peptide-synthesis
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Project Structure

```
├── src/
│   ├── run_pipeline.py              # Main training and inference pipeline
│   ├── evaluation/
│   │   ├── data_evaluation_script.py # Evaluation pipeline
│   │   └── privacy/                 # Privacy evaluation module
│   │       ├── AuthenticityEstimator.py # KNN-based authenticity assessment
│   │       └── estimate_privacy.py  # Privacy evaluation script
│   ├── models/                      # Generative models
│   │   ├── copulas/                 # Gaussian Copula implementation
│   │   ├── ctgan/                   # GAN-based models
│   │   └── AdversarialRandomForests/ # ARF implementation
│   └── data/                        # Data processing utilities
├── configs/
│   ├── training_and_inference_pipeline/ # Pipeline configurations
│   └── evaluation/                  # Evaluation configurations
│       ├── privacy_script.yaml      # Privacy evaluation configuration
│       └── authenticity_estimator/  # Authenticity assessment configs
├── tests/                           # Test files
└── outputs/                         # Generated outputs and logs
```

## Running the Pipeline

### 1. Training and Inference Pipeline

The main pipeline handles model training and synthetic data generation:

```bash
# Run with default configuration
python src/run_pipeline.py

# Run with custom configuration
python src/run_pipeline.py --config-path configs/training_and_inference_pipeline --config-name pipeline.yaml
```

#### Key Configuration Parameters

**Pipeline Configuration** (`configs/training_and_inference_pipeline/pipeline.yaml`):
```yaml
# Training and inference control
run_training: true          # Enable/disable model training
run_inference: true         # Enable/disable synthetic data generation

# Event type to focus on
event: no_event            # Options: "no_event", "hf", "ckd"

# Patient sampling
sampled_patients_num: 0     # Number of patients to sample (0 = use all)

# Experiment tracking
experiment_name: "Gaussian Copula - V2 - Extra peptides - NonNegativity"
run_name: "NE"
```

**Model Configuration** (`configs/training_and_inference_pipeline/model/gaussian_copula.yaml`):
```yaml
# Gaussian Copula model settings
marginal_distribution_estimator:
  continuous_distributions:
    - stats.beta
    - stats.norm
    - stats.lognorm
    - stats.truncnorm
    - stats.gamma
    - stats.t
    - stats.uniform
    - stats.expon
  
  # Metric for selecting best distribution
  univariate_distribution_metric:
    _target_: src.models.copulas.marginal_distributions.marginal_distribution_metrics.KullbackLeiblerDivergence

# Preprocessing settings
sdv_preprocessor:
  _target_: src.models.copulas.gaussian_copula.preprocessing.SDVPreprocessor

copula_type: gaussian
```

**Data Paths** (`configs/training_and_inference_pipeline/paths/paths.yaml`):
```yaml
# Data directory structure
root_data_path: /data1/prostrat-ai
synthetic_data_dir: ${.root_data_path}/synthetic_datasets

# Input data paths
real_dataset_path: ${.root_data_path}/data/merged_peptide_and_clinical.csv
synthetic_dataset_path: ${.synthetic_data_dir}

# Output directory
run_dir: ./
```

### 2. Evaluation Pipeline

The evaluation pipeline provides comprehensive analysis of synthetic data quality:

```bash
# Run evaluation with default settings
python src/evaluation/data_evaluation_script.py

# Run with custom configuration
python src/evaluation/data_evaluation_script.py --config-path configs/evaluation --config-name data_eval_config.yaml
```

### 3. Privacy Evaluation Pipeline

Privacy evaluation using authenticity estimation:

```bash
# Run privacy evaluation with default settings
python src/evaluation/privacy/estimate_privacy.py

# Run with custom configuration
python src/evaluation/privacy/estimate_privacy.py --config-path configs/evaluation --config-name privacy_script.yaml
```

#### Key Evaluation Parameters

**Evaluation Configuration** (`configs/evaluation/data_eval_config.yaml`):
```yaml
# Processing settings
num_threads: 8              # Number of parallel threads
dataset_size_limit: 100     # Limit dataset size for testing (0 = no limit)

# Event types to analyze
event_types:
  - name: "hf"
    display_name: "Heart Failure"
    time_to_event_column: "FU duration_CAD_Hfevent (to event or last visit)"
  - name: "ckd"
    display_name: "Chronic Kidney Disease"
    time_to_event_column: "time-to-CKDevent(to event or last visit)"

# Analysis parameters
p_value_threshold: 0.05     # Statistical significance threshold
num_quantiles: 5            # Number of quantiles for survival analysis
threads_per_worker_divisor: 2 # CPU usage control
```

**Privacy Evaluation Configuration** (`configs/evaluation/privacy_script.yaml`):
```yaml
# Privacy evaluation settings
experiment_name: "Authenticity_Estimation"
description: "Comparing Data Authenticity for Generated Data"
zero_percentage: 0.4            # Threshold for peptide filtering

# Authenticity estimator configuration
defaults:
  - authenticity_estimator: knn_authenticity
  - paths: paths
  - mlflow: mlflow
```

**KNN Authenticity Configuration** (`configs/evaluation/authenticity_estimator/knn_authenticity.yaml`):
```yaml
# KNN-based authenticity assessment
_target_: src.evaluation.privacy.AuthenticityEstimator.AuthenticityEstimator
scaler:
  _target_: sklearn.preprocessing.RobustScaler
authenticity_threshold: 1.0     # Threshold for suspicious samples
categorical_columns:            # Categorical features for proper encoding
  - "event_type"
  - "Geschlecht (1=female)"
  - "Kidney disease"
  - "diabetes"
  - "CVD"
  - "hypertension"
```

**Data Paths for Evaluation** (`configs/evaluation/paths/paths.yaml`):
```yaml
# Paths for evaluation data
real_data_path: "/path/to/real_data.csv"
synthetic_data_path: "/path/to/synthetic_data.csv"
```

## Available Models

### 1. Gaussian Copula (Default)
- **Configuration**: `configs/training_and_inference_pipeline/model/gaussian_copula.yaml`
- **Features**: Multiple marginal distributions, KL divergence optimization

### 2. GAN (Generative Adversarial Network)
- **Configuration**: `configs/training_and_inference_pipeline/model/gan.yaml`
- **Features**: CTGAN implementation with tabular data support

### 3. Adversarial Random Forest
- **Configuration**: `configs/training_and_inference_pipeline/model/arf.yaml`
- **Features**: Tree-based adversarial training

## Evaluation Metrics

The evaluation pipeline provides multiple analysis methods:

### 1. Statistical Analysis
- **Mann-Whitney U tests**: Compare distributions between real and synthetic data
- **Survival analysis**: Assess time-to-event predictions
- **eGFR analysis**: Kidney function biomarker comparison

### 2. Machine Learning Evaluation
- **Classifier performance**: Train on real, test on synthetic (and vice versa)
- **F1 scores**: Binary classification performance for event prediction
- **Cross-validation**: Robust performance assessment

### 3.Privacy and Authenticity Assessment 
- **KNN-based authenticity scoring**: Quantitative assessment of synthetic data authenticity
- **Binary authenticity classification**: Identify authentic vs. inauthentic synthetic samples
- **Authenticity ratio analysis**: Compare distances between synthetic and real data neighborhoods
- **Suspicious sample detection**: Automatically identify potentially problematic synthetic samples
- **Feature importance analysis**: Determine which features contribute most to inauthenticity
- **Comprehensive privacy auditing**: Generate detailed privacy assessment reports

### 4. Visualization
- **PCA projections**: Dimensionality reduction analysis
- **UMAP projections**: Non-linear dimensionality reduction
- **Distribution plots**: Marginal distribution comparisons
- **Clinical variable analysis**: Clinical feature preservation
- **Authenticity distribution plots**: Visualize authenticity scores and suspicious samples


## MLflow Integration

The project uses MLflow for experiment tracking:

```yaml
# MLflow configuration
mlflow:
  tracking_uri: "http://10.100.111.210:5002"
  experiment_name: ${experiment_name}
```

### Logged Metrics
- Model training metrics
- Synthetic data generation statistics
- Evaluation scores (F1, accuracy, etc.)
- Statistical test results
- Privacy and authenticity assessment results
- Authenticity distribution visualizations
- Suspicious sample reports
- Visualization artifacts

## Output Files

### Training Pipeline Outputs
- `synthetic_{event}_sampled_{n_patients}.csv`: Generated synthetic data
- `outputs/{date}/{time}/run_pipeline.log`: Execution logs
- MLflow artifacts: Model parameters 

### Evaluation Pipeline Outputs
- Statistical analysis reports
- Classifier performance metrics
- Visualization plots (PCA, UMAP, distributions)
- Survival analysis results

### Privacy Evaluation Pipeline Outputs
- **Authenticity assessment reports**: Comprehensive privacy evaluation results
- **Suspicious sample identification**: Lists of potentially problematic synthetic samples
- **Feature importance analysis**: Analysis of which features contribute to inauthenticity
- **Authenticity distribution plots**: Visualizations of authenticity scores and classifications
- **Privacy audit reports**: Detailed privacy assessment with recommendations



## License

See LICENSE file for details.

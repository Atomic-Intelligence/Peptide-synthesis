# Peptide Synthesis Research

<<<<<<< HEAD
A comprehensive pipeline for synthetic peptide data generation and evaluation using machine learning models, particularly focusing on heart failure and chronic kidney disease biomarker analysis with advanced privacy-preserving capabilities.
=======
A comprehensive pipeline for synthetic peptide data generation and evaluation using machine learning models, with a focus on heart failure and chronic kidney disease biomarker analysis and advanced privacy risk assessment.
>>>>>>> troubleshooting

## Overview

This project provides a complete workflow for:
- Training synthetic data generation models on real peptide datasets
- Generating synthetic peptide data that preserves clinical characteristics
<<<<<<< HEAD
- Evaluating the quality of synthetic data through multiple analysis methods
- Privacy evaluation and authenticity assessment of synthetic data
- Supporting multiple generative models (Gaussian Copula, GAN, Adversarial Random Forest)

A legacy version from an early project iteration can be found at `v0`
=======
- Evaluating data quality (fidelity) through multiple statistical and ML-based metrics
- Assessing privacy risk through state-of-the-art attack simulations
- Hyperparameter tuning for generative models

A legacy version from an early project iteration can be found at `v0`.
>>>>>>> troubleshooting

## Installation

### Prerequisites
- Python 3.12+
- Poetry for dependency management

### Setup
<<<<<<< HEAD
=======

>>>>>>> troubleshooting
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
<<<<<<< HEAD
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
├── resources/                       # Technical reports and documentation
│   ├── ARF_and_GAN_report.pdf       # Adversarial Random Forest and GAN analysis
│   ├── Copulas_report.pdf           # Gaussian Copula implementation details
│   └── Privacy_Appendix.pdf         # Privacy evaluation methodology
├── tests/                           # Test files
└── outputs/                         # Generated outputs and logs
=======
│   ├── run_pipeline.py                   # Main training and inference entry point
│   ├── logger.py                         # Logging setup
│   ├── mlflow_utils.py                   # MLflow server management
│   ├── inference/
│   │   └── inference_runner.py           # Synthetic data generation runner
│   ├── models/
│   │   ├── synthetization_model_interface.py  # Abstract model interface
│   │   ├── copulas/                      # Gaussian Copula implementation
│   │   │   ├── gaussian_copula/
│   │   │   │   ├── gaussian_copula_basic.py
│   │   │   │   ├── preprocessing.py
│   │   │   │   ├── nearest_psd.py        # Positive semi-definite matrix repair
│   │   │   │   └── generator_functions.py
│   │   │   └── marginal_distributions/
│   │   │       ├── marginal_distribution_estimator.py
│   │   │       ├── marginal_distribution_metrics.py  # KL, BIC, AIC, KS
│   │   │       └── custom_distributions.py           # Zero-inflated variants
│   │   ├── ctgan/
│   │   │   └── gan_interface.py          # CTGAN wrapper
│   │   ├── AdversarialRandomForests/
│   │   │   ├── ARFPipeline.py            # ARF model pipeline
│   │   │   ├── arf.py                    # Core ARF implementation
│   │   │   └── utils.py                  # Decision tree boundary helpers
│   │   └── Imputation/
│   │       ├── HistogramImputation.py    # Histogram-based generation
│   │       └── KDEImputation.py          # Kernel density fallback
│   ├── evaluation/
│   │   ├── evaluate.py                   # Unified evaluation orchestrator
│   │   ├── data_evaluation_script.py     # Evaluation pipeline entry point
│   │   ├── fidelity/
│   │   │   ├── fidelity_report.py
│   │   │   ├── marginal_fidelity.py      # KS, TVD, Wasserstein, Hellinger
│   │   │   ├── correlation_fidelity.py   # Correlation structure preservation
│   │   │   ├── joint_fidelity.py         # MMD, multivariate tests
│   │   │   ├── classifier_test.py        # Two-sample classifier test
│   │   │   ├── sparse_peptide_fidelity.py# Zero-inflated peptide metrics
│   │   │   └── effect_size.py            # Cohen's d, Hellinger, Wasserstein
│   │   ├── privacy/
│   │   │   ├── privacy_report.py         # Privacy metrics orchestrator
│   │   │   ├── AuthenticityEstimator.py  # KNN-based authenticity scoring
│   │   │   ├── anonymeter_attacks.py     # Giomi et al. privacy attack framework
│   │   │   ├── dcr.py                    # Distance to Closest Record
│   │   │   ├── reidentification_risk.py  # Re-identification gap ratio
│   │   │   ├── membership_inference.py   # Membership inference attacks
│   │   │   ├── estimate_privacy.py       # Standalone privacy evaluation script
│   │   │   ├── preprocessing.py          # Feature encoding & scaling
│   │   │   └── gower_distance.py         # Mixed-type distance metric
│   │   ├── analysis/
│   │   │   ├── single_peptide_analysis.py
│   │   │   ├── multi_peptide_analysis.py # Survival analysis, time-to-event
│   │   │   ├── correlation_uncertainty.py# Bootstrap CI on correlations
│   │   │   └── clinical.py
│   │   ├── classifiers/
│   │   │   ├── ClassifierFactory.py
│   │   │   ├── machine_learning_efficiency.py
│   │   │   └── svm_grid_search.py
│   │   ├── projections/
│   │   │   ├── pca_projections.py
│   │   │   └── umap_projections.py
│   │   └── utils/
│   │       ├── plotting.py
│   │       └── eval_utils.py
│   └── data/
│       ├── PeptideDataset.py
│       ├── read_data.py
│       ├── utils.py                      # Column definitions, data manipulation
│       ├── make_toy_data.py
│       └── merge_synthetic_datasets.py
├── scripts/
│   ├── read_data.py                      # Excel peptide table reader (Polars)
│   ├── tune_ctgan.py                     # Optuna-based CTGAN hyperparameter tuning
│   ├── quantify_correlation_clipping.py
│   ├── peptide_distribution_comparison.py
│   └── pca_copula_synthetic_data.py
├── configs/
│   ├── training_and_inference_pipeline/
│   │   ├── pipeline.yaml
│   │   ├── model/
│   │   │   ├── gaussian_copula.yaml
│   │   │   ├── gan.yaml
│   │   │   ├── arf.yaml
│   │   │   ├── dummy.yaml
│   │   │   └── dummy_pretrained.yaml
│   │   ├── imputation/
│   │   │   ├── histogram_imputation.yaml
│   │   │   └── kde_imputation.yaml
│   │   ├── data_processor/processor.yaml
│   │   ├── inference/inference_runner.yaml
│   │   └── paths/paths.yaml
│   └── evaluation/
│       ├── evaluate.yaml
│       ├── fidelity.yaml
│       ├── privacy_script.yaml
│       ├── paths/paths.yaml
│       ├── authenticity_estimator/knn_authenticity.yaml
│       ├── classifier_models/classifiers.yaml
│       ├── svm_peptide_ids/
│       │   ├── ids.yaml
│       │   ├── hf_ids.yaml
│       │   └── ckd_ids.yaml
│       └── mlflow/mlflow.yaml
├── resources/
│   ├── ARF_and_GAN_report.pdf
│   ├── Copulas_report.pdf
│   ├── Privacy_Appendix.pdf
│   └── toy_data.csv
├── outputs/                              # Timestamped run outputs
└── tests/
>>>>>>> troubleshooting
```

## Running the Pipeline

<<<<<<< HEAD
### 1. Training and Inference Pipeline
=======
### 1. Training and Inference
>>>>>>> troubleshooting

The main pipeline handles model training and synthetic data generation:

```bash
<<<<<<< HEAD
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

## Toy Data Generation

For testing and experimentation purposes, you can generate synthetic toy datasets that mimic the structure of real peptide data:

```bash
# Generate toy data with default parameters (2000 samples, 30 peptide columns)
python src/data/make_toy_data.py

# Or import the function in your own scripts
=======
# Run with default configuration (Gaussian Copula)
python src/run_pipeline.py

# Override model and event type via CLI
python src/run_pipeline.py model=gan event=hf
python src/run_pipeline.py model=arf event=ckd
```

**Key pipeline configuration** (`configs/training_and_inference_pipeline/pipeline.yaml`):
```yaml
run_training: true
run_inference: true
run_evaluation: false

event: no_event        # Options: no_event | hf | ckd | all
sampled_patients_num: 0  # 0 = use all patients

experiment_name: "Gaussian Copula - V2"
run_name: "NE"
```

### 2. Full Evaluation Pipeline

Runs fidelity and privacy assessment on a real/synthetic dataset pair:

```bash
python src/evaluation/data_evaluation_script.py
```

**Key evaluation configuration** (`configs/evaluation/evaluate.yaml`):
```yaml
num_threads: 8
dataset_size_limit: 0    # 0 = no limit

fidelity:
  enabled: true
  # marginal, correlation, joint, classifier, effect_size, sparse_peptide

privacy:
  enabled: true
  # dcr, authenticity, reidentification, membership_inference, anonymeter
```

### 3. Standalone Privacy Evaluation

```bash
python src/evaluation/privacy/estimate_privacy.py
```

**Privacy configuration** (`configs/evaluation/privacy_script.yaml`):
```yaml
experiment_name: "Authenticity_Estimation"
zero_percentage: 0.4       # Peptide sparsity threshold
```

### 4. Hyperparameter Tuning (CTGAN)

Uses Optuna to tune CTGAN on a composite fidelity objective (KS + two-sample AUC):

```bash
python scripts/tune_ctgan.py ++n_trials=30 event=ckd
```

### 5. Data Reading

Read and transpose raw Excel peptide tables into a unified Polars DataFrame:

```bash
python scripts/read_data.py
```

## Available Models

### 1. Gaussian Copula (default)
- **Config**: `configs/training_and_inference_pipeline/model/gaussian_copula.yaml`
- Flexible marginal distributions (Beta, Normal, Log-normal, Truncated Normal, Gamma, Student-t, Uniform, Exponential)
- Distribution selection via KL divergence, BIC, AIC, or KS test
- Zero-inflated variants for sparse peptide columns
- Nearest positive semi-definite matrix repair
- Pearson, Kendall, or Spearman correlation modes

### 2. CTGAN
- **Config**: `configs/training_and_inference_pipeline/model/gan.yaml`
- Deep GAN-based tabular synthesis (SDV)
- QuantileTransformer pre-processing
- Supports Optuna-based hyperparameter tuning via `scripts/tune_ctgan.py`

### 3. Adversarial Random Forest (ARF)
- **Config**: `configs/training_and_inference_pipeline/model/arf.yaml`
- Tree-based adversarial training for density estimation (FORDE) and synthesis (FORGE)
- Decision tree boundary helpers in `src/models/AdversarialRandomForests/utils.py`

### 4. Histogram Imputation
- **Location**: `src/models/Imputation/HistogramImputation.py`
- Bin-based generation for sparse/zero-inflated columns
- Uniform or median bin-centre sampling strategies

All models implement `SynthetizationModelInterface` and log parameters and artifacts to MLflow automatically.

## Evaluation Metrics

### Fidelity (Data Quality)

| Category | Metrics |
|----------|---------|
| **Marginal** | Kolmogorov-Smirnov, Total Variation Distance, Wasserstein, Hellinger |
| **Correlation** | Frobenius norm, MACE distance |
| **Joint** | Maximum Mean Discrepancy (MMD), two-sample classifier test |
| **Effect Size** | Cohen's d, Hellinger, Wasserstein, CLES, rank-biserial |
| **Sparse Peptide** | Special handling for zero-inflated distributions |
| **Correlation Uncertainty** | Bootstrap confidence intervals (200–1000 resamples) |

### Privacy (Risk Assessment)

| Attack | Method |
|--------|--------|
| **Distance to Closest Record (DCR)** | Minimum synthetic-to-real distance vs. real-to-real holdout baseline |
| **Authenticity** | KNN-based ratio (d₁ NN in real vs. d₂ 2nd NN); suspicious sample detection + feature importance |
| **Re-identification** | Gap-ratio approach (d₂/d₁) for linking synthetic to originals |
| **Anonymeter Attacks** | Giomi et al. (2023): singling out, linkability, attribute inference |
| **Membership Inference** | Attack signal on synthetic vs. real holdout splits |

### Analysis Modules
- **Single Peptide**: Mann-Whitney U tests, eGFR score analysis, distribution comparisons
- **Multi-Peptide**: Kaplan-Meier survival analysis, time-to-event predictions, clinical correlations
- **Projections**: PCA and UMAP dimensionality reduction for visual inspection

## Toy Data Generation

Generate synthetic toy datasets that mimic the structure of real peptide data for testing:

```bash
python src/data/make_toy_data.py

# Or import directly
>>>>>>> troubleshooting
from src.data.make_toy_data import make_toy_data
toy_data = make_toy_data(num_samples=1000, num_peptide_cols=50)
```

<<<<<<< HEAD
### Toy Data Features
- **Clinical columns**: Includes all categorical and numerical clinical variables
- **Time-to-event columns**: Synthetic survival analysis data
- **Peptide columns**: Configurable number of synthetic peptide measurements
- **Realistic correlations**: Uses Gaussian copula with random marginal distributions
- **Statistical diversity**: Employs gamma, lognormal, and exponential distributions

The generated toy data maintains the same structure as real datasets, making it ideal for:
- Testing pipeline functionality
- Developing new features
- Demonstrating the system without sensitive data
- Performance benchmarking

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

### 4. Histogram Imputation
- **Location**: `src/models/Imputation/HistogramImputation.py`
- **Features**: Histogram-based data imputation and generation using bin-wise probability distributions. 

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

The project uses MLflow for experiment tracking with automatic local server setup:

```yaml
# MLflow configuration
=======
Pre-generated demo data is available at `resources/toy_data.csv`. Running the pipeline with default config on toy data takes under 20 seconds and produces a UUID-named CSV in `outputs/`.

### Expected Output Columns

**Clinical:**
- `idAuswertung`, `Geschlecht (1=female)`, `Kidney disease`, `diabetes`, `CVD`, `hypertension`
- `Blutdruck, diastolischM`, `Blutdruck, systolischM`, `GFR_CKD_EPI_M`, `BMI_M`, `Alter_M`
- `time-to-CKDevent(to event or last visit)`, `FU duration_CAD_Hfevent (to event or last visit)`

**Peptide:** `Patient ID`, `Peptide_1` … `Peptide_N`

## MLflow Integration

The project uses MLflow for experiment tracking:

```yaml
>>>>>>> troubleshooting
mlflow:
  tracking_uri: "http://10.100.111.210:5002"
  experiment_name: ${experiment_name}
```

<<<<<<< HEAD
### Automatic Local MLflow Server

The system automatically handles MLflow server connectivity:

1. **Remote Server**: If the configured `tracking_uri` is accessible, it connects to the remote MLflow server
2. **Local Fallback**: If the remote server is unavailable, it automatically launches a local MLflow server at `http://127.0.0.1:5000`

The local MLflow server is automatically started when needed and provides a shutdown hook for clean termination. You can view your experiments at the displayed URL during execution.

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

## Technical Documentation

The `resources/` directory contains detailed technical reports on the project components:

- **ARF_and_GAN_report.pdf**: Comprehensive analysis of Adversarial Random Forest and GAN model implementations, including performance comparisons and technical specifications
- **Copulas_report.pdf**: Detailed documentation of the Gaussian Copula implementation, marginal distribution selection, and mathematical foundations
- **Privacy_Appendix.pdf**: In-depth methodology for privacy evaluation, authenticity assessment techniques, and privacy-preserving synthetic data generation


=======
If the remote server is unreachable, a local MLflow server starts automatically at `http://127.0.0.1:5000`. A shutdown hook cleans it up on exit.

**Logged artifacts include:**
- Model parameters and hyperparameters
- Synthetic data generation statistics
- All fidelity and privacy metric scores
- PCA / UMAP / distribution / authenticity figures
- Suspicious sample reports with feature importance

## Output Files

### Training Pipeline
- `synthetic_{event}_sampled_{n}.csv` — generated synthetic dataset
- `outputs/{date}/{time}/run_pipeline.log` — execution log
- MLflow: model parameters, dataset metadata

### Evaluation Pipeline
- Fidelity report: statistical test results, classifier AUCs, correlation analysis
- Privacy report: DCR scores, authenticity assessments, re-identification risk, anonymeter results
- Figures: distribution plots, PCA/UMAP projections, authenticity histograms
- Suspicious sample lists with per-feature importance rankings

## Technical Documentation

The `resources/` directory contains technical reports:

- **ARF_and_GAN_report.pdf** — Adversarial Random Forest and GAN model analysis, performance comparisons
- **Copulas_report.pdf** — Gaussian Copula implementation, marginal distribution selection, mathematical foundations
- **Privacy_Appendix.pdf** — Privacy evaluation methodology, authenticity assessment, and privacy-preserving generation techniques
>>>>>>> troubleshooting

## License

See LICENSE file for details.
<<<<<<< HEAD

[//]: ## (## Bootstraping:)
 
[//]: ## (If bootstraping is enabled in `configuration.yaml`, script will evaluate the similarity between synthetic and original peptide datasets through a bootstrapping procedure:)

[//]: ## (1. **Data Preparation**: Loads the datasets and selects columns with sufficient non-zero values for analysis.)

[//]: ## (2. **Bootstrapping**: Iteratively samples subsets of synthetic data, comparing them to the original dataset using two metrics:)

[//]: ## (**Kolmogorov-Smirnov  p-value**: Measures distributional similarity.)


[//]: ## (- **Kullback-Leibler Divergence**: Quantifies differences in probability distributions.)

[//]: ## ( 3. **Best Subset Selection**: Identifies the subset with the lowest KS results and improved KL divergence.)

[//]: ## (4. **Output**:)

[//]: ##    (- Saves the refined synthetic peptide subset, corresponding clinical data, and statistical metrics for documentation.)

 
[//]: ## (This process ensures the synthetic dataset closely mirrors the original in key statistical properties.)

 
## Demo data

In `/resources` directory, demo data can be seen in a file called `toy_data.csv`. This data can be used for testing the provided code. When running the main script with the default configuration, the code should run for no more than 20 seconds and after that, your `/output` directory should contain generated csv named with a unique identifier id (uuid).
 
The synthesized data csv should containt the following columns:

#### Clinical data:
  - idAuswertung
  - "Geschlecht (1=female)"
  - "Kidney disease"
  - "diabetes"
  - "CVD"
  - "hypertension"
  - "Blutdruck, diastolischM"
  - "Blutdruck, systolischM"
  - "GFR_CKD_EPI_M"
  - "BMI_M"
  - "Alter_M"
  - "time-to-CKDevent(to event or last visit)"
  - "FU duration_CAD_Hfevent (to event or last visit)"

#### Peptide data:
- Patient ID
- Peptide_1
- Peptide_2
- Peptide_3
- Peptide_4
- Peptide_5
- Peptide_6
- Peptide_7
- Peptide_8
- Peptide_9
- Peptide_10
- Peptide_11
- Peptide_12
- Peptide_13
- Peptide_14
- Peptide_15...
=======
>>>>>>> troubleshooting

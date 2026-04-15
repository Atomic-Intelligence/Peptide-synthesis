# Peptide Synthesis Research

A comprehensive pipeline for synthetic peptide data generation and evaluation using machine learning models, with a focus on heart failure and chronic kidney disease biomarker analysis and advanced privacy risk assessment.

## Overview

This project provides a complete workflow for:
- Training synthetic data generation models on real peptide datasets
- Generating synthetic peptide data that preserves clinical characteristics
- Evaluating data quality (fidelity) through multiple statistical and ML-based metrics
- Assessing privacy risk through state-of-the-art attack simulations
- Hyperparameter tuning for generative models

A legacy version from an early project iteration can be found at `v0`.

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
```

## Running the Pipeline

### 1. Training and Inference

The main pipeline handles model training and synthetic data generation:

```bash
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
from src.data.make_toy_data import make_toy_data
toy_data = make_toy_data(num_samples=1000, num_peptide_cols=50)
```

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
mlflow:
  tracking_uri: "http://10.100.111.210:5002"
  experiment_name: ${experiment_name}
```

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

## License

See LICENSE file for details.

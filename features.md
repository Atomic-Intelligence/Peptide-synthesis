# Features Plan

This document outlines the planned integration of four new features into the Peptide-synthesis evaluation pipeline.

---

## Feature 1: Correlation Uncertainty Estimation

### Goal

Quantify the noise/uncertainty in Pearson and Spearman correlations between a user-specified set of columns. This is useful to determine which pairwise correlations in the synthetic data are reliably captured versus which ones are noisy artifacts of the generation process.

### Motivation

The Gaussian Copula model captures pairwise correlations between columns in its correlation matrix. However, the estimated correlations have statistical uncertainty depending on sample size, marginal distribution shape, and the presence of censored or zero-inflated values (common in peptide data). Without quantifying this uncertainty, it is unclear whether a given synthetic–real correlation difference is a model failure or a sampling artefact.

### Design

#### Location
`src/evaluation/analysis/correlation_uncertainty.py`
Config: `configs/evaluation/correlation_uncertainty.yaml`

#### Inputs
- Real dataset (Polars DataFrame)
- Synthetic dataset (Polars DataFrame)
- A list of column names to include in the analysis (e.g. a subset of peptide columns or clinical variables)
- Optional: correlation method (`pearson` or `spearman`), number of bootstrap iterations, confidence level

#### Method: Bootstrapped Correlation Confidence Intervals

1. **Bootstrap on real data**: Resample the real dataset with replacement `B` times (default B=1000). For each bootstrap sample, compute the full pairwise correlation matrix over the specified columns. This yields a distribution of correlation estimates for each pair.
2. **Bootstrap on synthetic data**: Same procedure applied to the synthetic dataset.
3. **Confidence intervals**: From the bootstrap distributions, compute the `alpha/2` and `1 - alpha/2` percentiles for each correlation pair (default: 95% CI).
4. **Overlap check**: For each column pair, determine whether the real and synthetic 95% CIs overlap. Non-overlapping intervals flag correlations where the synthetic model is statistically failing.
5. **Fisher z-transform**: Apply Fisher's z-transform before computing CIs to ensure valid interval bounds for correlation values, then back-transform.

#### Outputs
- `CorrelationUncertaintyResults` dataclass containing:
  - Real correlation matrix (mean over bootstrap)
  - Synthetic correlation matrix (mean over bootstrap)
  - Lower and upper CI bounds for both (stored as DataFrames indexed by column pairs)
  - Boolean mask: which pairs have non-overlapping CIs
  - Summary table: column pair | real corr | real CI | synth corr | synth CI | overlap

#### Visualisations
- **Heatmap of real vs synthetic correlations** with error bars (or separate upper/lower bound heatmaps)
- **Scatter plot** of real correlation vs synthetic correlation for all pairs, with CI error bars drawn on both axes; points outside diagonal tolerance band highlighted
- **Interval overlap heatmap**: binary grid showing which pairs diverge

#### Integration
- Call from `data_evaluation_script.py` as an independent parallel task (analogous to how PCA and UMAP are dispatched)
- Log results to MLflow: summary table as an artifact, heatmaps as figures
- Config exposes: `columns` (list), `method`, `n_bootstrap`, `confidence_level`, `zero_percentage_filter` (skip columns with too many zeros)

#### Config Structure (`correlation_uncertainty.yaml`)
```yaml
method: spearman          # pearson or spearman
n_bootstrap: 1000
confidence_level: 0.95
columns: []               # empty = use all peptide columns
max_columns: 50           # cap to avoid O(n^2) explosion
```

---

## Feature 2: DCR Privacy Test and Membership Inference Attack

### Goal

Extend the existing privacy module with two additional, widely-used privacy metrics:
1. **Distance to Closest Record (DCR)**: Measures how close each synthetic sample is to the nearest real sample in feature space.
2. **Membership Inference Attack (MIA)**: Assesses whether an adversary can determine whether a particular real record was used in training the generative model.

### Motivation

The existing `AuthenticityEstimator` uses a KNN ratio approach. DCR and MIA are standard complementary benchmarks used in privacy auditing of synthetic data (e.g. in SDV, MOSTLY AI, and academic literature). DCR gives an absolute proximity measure; MIA gives a worst-case attacker perspective.

### Design

#### Location
`src/evaluation/privacy/dcr.py`
`src/evaluation/privacy/membership_inference.py`
Both integrated under the existing `estimate_privacy.py` orchestrator.

---

### 2a: Distance to Closest Record (DCR)

#### Method

1. **Preprocessing**: Apply the same `FeatureProcessor` pipeline already used in `AuthenticityEstimator` (RobustScaler + one-hot encoding of categoricals).
2. **Synthetic → Real DCR**: For each synthetic record, find its nearest neighbour in the real dataset. Record that distance. This is the DCR for synthetic data.
3. **Real → Real DCR (holdout baseline)**: Split the real dataset in half. For each record in the second half, find its nearest neighbour in the first half. This gives a baseline distribution of what "close" means within real data alone.
4. **DCR ratio**: `DCR_synth / DCR_real_holdout`. A ratio near 1.0 means synthetic records are as close to real records as real records are to each other—a good outcome. A ratio much less than 1.0 signals memorisation.
5. **Privacy at risk (PaR)**: Proportion of synthetic records whose DCR is below the 5th percentile of the real holdout DCR distribution. These are at-risk records.

#### Outputs
- Distribution of DCR values for synthetic and holdout real (for plotting)
- Median DCR ratio
- Privacy-at-risk proportion
- Index of synthetic records below threshold (for auditing)

#### Visualisations
- Overlapping KDE plot of DCR distributions (real holdout vs synthetic)
- Cumulative distribution function (CDF) comparison plot

---

### 2b: Membership Inference Attack (MIA)

#### Method

The shadow model / threshold attack approach:

1. **Reference set construction**:
   - **Members**: Real training records (those actually used to fit the generative model).
   - **Non-members**: A held-out set of real records not seen by the model.
   - **Synthetic set**: Samples generated by the model.

2. **Attack signal — DCR-based**: For each candidate record (member or non-member), compute its DCR to the synthetic dataset. Members that the model has memorised will have lower DCR to synthetic data (synthetic points cluster around them).

3. **Attack signal — likelihood-based (Gaussian Copula specific)**: Use the fitted copula model to compute the log-likelihood of each candidate record. Members should (on average) have higher likelihood than non-members.

4. **Attack classifier**: Train a simple logistic regression or threshold classifier on the attack signal(s) to distinguish members from non-members.

5. **Metrics**:
   - **Advantage**: `2 * (AUC - 0.5)`. An advantage of 0 is perfect privacy, 1.0 is complete leakage.
   - **True Positive Rate at low FPR** (TPR @ FPR=0.01): Worst-case attacker precision.
   - **AUC-ROC curve**

6. **Fallback when held-out real data is unavailable**: Use cross-validation—train the model on a fold, treat the excluded fold as non-members.

#### Outputs
- `MIAResults` dataclass: AUC, advantage, TPR@FPR thresholds, ROC curve data
- Per-record risk scores (for audit purposes)

#### Visualisations
- ROC curve with chance line
- Histogram of attack scores for members vs non-members

---

### Integration

- `estimate_privacy.py` updated to optionally run DCR and MIA alongside the existing `AuthenticityEstimator`
- New config keys in `privacy_script.yaml`:
  ```yaml
  run_dcr: true
  run_mia: true
  mia:
    holdout_fraction: 0.2    # fraction of real data held out as non-members
    attack_signal: dcr       # dcr | likelihood | both
  dcr:
    par_percentile: 5        # percentile threshold for privacy-at-risk
  ```
- All results logged to MLflow under the existing `Authenticity_Estimation` experiment

---

## Feature 3: Fidelity Metrics Module

### Goal

Create a dedicated, self-contained module `src/evaluation/fidelity/` that computes standard statistical fidelity metrics comparing synthetic to real data. This consolidates and extends the existing marginal/correlation analysis scattered across `analysis/` into a systematic, reproducible benchmark.

### Motivation

Currently, fidelity assessment is spread across `single_peptide_analysis.py` (marginals, Mann-Whitney), `multi_peptide_analysis.py` (survival), and `projections/`. A dedicated fidelity module with a unified interface will make it easier to track overall fidelity scores across model runs via MLflow and to add new metrics independently.

### Design

#### Location
```
src/evaluation/fidelity/
    __init__.py
    marginal_fidelity.py
    correlation_fidelity.py
    joint_fidelity.py
    fidelity_report.py        # orchestrator
```
Config: `configs/evaluation/fidelity.yaml`

---

### 3a: Marginal Fidelity (`marginal_fidelity.py`)

Compares the univariate distribution of each column between real and synthetic data.

**Metrics per column:**
- **Kolmogorov-Smirnov (KS) statistic + p-value**: Maximum absolute difference between empirical CDFs.
- **Total Variation Distance (TVD)**: For categorical columns, half the L1 distance between probability vectors.
- **Wasserstein-1 distance (Earth Mover's Distance)**: For continuous columns, a transport-distance measure that respects the geometry of the real line and is more informative than KS for shape comparisons.
- **Mean and standard deviation ratio**: Simple sanity check (synthetic mean / real mean, synthetic std / real std).
- **Zero proportion difference**: Percentage of zeros in real vs synthetic (relevant for peptide data).

**Aggregates:**
- Mean KS across all peptide columns
- Mean Wasserstein across all peptide columns
- Proportion of columns with KS p-value < 0.05 (significantly different marginals)

---

### 3b: Correlation Fidelity (`correlation_fidelity.py`)

Compares the pairwise correlation structure.

**Metrics:**
- **Frobenius norm of the correlation matrix difference**: `||C_real - C_synth||_F`. A single scalar summarising how much the entire correlation structure differs.
- **Maximum absolute correlation error**: `max |C_real[i,j] - C_synth[i,j]|` — identifies the worst-case pair.
- **Mean absolute correlation error (MACE)**: Average over all pairs.
- **Correlation matrix heatmap**: Side-by-side real vs synthetic with a difference matrix.

---

### 3c: Joint / Multivariate Fidelity (`joint_fidelity.py`)

Assesses the joint distribution rather than only marginals or pairwise dependencies.

**Metrics:**
- **Maximum Mean Discrepancy (MMD)**: A kernel-based two-sample test statistic that measures the distance between real and synthetic distributions in a reproducing kernel Hilbert space. Use an RBF kernel with median heuristic bandwidth. Lower MMD = higher fidelity.
- **Precision and Recall for distributions** (from Kynkäänniemi et al. 2019):
  - **Precision**: Fraction of synthetic samples that fall within the support of real data (high precision = no mode hallucination).
  - **Recall**: Fraction of real data modes covered by synthetic data (high recall = no mode dropping).
  - Implemented via KNN manifold estimation (k=5 by default).
- **Density and Coverage** (Naeem et al. 2020): Variants of Precision/Recall that are more robust to outliers. Coverage measures the fraction of real data balls that contain at least one synthetic point.

---

### 3d: Fidelity Report Orchestrator (`fidelity_report.py`)

**`FidelityReport` class:**
- Accepts real DataFrame, synthetic DataFrame, column configuration
- Calls each sub-module in sequence (or in parallel using `concurrent.futures`)
- Collects all metrics into a single `FidelityResults` dataclass
- `summary()` method: returns a flat dict of headline metrics (suitable for MLflow `log_metrics`)
- `to_json()` method: serialises full results for artifact logging
- `plot_all()` method: generates all standard plots and returns a dict of matplotlib figures

**Headline metrics (logged to MLflow as scalars):**
- `fidelity/mean_ks_statistic`
- `fidelity/mean_wasserstein`
- `fidelity/correlation_frobenius`
- `fidelity/mmd`
- `fidelity/precision`
- `fidelity/recall`
- `fidelity/coverage`

**Config (`fidelity.yaml`):**
```yaml
marginal:
  enabled: true
  metrics: [ks, wasserstein, tvd]
correlation:
  enabled: true
  method: spearman
joint:
  enabled: true
  mmd_kernel: rbf
  precision_recall_k: 5
columns:
  exclude: []           # columns to skip
  max_correlation_cols: 60
```

#### Integration
- `data_evaluation_script.py` adds a `run_fidelity_report()` task dispatched in parallel with PCA/UMAP
- Results logged to MLflow: headline scalars + JSON artifact + all figures

---

## Feature 4: Two-Sample Classifier Test

### Goal

Add a **two-sample classifier test** (also called the "train-on-synthetic, classify-real-vs-synthetic" test) as a holistic fidelity metric. A classifier is trained to discriminate real from synthetic samples; if it cannot do better than chance, the synthetic data is high fidelity.

### Motivation

The existing classifiers in `src/evaluation/classifiers/` assess **machine learning efficiency** (train on one domain, predict a clinical outcome in the other). The two-sample classifier test is a different concept: it evaluates **distributional similarity** by treating the discrimination problem itself as the metric, requiring no outcome labels.

### Design

#### Location
`src/evaluation/fidelity/classifier_test.py` (sits within the new fidelity module)
Alternatively: `src/evaluation/classifiers/two_sample_classifier.py` if kept with the classifier code.

#### Method

1. **Data preparation**:
   - Label real samples as class 0, synthetic samples as class 1.
   - Concatenate into a single DataFrame.
   - Apply `FeatureProcessor` from the privacy module (RobustScaler + one-hot encoding).

2. **Cross-validated classification**:
   - Use stratified k-fold CV (default k=5).
   - Train a binary classifier in each fold.
   - Classifier options (configurable): Gradient Boosted Trees (XGBoost/LightGBM), Random Forest, Logistic Regression. Default: Random Forest (fast, non-parametric, handles mixed feature types well after encoding).

3. **Metrics**:
   - **AUC-ROC**: Primary metric. AUC = 0.5 means perfect indistinguishability; AUC = 1.0 means the classifier perfectly separates real from synthetic.
   - **Classifier accuracy** and **F1 score**: Secondary.
   - **Feature importance**: Which features most help the classifier distinguish real from synthetic. This pinpoints where the generative model is failing.
   - **Detection rate** (`1 - specificity at 50% threshold`): A practical measure of how many synthetic samples are flagged as fake.

4. **Interpretation scale** (to be documented in the report):
   - AUC 0.50–0.55: Excellent fidelity
   - AUC 0.55–0.65: Good fidelity
   - AUC 0.65–0.80: Moderate — review feature importances
   - AUC > 0.80: Poor fidelity — synthetic data is distinguishable

5. **Permutation baseline**: Optionally run the classifier on a permuted label assignment (all-real vs shuffled-real) to confirm the CV AUC is calibrated near 0.5 under the null hypothesis.

#### Outputs
- `TwoSampleClassifierResults` dataclass:
  - Per-fold AUC scores + mean and std
  - Mean accuracy, F1
  - Feature importances (sorted)
  - ROC curve (mean ± std over folds)

#### Visualisations
- ROC curve with fold-level shading and chance line
- Bar chart of top-N feature importances
- Calibration plot (optional)

#### Config (`classifier_test.yaml` or section within `fidelity.yaml`)
```yaml
two_sample_classifier:
  enabled: true
  classifier: random_forest    # random_forest | logistic_regression | gradient_boosted
  n_folds: 5
  n_estimators: 200            # for tree-based classifiers
  top_n_features: 20           # feature importances to display
  permutation_baseline: false
```

#### Integration
- Invoked as part of `FidelityReport.plot_all()` / `fidelity_report.py` orchestrator
- AUC logged to MLflow as `fidelity/two_sample_auc`
- Feature importance table saved as MLflow artifact
- Can also be run standalone from `data_evaluation_script.py` as an independent parallel task

---

## Implementation Order

The features have the following dependencies and recommended build order:

| Step | Feature | Depends on |
|------|---------|------------|
| 1 | Shared `FeatureProcessor` refactor — move out of `AuthenticityEstimator` into a shared `privacy/preprocessing.py` so DCR, MIA, and classifier test can all reuse it | Nothing |
| 2 | DCR (`dcr.py`) | Step 1 |
| 3 | MIA (`membership_inference.py`) | Step 1, Step 2 |
| 4 | Marginal + Correlation fidelity (`marginal_fidelity.py`, `correlation_fidelity.py`) | Nothing |
| 5 | Correlation uncertainty (`correlation_uncertainty.py`) | Step 4 (shares correlation computation) |
| 6 | Joint fidelity / MMD + Precision-Recall (`joint_fidelity.py`) | Step 1 |
| 7 | Two-sample classifier test (`classifier_test.py`) | Step 1, Step 4 |
| 8 | `FidelityReport` orchestrator (`fidelity_report.py`) | Steps 4–7 |
| 9 | Integration into `data_evaluation_script.py` and `estimate_privacy.py` | All above |
| 10 | MLflow config and YAML additions | All above |

---

## New Files Summary

```
src/evaluation/
    fidelity/
        __init__.py
        marginal_fidelity.py          # Feature 3a
        correlation_fidelity.py       # Feature 3b
        joint_fidelity.py             # Feature 3c — MMD, Precision, Recall, Coverage
        classifier_test.py            # Feature 4
        fidelity_report.py            # Feature 3d — orchestrator for all fidelity metrics
    analysis/
        correlation_uncertainty.py    # Feature 1
    privacy/
        dcr.py                        # Feature 2a
        membership_inference.py       # Feature 2b
        preprocessing.py              # Refactored FeatureProcessor (shared utility)

configs/evaluation/
    fidelity.yaml                     # Config for Features 3 & 4
    correlation_uncertainty.yaml      # Config for Feature 1
    # privacy_script.yaml updated with DCR and MIA keys
```

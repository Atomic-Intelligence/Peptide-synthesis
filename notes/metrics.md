# Evaluation Metrics Reference

Ranges for interpreting synthetic data evaluation metrics. Each metric is categorised as **Excellent**, **Average**, or **Poor**.

---

## Fidelity Metrics

### Marginal (Univariate) Fidelity

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| Mean KS Statistic | < 0.05 | 0.05 -- 0.15 | > 0.15 |
| Fraction KS Significant | < 0.10 | 0.10 -- 0.40 | > 0.40 |
| Mean Wasserstein (normalised) | < 0.02 | 0.02 -- 0.10 | > 0.10 |
| Mean/Std Ratio | 0.95 -- 1.05 | 0.85 -- 0.95 or 1.05 -- 1.15 | < 0.85 or > 1.15 |
| Mean TVD (categorical) | < 0.05 | 0.05 -- 0.15 | > 0.15 |

### Correlation Fidelity

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| MACE (Mean Abs Correlation Error) | < 0.05 | 0.05 -- 0.15 | > 0.15 |
| Max Absolute Correlation Error | < 0.10 | 0.10 -- 0.25 | > 0.25 |
| Frobenius Norm | < 1.0 | 1.0 -- 3.0 | > 3.0 |
| Fraction CI Overlap | > 0.90 | 0.70 -- 0.90 | < 0.70 |

### Joint (Multivariate) Fidelity

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| MMD | < 0.01 | 0.01 -- 0.10 | > 0.10 |
| Precision | > 0.90 | 0.70 -- 0.90 | < 0.70 |
| Recall | > 0.90 | 0.70 -- 0.90 | < 0.70 |
| Coverage | > 0.90 | 0.70 -- 0.90 | < 0.70 |

### Two-Sample Classifier Test

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| AUC-ROC | 0.50 -- 0.55 | 0.55 -- 0.65 | > 0.65 |
| Mean Accuracy | 0.50 -- 0.55 | 0.55 -- 0.65 | > 0.65 |
| Mean F1 | 0.50 -- 0.55 | 0.55 -- 0.65 | > 0.65 |

*Closer to 0.5 = real and synthetic are indistinguishable.*

### Effect Size (Per-Column Divergence)

**Continuous columns:**

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| Cohen's d (mean abs) | < 0.10 | 0.10 -- 0.30 | > 0.30 |
| Rank-Biserial / Cliff's delta (mean abs) | < 0.10 | 0.10 -- 0.30 | > 0.30 |
| Median Absolute Shift (mean abs) | < 0.10 | 0.10 -- 0.30 | > 0.30 |
| CLES | 0.48 -- 0.52 | 0.45 -- 0.48 or 0.52 -- 0.55 | < 0.45 or > 0.55 |
| Overlap Coefficient | > 0.90 | 0.75 -- 0.90 | < 0.75 |
| Normalised Wasserstein (mean abs) | < 0.05 | 0.05 -- 0.15 | > 0.15 |

**Categorical columns:**

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| Hellinger Distance | < 0.05 | 0.05 -- 0.15 | > 0.15 |
| Jensen-Shannon Divergence | < 0.05 | 0.05 -- 0.15 | > 0.15 |

### Sparse Peptide Fidelity

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| Zero Fraction Difference | < 0.03 | 0.03 -- 0.10 | > 0.10 |
| Divergence Score | < 0.10 | 0.10 -- 0.25 | > 0.25 |

---

## Privacy Metrics

### Distance to Closest Record (DCR)

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| Median DCR Ratio | > 1.0 | 0.5 -- 1.0 | < 0.5 |
| Privacy-at-Risk (PAR) | < 0.05 | 0.05 -- 0.15 | > 0.15 |

*DCR Ratio = median(synth-to-real distance) / median(holdout-to-real distance). Values >= 1 mean synthetic records are no closer to real data than real records are to each other.*

### Authenticity

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| Mean Binary Authenticity Score | > 0.90 | 0.70 -- 0.90 | < 0.70 |
| Mean Authenticity Ratio | 0.90 -- 1.10 | 0.70 -- 0.90 or 1.10 -- 1.50 | < 0.70 or > 1.50 |

*Score near 1.0 means synthetic samples are plausible members of the real distribution without memorising specific records.*

### Membership Inference Attack (MIA)

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| AUC-ROC | 0.50 -- 0.55 | 0.55 -- 0.65 | > 0.65 |
| Membership Advantage | < 0.10 | 0.10 -- 0.30 | > 0.30 |
| TPR @ FPR=0.01 | < 0.02 | 0.02 -- 0.10 | > 0.10 |

*AUC near 0.5 = attacker cannot distinguish training members from non-members (strong privacy).*

### Re-identification Risk

| Metric | Excellent | Average | Poor |
|---|---|---|---|
| Mean Distance Ratio | > 0.80 | 0.50 -- 0.80 | < 0.50 |
| Re-identification Rate | < 0.05 | 0.05 -- 0.15 | > 0.15 |

*Distance ratio close to 1 means 1st and 2nd nearest real neighbours are equidistant (low re-identification risk). Re-id rate is the fraction of synthetic records with ratio below the risk threshold (default 0.5).*

# Membership Inference Attack — Module Explanation

## What Is a Membership Inference Attack?

A membership inference attack (MIA) answers a simple but privacy-critical question:

> **Was a specific record used to train the generative model?**

If an attacker can reliably answer "yes" for a given record, the model is leaking information about its training set. In the context of this project — peptide synthesis with generative models — that means the synthetic data is not sufficiently private: an adversary could determine whether a particular peptide sequence was part of the original dataset.

---

## How the Attack Works in This Module

The implementation in `src/evaluation/privacy/membership_inference.py` follows a **shadow-model-free, signal-based** approach. The pipeline has three stages: data splitting, signal computation, and classification.

### Stage 1 — Splitting Real Data into Members and Non-Members (`fit`)

The real dataset is randomly split into two groups:

| Group | Role | Default size |
|-------|------|--------------|
| **Members** | Simulate the training set — records the generative model "saw" | 80% of real data |
| **Non-members** | Held-out records the model never saw | 20% (`holdout_fraction`) |

Both groups are transformed into numeric feature arrays via a shared `FeatureProcessor` (scaling + one-hot encoding of categoricals). The processor is **fit on members only**, so the transformation reflects training-set statistics — this mirrors what the generative model itself learned from.

### Stage 2 — Computing Attack Signals (`estimate`)

For every record (member or non-member), the module computes one or more **attack signals** — numeric features that an attacker could observe and that are expected to differ between members and non-members.

#### Signal 1: DCR (Distance to Closest Record)

```
For each candidate record → find its nearest neighbour in the synthetic dataset
```

**Intuition:** A generative model that memorises training records will produce synthetic points that cluster near those memorised records. Therefore:

- **Members** (training records) will tend to have a **small** distance to their closest synthetic neighbour.
- **Non-members** (unseen records) will tend to have a **larger** distance.

Implementation detail: distances are **negated** (`-dcr`) so that a *higher* score means *more likely to be a member*, which aligns with the convention expected by the downstream classifier and AUC computation.

The nearest-neighbour search uses `sklearn.neighbors.NearestNeighbors` with a configurable algorithm (default: `ball_tree`).

#### Signal 2: Likelihood (optional)

When the generative model exposes a `log_prob` method (e.g. a Gaussian copula), the log-probability of each candidate record under the model is used as a signal.

**Intuition:** A model assigns higher likelihood to data it was trained on.

If the model does not support `log_prob`, this signal is silently skipped and DCR is used as the sole signal.

#### Signal 3: Both

When `attack_signal="both"`, DCR and likelihood are concatenated into a 2-dimensional feature vector per record, giving the classifier more information to work with.

### Stage 3 — Training the Attack Classifier

The attack signals are assembled into a feature matrix `X` with binary labels `y`:

```
X = [member_signals; nonmember_signals]    shape: (N_members + N_nonmembers, n_signals)
y = [1, 1, ..., 1, 0, 0, ..., 0]          1 = member, 0 = non-member
```

A **logistic regression** classifier (`sklearn.linear_model.LogisticRegression`) is trained on this data. It learns a decision boundary that separates members from non-members based on the attack signal(s).

After training, the classifier outputs `P(member | signal)` for every record — these are the **attack scores**.

> **Note:** The classifier is trained and evaluated on the *same* data (no train/test split of the attack dataset). This is intentional — the goal is to measure **how separable** the two populations are, not to estimate generalisation of the attack classifier. The AUC on this data is a *lower bound* on the privacy risk.

---

## Metrics — Interpreting the Results

The `MIAResults` dataclass reports three key numbers:

| Metric | Perfect privacy | Complete leakage | Meaning |
|--------|----------------|-------------------|---------|
| **AUC** | 0.5 | 1.0 | Area under the ROC curve. 0.5 means the attacker does no better than random guessing. |
| **Advantage** | 0.0 | 1.0 | `2 * (AUC - 0.5)`. Rescales AUC to a 0-1 "privacy leakage" scale. |
| **TPR @ FPR = 0.01** | ~0.01 | 1.0 | True positive rate when the false positive rate is capped at 1%. Measures worst-case attacker precision — "if the attacker is very conservative, how many members can they still identify?" |

### Visualisations (`plot`)

The module produces a two-panel figure:

1. **ROC Curve** — plots TPR vs FPR. A curve hugging the diagonal means good privacy; a curve bowing toward the top-left means the model is leaking membership.
2. **Attack Score Histogram** — overlays the distribution of `P(member)` scores for members (blue) and non-members (orange). Overlapping distributions = good privacy; separated distributions = leakage.

---

## Data Flow Summary

```
real_data
    │
    ├──[80%]──► members_df ──► feature_processor.fit_transform() ──► members_array
    │
    └──[20%]──► nonmembers_df ──► feature_processor.transform() ──► nonmembers_array

synthetic_data ──► feature_processor.transform() ──► synth_array

members_array    ──► nearest-neighbour distance to synth_array ──► negate ──► member_dcr_signal
nonmembers_array ──► nearest-neighbour distance to synth_array ──► negate ──► nonmember_dcr_signal

[member_dcr_signal, label=1]  ┐
                               ├──► LogisticRegression.fit() ──► attack_scores ──► AUC, advantage, TPR@FPR
[nonmember_dcr_signal, label=0]┘
```

---

## Why This Matters for Peptide Synthesis

Synthetic peptide datasets are meant to be shared or published without revealing the original proprietary sequences. If the MIA AUC is significantly above 0.5, the generative model has effectively memorised parts of the training set, and the synthetic data cannot be considered private. This metric provides a quantitative, reproducible check before any synthetic dataset is released.

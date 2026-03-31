# Membership Inference Attack — Integration Notes

## What was done

The `MembershipInferenceAttack` (MIA) class, already implemented in
`src/evaluation/privacy/membership_inference.py`, has been integrated into the
`PrivacyReport` orchestrator (`src/evaluation/privacy/privacy_report.py`) as a
first-class, configurable privacy metric alongside the existing DCR and
Authenticity estimators.

### Changes made

#### 1. `membership_inference.py` — shared FeatureProcessor support

`MembershipInferenceAttack.__init__` gained a `fitted_feature_processor`
parameter (mirrors the same pattern in `DCREstimator` and
`AuthenticityEstimator`).  When a pre-fitted processor is supplied, the class
skips its own `fit` step and calls only `transform()`.  This lets `PrivacyReport`
fit the feature processor once on the full real dataset and share it across all
three estimators, avoiding redundant scaler fitting.

#### 2. `privacy_report.py` — MIA integrated into orchestrator

- **`PrivacyResults` dataclass** gained an `mia: Optional[MIAResults]` field.
  Its `summary()` method now prefixes MIA scalars with `"privacy/mia_"` so they
  slot into the existing MLflow metric namespace without collision.

- **`PrivacyReport.__init__`** accepts three new parameters:

  | Parameter | Type | Default | Meaning |
  |---|---|---|---|
  | `run_mia` | `bool` | `False` | Enable/disable the attack entirely |
  | `mia_holdout_fraction` | `float` | `0.2` | Fraction of real data held out as non-members |
  | `mia_attack_signal` | `str` | `"dcr"` | Signal used by the attack classifier (`"dcr"` \| `"likelihood"` \| `"both"`) |

  `run_mia` defaults to `False` to preserve the existing behaviour of any code
  that constructs `PrivacyReport` without the new parameters.

- **`PrivacyReport.run()`** runs MIA after DCR and Authenticity when `run_mia`
  is `True`, passing the shared `FeatureProcessor` so no extra fit is needed.
  Failures are caught and logged without aborting the other metrics.

---

## How the attack works

### Setup

The real dataset is randomly split into two groups:

- **Members** (1 − `holdout_fraction` of real data): the records that the
  generative model "saw" during training.
- **Non-members** (`holdout_fraction` of real data): records held back from the
  model, used as the negative class.

Both groups go through the shared `FeatureProcessor` (RobustScaler +
one-hot encoding) so distances are computed in a normalised feature space.

### Attack signal: DCR to synthetic

For each candidate record (member or non-member), the distance to its nearest
synthetic neighbour is computed with a 1-NN index fitted on the synthetic
dataset.

**Intuition**: a generative model tends to produce synthetic points that cluster
around the real records it memorised.  Members therefore have a *smaller*
distance to the synthetic dataset than non-members.  The negated DCR
(`-distance`) is used as the attack score so that a higher score means "more
likely a member".

When `attack_signal="likelihood"` or `"both"` and the generative model exposes a
`log_prob(df) -> np.ndarray` method, the log-likelihood of each candidate under
the fitted model is appended as a second signal (members typically have higher
likelihood).

### Attack classifier

A logistic regression (`max_iter=500, solver="lbfgs"`) is trained on the
concatenated attack signals to predict membership.  Because both groups use the
same data, this is an in-sample fit—the AUC measures the best-case attacker
advantage on the training set (worst-case privacy estimate for the data owner).

### Metrics

| Metric | Interpretation |
|---|---|
| `mia_auc` | AUC-ROC of the attack classifier.  0.5 = perfect privacy, 1.0 = complete leakage. |
| `mia_advantage` | `2 × (AUC − 0.5)`.  Range [0, 1]; 0 = no information gained. |
| `mia_tpr_at_fpr_001` | True positive rate when the false positive rate is ≤ 0.01.  Worst-case precision: how many true members the attacker recovers while falsely accusing only 1 % of non-members. |

### Visualisations (`MembershipInferenceAttack.plot`)

- **ROC curve** with AUC annotated and a chance diagonal.
- **Attack score histogram** overlaying the score distributions for members vs
  non-members; a well-separated bimodal distribution signals high leakage.

---

## Configuration

`privacy_script.yaml` already exposes the relevant keys:

```yaml
run_mia: true
mia:
  holdout_fraction: 0.2    # fraction of real data held out as non-members
  attack_signal: dcr       # dcr | likelihood | both
```

When constructing `PrivacyReport` directly in Python:

```python
report = PrivacyReport(
    categorical_columns=CATEGORICAL_CLINICAL_COLUMNS + ["event_type"],
    run_dcr=True,
    run_authenticity=True,
    run_mia=True,
    mia_holdout_fraction=0.2,
    mia_attack_signal="dcr",
)
results = report.run(real_df, synth_df)
print(results.summary())   # includes privacy/mia_auc, privacy/mia_advantage, etc.
```

---

## MLflow logging

`estimate_privacy.py` already logs MIA results under the
`Authenticity_Estimation` experiment:

```python
if mia_results is not None:
    mlflow.log_metrics(mia_results.summary())   # mia_auc, mia_advantage, mia_tpr_at_fpr_001
    mlflow.log_figure(mia_figure, "mia_results.png")
```

When running through `PrivacyReport`, call `results.summary()` which returns all
metrics prefixed `privacy/mia_*` for consistent namespacing with DCR and
Authenticity.

---

## Design decisions

- **`run_mia=False` by default** in `PrivacyReport`: avoids breaking existing
  call sites that construct the report without MIA intent.  `estimate_privacy.py`
  continues to read `run_mia` from the YAML config.
- **Shared FeatureProcessor**: fitting the scaler and encoder once on the full
  real dataset ensures that DCR, MIA, and Authenticity all operate in the same
  feature space, making their outputs directly comparable.
- **In-sample attack classifier**: the logistic regression is trained and
  evaluated on the same set.  This gives an upper bound on attacker advantage
  and is standard practice for MIA auditing (Carlini et al. 2022).  A
  cross-validated variant would reduce variance at the cost of complexity, and
  can be added later without interface changes.

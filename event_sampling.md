# Event Sampling — Current State & Improvement Ideas

## Current Approach

Data is split by `event_type` column (`"hf"`, `"ckd"`, `"no_event"`) at two stages:

1. **Pre-training sampling** (`sample_patients_with_preserved_ratios`): patients are sampled at the patient ID level, preserving the empirical event-type ratios in the full cohort.
2. **Training split** (`split_event_control`): the sampled dataset is split into an event group and a control group — but only the event group is passed to the copula. The control group is discarded (`_`).

**Result:** the copula learns the distribution of a single event class only, synthetic output is unlabeled, and the control group's information is wasted.

---

## Problem Summary

| Issue | Location | Impact |
|---|---|---|
| Control group discarded during training | `run_pipeline.py:50-51` | Copula biased toward one event class; smaller effective training set |
| `event_type` dropped before model sees it | `run_pipeline.py:68` | Synthetic rows have no event label |
| CAD filtered twice, unconditionally | `utils.py:101` + `utils.py:219` | CAD data never usable without code changes |
| Imputation data collected before sampling | `utils.py:158` | Imputation model sees a different distribution than the copula |
| Ratios preserved, but only within the sampled subset | `utils.py:226-260` | If `sampled_patients_num` is small, rare event classes drop to very few rows |
| No per-event-type conditional generation at inference | Entire pipeline | Cannot request "generate N HF patients" |
| Time-to-event columns treated as plain features | Entire pipeline | Synthetic survival times may be inconsistent with event status |

---

## Improvement Ideas

### 1. Train on the Full Dataset (Event + Control Together)

**What:** remove `split_event_control` from the training path. Treat `event_type` as just another categorical feature the copula must learn.

**Why it helps:** the model captures correlations across event classes (e.g., how peptide profiles differ between HF and CKD patients) and has access to the full sampled cohort, not just one slice.

**Implementation sketch:**
- Keep `event_type` in the dataframe passed to the copula.
- Remove the `split_event_control` call from `train_model()`.
- Update configs: change `event: no_event` → `event: null` / remove field.
- At inference time, optionally condition on `event_type` via `generate_conditional()`.

---

### 2. Per-Event-Type Copula Models

**What:** train one copula per event class and store them separately.

**Why it helps:** each event class may have a different correlation structure among peptides and clinical variables. A single joint model must learn to partition these — separate models each specialise.

**Implementation sketch:**
- In `train_model()`, loop over distinct event types, split, train, save a model per event.
- At inference time, generate N rows from each event-specific model and concatenate with the correct `event_type` label.
- Useful as an A/B comparison against Idea 1 — whichever produces higher fidelity wins.

**Trade-off:** requires more training runs and larger storage; rare events may have too few training samples.

---

### 3. Conditional Synthetic Generation at the Pipeline Level

**What:** expose a config knob (e.g., `synthetic_event_counts: {no_event: 1000, hf: 500, ckd: 300}`) that drives inference so the user can specify how many rows of each class to generate.

**Why it helps:** downstream evaluators often need balanced classes or specific ratios that differ from the real-data prevalence.

**Implementation sketch:**
- Extend inference config with per-event counts.
- If using Idea 1 (joint copula): call `generate_conditional(event_type="hf", n=500)` etc.
- If using Idea 2 (per-event copulas): call each model's `generate(n=N_event)`.
- Concatenate and shuffle before saving.

---

### 4. Minimum-Class Guardrail During Sampling

**What:** add a `min_patients_per_event` parameter to `sample_patients_with_preserved_ratios`. If the ratio-preserving count for any class falls below this threshold, either over-sample that class or warn the user.

**Why it helps:** with small `sampled_patients_num` (e.g., 200) and a rare event like CKD at 5%, only 10 rows reach the copula — not enough to learn a reliable marginal, let alone joint distribution.

**Options:**
- **Oversample:** inflate the rare class count to the minimum, accept that ratios are no longer exactly preserved.
- **Warn + skip:** emit a warning and skip training if any class is below the threshold.
- **SMOTE-style resampling:** generate synthetic augmentations from the rare class before copula training (bootstrap rows, add small Gaussian noise to continuous features).

---

### 5. Event-Aware Peptide Non-Zero Threshold Filtering

**What:** the current `filter_peptides` applies a single global non-zero threshold across the entire dataset. Some peptides may be informative for one event class but near-zero in the full cohort.

**Why it helps:** a peptide that is zero for 95% of no_event patients but measured for 60% of HF patients would be filtered out at `non_zero_threshold=0.1` if HF is a small fraction of the dataset.

**Implementation sketch:**
- Compute non-zero fraction per peptide **per event group**.
- Keep a peptide if it passes the threshold in **any** event group (union) rather than the full dataset (intersection is the current effective behaviour).
- Alternatively, expose `per_event_threshold: true` as a config flag.

---

### 6. Move Imputation Data Collection to After Sampling

**What:** `dfs_for_imputation` is currently populated inside `filter_peptides()` before `sample_patients_with_preserved_ratios()` runs. The histogram imputation model therefore trains on the full pre-sampling distribution.

**Why it matters:** if patients are sampled at a ratio different from the full cohort, the imputation model's zero-spike and tail estimates come from a different population than the copula's training data. This can introduce subtle distribution mismatch in sparse columns.

**Fix:** collect imputation data after sampling, or fit the histogram imputation inside `train_model()` on the same dataframe used for the copula.

---

### 7. Treat Time-to-Event Columns with Survival Awareness

**What:** `"FU duration_CAD_Hfevent"` and `"time-to-CKDevent"` are currently modelled as ordinary continuous features. They are right-censored survival times.

**Why it matters:** a no_event patient's time-to-event is the follow-up duration (censored), not a true event time. The copula encodes the correlation between event status and time without understanding censoring, which can produce synthetic rows where a no_event patient has a shorter "time-to-event" than an HF patient, or where the marginal distribution of survival times is wrong.

**Options (in increasing complexity):**
- **Stratified imputation:** fit separate marginals for event vs. control patients, so their time distributions are independently modelled.
- **Truncation:** clip synthetic survival times to plausible clinical ranges (e.g., 0–10 years).
- **Post-processing consistency check:** after generation, if `event_type == "no_event"` and `time-to-CKDevent < threshold`, resample that row.
- **Full survival copula:** replace the continuous marginal for time-to-event with a Kaplan-Meier or parametric survival marginal (Weibull) that respects censoring.

---

### 8. Make Event Filtering Configurable

**What:** replace the hardcoded `filter(pl.col(event_col) != "cad")` with a config field, e.g. `exclude_event_types: ["cad"]`.

**Why:** CAD patients cannot currently be used without modifying source code. If CAD data quality improves, or if a user wants to investigate CAD, the current design requires a code change.

---

## Prioritisation

| Priority | Idea | Effort | Impact |
|---|---|---|---|
| High | 1 — Train on full dataset | Low | Fixes the biggest bias immediately |
| High | 6 — Move imputation collection | Low | Fixes distribution mismatch with one refactor |
| High | 8 — Configurable event filtering | Low | Removes hardcoded logic |
| Medium | 3 — Conditional generation config | Medium | Enables downstream use cases |
| Medium | 4 — Minimum-class guardrail | Low | Prevents silent failure on rare classes |
| Medium | 5 — Per-event peptide threshold | Medium | Recovers event-specific biomarkers |
| Low | 2 — Per-event copula models | High | Better specialisation but high complexity |
| Low | 7 — Survival-aware modelling | High | Correct but requires statistical machinery |

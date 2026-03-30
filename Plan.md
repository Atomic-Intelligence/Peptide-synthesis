# Fix Plan

This plan addresses every issue in `Troubleshooting.md`, organised into phases by severity and dependency order. Each fix includes the exact file, the change, and the rationale.

---

## Phase 1 -- Critical Bugs

These produce silently wrong results and should be fixed first.

### 1.1 Fix truncated Gaussian sampling (`preprocessing.py:132-134`)

**Problem:** The `stats.truncnorm.rvs(...)` result is immediately overwritten by `stats.uniform.rvs(...)`. Additionally, the `truncnorm` call passes raw bounds instead of standardised bounds `(bound - loc) / scale`.

**Fix:**
1. Remove the `stats.uniform.rvs(...)` line entirely.
2. Correct the `truncnorm` parameters. `scipy.stats.truncnorm` expects `a` and `b` as standardised bounds:
   ```python
   mean = (lower + upper) / 2
   std_dev = (upper - lower) / 4
   a_std = (lower - mean) / std_dev
   b_std = (upper - mean) / std_dev
   samples = stats.truncnorm.rvs(a=a_std, b=b_std, loc=mean, scale=std_dev, size=size)
   ```
3. Add a unit test that verifies: (a) all returned samples fall within `[lower, upper]`, (b) the distribution is approximately Gaussian (Shapiro-Wilk on a large sample), (c) edge case where `lower == upper` returns a constant.

---

### 1.2 Remove hard-coded `.head(100)` in evaluation (`data_evaluation_script.py:385-386`)

**Problem:** Evaluation always runs on the first 100 rows regardless of dataset size. All metrics, plots, and survival analyses are computed on a truncated dataset.

**Fix:**
1. Remove `.head(100)` from both lines:
   ```python
   real_dataset = pl.read_csv(cfg.paths.real_data_path)
   synthetic_dataset = pl.read_csv(cfg.paths.synthetic_data_path)
   ```
2. If a debug/quick-eval mode is desired, make it a config option (e.g. `eval_max_rows: null` in the eval config) rather than a hard-coded truncation.

---

### 1.3 Handle NaN in correlation matrix (`gaussian_copula_basic.py:282-288`)

**Problem:** NaN in the estimated correlation matrix is warned but not handled. Downstream, NaN propagates through the copula and gets silently replaced with 0.0 via `fill_null`.

**Fix:**
1. After computing the correlation matrix, replace NaN entries with 0.0 (zero correlation assumption for unmeasurable pairs) and log which column-pairs were affected:
   ```python
   if np.isnan(estimated_correlation).any():
       nan_mask = np.isnan(estimated_correlation)
       nan_pairs = list(zip(*np.where(nan_mask)))
       logger.warning(f"NaN in correlation matrix for {len(nan_pairs)} pairs. Replacing with 0.0.")
       estimated_correlation = np.nan_to_num(estimated_correlation, nan=0.0)
   ```
2. After the `corr_clipped` correction, add a positive-definiteness check (addresses issue 3.4 simultaneously):
   ```python
   eigenvalues = np.linalg.eigvalsh(corrected_correlation)
   if np.any(eigenvalues <= 0):
       logger.warning(f"Correlation matrix not PSD after clipping (min eigenvalue: {eigenvalues.min():.6f}). Using corr_nearest instead.")
       corrected_correlation = corr_nearest(estimated_correlation)
   ```
3. Import `corr_nearest` from `statsmodels.stats.correlation_tools`.

---

### 1.4 Guard `histogram_imputation_model` against None (`run_pipeline.py:130-136`)

**Problem:** If `run_training: false` and `run_inference: true`, `histogram_imputation_model` is `None`, causing an `AttributeError`.

**Fix:**
1. Add a guard before the inference call:
   ```python
   if cfg.run_inference:
       if histogram_imputation_model is None:
           raise ValueError(
               "run_inference=true but no histogram imputation model is available. "
               "Either set run_training=true or provide a pre-trained model path."
           )
       synthetic_data = run_inference(...)
   ```
2. Long-term: add model serialisation (pickle/joblib) so that a pre-trained model can be loaded from disk when `run_training=false`. This is a separate feature and can be deferred.

---

## Phase 2 -- Logic Errors

These produce incorrect but not immediately crashing behaviour.

### 2.1 Stop treating all integers as nominal categoricals (`gaussian_copula_basic.py:97-108`)

**Problem:** `convert_int_to_categorical` casts every `Int64` column to `Utf8`, treating continuous/ordinal columns like `Alter_M`, `BMI_M`, `GFR_CKD_EPI_M` as nominal categoricals.

**Fix:**
1. Add a config list `ordinal_int_columns` (or `categorical_int_columns` -- whichever list is shorter) to the model config.
2. Only cast columns explicitly listed as categorical. Leave continuous integer columns as numeric:
   ```python
   def convert_int_to_categorical(self, data: pl.DataFrame) -> pl.DataFrame:
       categorical_cols = set(self.cfg.get("categorical_int_columns", []))
       int_cols = [col for col in data.columns if data[col].dtype == pl.Int64 and col in categorical_cols]
       return data.with_columns([pl.col(c).cast(pl.Utf8) for c in int_cols])
   ```
3. Fix the misleading comment (`# Cast all integer columns to boolean` -> remove or rewrite).

---

### 2.2 Fix case-sensitive peptide matching (`gaussian_copula_basic.py:76`)

**Problem:** `cs.matches("Peptide")` is case-sensitive, but `DataProcessor.get_peptide_columns` uses `re.IGNORECASE`. Toy data uses lowercase `peptide_X`.

**Fix:**
Change the regex to be case-insensitive:
```python
target_col_names = data.select(cs.numeric() & cs.matches("(?i)peptide")).columns
```

---

### 2.3 Remove double CAD filtering (`utils.py:101`, `utils.py:219`)

**Problem:** CAD events are filtered in `DataProcessor.__init__` and again in `sample_patients_with_preserved_ratios`.

**Fix:**
1. Remove the unconditional CAD filter from `__init__` (line 101).
2. Make event-type filtering configurable. Add `exclude_event_types: [cad]` to the data processor config.
3. Apply the filter once, in `sample_patients_with_preserved_ratios` or in a dedicated `filter_events()` method, controlled by the config value.

---

### 2.4 Include control group in training (`run_pipeline.py:48-52`)

**Problem:** The `_` discards the control group. The model is trained only on event patients.

**Fix:**
This depends on the scientific intent. Two options:

- **Option A (train on both):** Remove the `split_event_control` call from the training path entirely. Train the copula on the full dataset including both event and control patients. The `event_type` column becomes just another feature in the copula.
- **Option B (train separate models):** Train one model per event type and generate from each. This requires pipeline restructuring.

Recommend **Option A** as the simpler, more correct default. The split should only happen at evaluation time.

```python
real_dataset = (
    data_processor.filter_peptides(non_zero_threshold=cfg.non_zero_threshold)
    .get_processed_data()
)
```

---

### 2.5 Update imputation data after sampling (`utils.py`, `run_pipeline.py`)

**Problem:** `dfs_for_imputation` is collected during `filter_peptides` (before sampling), so the histogram imputation model trains on a different distribution than the copula.

**Fix:**
Move the imputation data collection to after `sample_patients_with_preserved_ratios`. Modify `sample_patients_with_preserved_ratios` to also update `self.dfs_for_imputation` by filtering to only the sampled patients.

---

### 2.6 Change `event: no_event` default (`pipeline.yaml:11`)

**Problem:** Default trains on `no_event` patients only, producing single-class synthetic data.

**Fix:**
If we adopt fix 2.4 Option A (train on both groups), this config key becomes irrelevant for training. Keep it only for evaluation purposes. Update the default to reflect the intended use:
```yaml
event: null  # Train on all event types. Set to a specific type for evaluation filtering.
```

---

### 2.7 Fix `non_zero_threshold` unit inconsistency (`utils.py`)

**Problem:** `_filter_peptides_in_df` uses percentage (0-100), `split_peptide_columns_by_zero_percentage` uses fraction (0-1).

**Fix:**
Standardise on percentage (0-100) everywhere. In `split_peptide_columns_by_zero_percentage`, multiply the threshold or change the comparison:
```python
# Before: > zero_perc_threshold  (where default was 0.3)
# After:  > zero_perc_threshold  (where default is 30.0, matching the percentage convention)
```
Update the default value to `30.0` and rename the parameter to `zero_percentage_threshold` for clarity.

---

### 2.8 Make `idAuswertung` configurable (`synthetization_model_interface.py:54`)

**Problem:** A German column name is hard-coded in the abstract base class.

**Fix:**
Add a `patient_id_column` field to the model config (already partially present as `primary_key` in pipeline.yaml). Use it in the base class:
```python
id_col = getattr(self, 'primary_key', None) or self.cfg.get('primary_key', None)
if id_col and id_col in real_dataset.columns:
    real_dataset = real_dataset.drop(id_col)
```

---

## Phase 3 -- Evaluation & MLflow Fixes

### 3.1 Close MLflow run on exception (`data_evaluation_script.py:201`)

**Fix:**
Wrap the MLflow run in a context manager:
```python
with mlflow.start_run(run_name=cfg.run_name):
    # ... all evaluation code ...
```

---

### 3.2 Fix mixed logging frameworks (`data_evaluation_script.py`)

**Fix:**
1. Remove `import logging` and `logging.basicConfig(...)`.
2. Use `loguru.logger` consistently throughout the file.
3. If third-party libraries emit stdlib `logging` messages, add an intercept handler once at the top:
   ```python
   import logging
   class InterceptHandler(logging.Handler):
       def emit(self, record):
           logger.opt(depth=6, exception=record.exc_info).log(record.levelname, record.getMessage())
   logging.basicConfig(handlers=[InterceptHandler()], level=0)
   ```

---

### 3.3 Remove `warnings.filterwarnings("ignore")` (`data_evaluation_script.py:44`)

**Fix:**
Delete the line. If specific warnings are known to be noisy and non-actionable, suppress them individually:
```python
warnings.filterwarnings("ignore", message=".*convergence.*", category=ConvergenceWarning)
```

---

### 3.4 Improve MLflow server diagnostics (`mlflow_utils.py`)

**Fix:**
1. Stop suppressing stdout/stderr from the subprocess. Log them to a file:
   ```python
   log_file = open("mlflow_server.log", "w")
   process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
   ```
2. Log the original connection error in the `except` block instead of swallowing it.
3. Replace the `shell=True` kill command with direct process termination:
   ```python
   def cleanup():
       process.terminate()
       process.wait(timeout=5)
   ```

---

## Phase 4 -- Code Quality Cleanup

### 4.1 Remove debug print statements

- `src/data/utils.py:109` -- remove `print(len(df))`
- `src/data/utils.py:231` -- remove `print(event_counts)`
- `src/evaluation/data_evaluation_script.py:400-401` -- remove `print(real_distribution)` and `print(synthetic_distribution)`
- `src/run_pipeline.py:40` -- remove the commented-out dead code

Replace any that provide useful information with `logger.debug(...)` calls.

---

### 4.2 Fix typo in error message (`marginal_distribution_estimator.py:95`)

```python
# Before:
f"Failed to fit distribution '{distribution.name}' with method '{method}': /n{e}"
# After:
f"Failed to fit distribution '{distribution.name}' with method '{method}':\n{e}"
```

---

### 4.3 Fix type annotation (`synthetization_model_interface.py:36`)

```python
# Before:
description: Optional[dict[str, any]] = None
# After:
from typing import Any
description: Optional[dict[str, Any]] = None
```

---

### 4.4 Fix parallel logger leak (`marginal_distribution_estimator.py:73`)

**Fix:**
Remove `setup_logger()` from the worker function `_process_single_column`. Instead, configure logging once in the main process. Worker subprocesses can use `print()` or pass log messages back via the result tuple.

Alternatively, use `loguru`'s built-in multiprocessing support with `enqueue=True`.

---

## Phase 5 -- Tests

### 5.1 Delete or rewrite the broken test (`tests/dummy_training_and_inference.py`)

The test imports from `v0.src` which no longer exists and hard-codes paths to another developer's machine. It is not salvageable.

**Fix:**
1. Delete `tests/dummy_training_and_inference.py`.
2. Create a minimal integration test suite using `pytest`:

```
tests/
  conftest.py             # Shared fixtures (toy datasets, tmp dirs)
  test_preprocessing.py   # Round-trip: preprocess -> reverse_preprocess
  test_marginal_fitting.py # Fit known distributions, verify parameters
  test_copula_fit.py      # Fit copula on toy data, verify output shape/range
  test_pipeline_e2e.py    # End-to-end: config -> train -> generate -> verify
```

Key test cases:
- **Preprocessing round-trip:** `preprocess(data)` -> `reverse_preprocess(synthetic)` should produce values in the original domain.
- **Truncated Gaussian:** Samples are within bounds, approximately Gaussian.
- **NaN handling:** Correlation matrix with NaN inputs produces valid output.
- **Integer column handling:** Continuous integer columns remain numeric, categorical ones are encoded.
- **Case-insensitive peptide matching:** Both `Peptide_X` and `peptide_X` columns are detected.
- **Histogram imputation None guard:** Calling inference without training raises `ValueError`.
- **End-to-end with toy data:** Pipeline runs, produces a CSV with the expected columns and no NaNs.

---

## Phase 6 -- Configuration Improvements

### 6.1 Use environment variables for MLflow URI

```yaml
# pipeline.yaml
ml_flow_tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,http://127.0.0.1:5000}
```
This uses Hydra's OmegaConf resolver to read from the environment with a local fallback.

---

### 6.2 Use relative or environment-based data paths

```yaml
# paths.yaml
real_data_path: ${oc.env:DATA_DIR,./data}/merged_peptide_and_clinical.csv
```

---

### 6.3 Increase default `sampled_patients_num`

Change from 100 to a more reasonable default (e.g. 500 or `null` for "use all available data"). Document the trade-off in a comment.

---

### 6.4 Add random seed control

Add a `random_seed` config parameter at the top level. Pass it through to:
- `DataProcessor.sample_patients_with_preserved_ratios` (already accepts it)
- `np.random.default_rng(seed)` for copula sampling
- Any other stochastic operation

---

## Execution Order

| Order | Phase | Items | Est. Complexity |
|-------|-------|-------|-----------------|
| 1 | Critical bugs | 1.1, 1.2, 1.3, 1.4 | Small, isolated fixes |
| 2 | Logic errors (independent) | 2.2, 2.7, 2.8 | Small, no dependencies |
| 3 | Logic errors (interrelated) | 2.1, 2.3, 2.4, 2.5, 2.6 | Medium, changes interact |
| 4 | Eval & MLflow | 3.1, 3.2, 3.3, 3.4 | Small |
| 5 | Code quality | 4.1, 4.2, 4.3, 4.4 | Trivial |
| 6 | Tests | 5.1 | Medium |
| 7 | Configuration | 6.1, 6.2, 6.3, 6.4 | Small |

Phases 1-2 should be done first and each fix verified individually before moving on. Phase 3 (the interrelated logic errors around training data selection) should be done as a single coherent change since fixes 2.1/2.3/2.4/2.5/2.6 all touch the training data flow.

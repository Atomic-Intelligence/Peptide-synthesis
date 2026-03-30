# HistogramImputation — Code Review

## Bugs

### 1. Bin edge overlap (double-counting)
The bin membership check uses `>= low AND <= up` for every bin, so a value exactly on an intermediate bin edge is counted in two bins. This skews the median computation for those values.

`np.histogram` uses half-open intervals `[low, up)` for all bins except the last (`[low, up]`). The manual loop should mirror this:

```python
# All bins except last: [low, up)
bin_idx = np.logical_and(col_values >= low, col_values < up)
# Last bin: [low, up]
if i == len(bin_edges) - 2:
    bin_idx = np.logical_and(col_values >= low, col_values <= up)
```

### 2. Mutating `self.column_names` during iteration
The loop iterates over `self.column_names`, then reassigns `self.column_names` inside the loop body when a column is skipped. This works by accident today (Python captured the original list at loop start), but it is fragile and misleading. The fix is to collect valid columns into a separate list and assign once after the loop:

```python
valid_columns = []
for col in tqdm(self.column_names, ...):
    ...
    if len(col_values) == 0:
        ...
        continue
    ...
    valid_columns.append(col)
self.column_names = valid_columns
```

---

## Design issues

### 3. Inefficient median computation
The per-bin median loop reindexes `col_values` once per bin per column. For large datasets with many bins this is slow. `np.digitize` assigns each value to a bin in one vectorised call, after which a single groupby-style pass computes all medians:

```python
bin_indices = np.digitize(col_values, bin_edges[1:-1])  # 0-indexed bin per value
for i in range(self.num_bins):
    mask = bin_indices == i
    self.medians[col].append(
        np.median(col_values[mask]) if mask.any() else (bin_edges[i + 1] + bin_edges[i]) / 2
    )
```

### 4. `data.select(col).to_numpy()` returns shape `(n, 1)`
The `.flatten()` corrects for this, but the idiomatic Polars way to get a 1-D numpy array is:

```python
col_values = data[col].to_numpy()
```

This avoids creating an intermediate DataFrame and the subsequent flatten.

### 5. `np.isnan` fails on non-float dtypes
If a column is an integer type (e.g. `pl.Int64`), `to_numpy()` returns an integer array and `np.isnan` raises a `TypeError`. Polars nullable integers are exported as `float64` with NaN when `allow_copy=True` (the default), so this is usually fine in practice, but it is worth being explicit:

```python
col_values = data[col].cast(pl.Float64).to_numpy()
```

### 6. Inconsistent internal state types
`self.medians[col]` stores a Python `list`; `self.col_densities[col]` stores a `np.ndarray`. Keeping both as numpy arrays makes `generate` more predictable and avoids an implicit conversion on every `choice` call.

### 7. No input validation
`num_bins <= 0` or an empty `column_names` list will produce confusing errors later. A guard in `__init__` catches these early:

```python
if num_bins <= 0:
    raise ValueError(f"num_bins must be positive, got {num_bins}")
```

---

## Alternative sampling strategies

### Uniform sampling within bins vs. median

The current approach collapses each bin to its median and then samples from that discrete set. This means the generator can only ever produce at most `num_bins` distinct values per column, regardless of how large the dataset is. The marginal distribution of the generated column is a point-mass mixture, not a continuous distribution — visible as vertical stripes in a scatter plot or spikes in a density estimate.

Sampling uniformly within the chosen bin avoids this:

```python
# generate: instead of storing medians, store bin edges
bin_idx = np.random.choice(self.num_bins, p=self.col_densities[col], size=n)
low  = self.bin_edges[col][bin_idx]
high = self.bin_edges[col][bin_idx + 1]
samples = np.random.uniform(low, high)
```

This is strictly better in every way the current design cares about:
- Produces a continuous marginal (piecewise-uniform) rather than a discrete approximation.
- Same storage cost (bin edges instead of medians — same number of floats).
- Same or faster generation (no `np.median` at fit time).
- The median is the right representative only if the within-bin distribution is symmetric, which you cannot know. Uniform sampling at least has no such bias assumption.

The only argument for keeping medians is if the downstream consumer explicitly requires imputed values to be observed data points (i.e. members of the original dataset). In that case the median is still wrong — you'd want to sample the actual observed values within each bin.

---

### KDE vs. histogram

KDE is the natural next step, and for continuous columns it is generally preferable. The tradeoff:

| | Histogram | KDE |
|---|---|---|
| Fit cost | O(n) | O(n) with fast implementations (`scipy.stats.gaussian_kde` is O(n²) at *evaluation* time; `statsmodels` KDEUnivariate with FFT is O(n log n)) |
| Sample cost | O(1) per sample | O(n) naively; O(1) with inverse-CDF or rejection sampling |
| Hyperparameter | `num_bins` (hard to tune) | bandwidth (can be estimated automatically, e.g. Scott's or Silverman's rule) |
| Boundary handling | Natural — bins stop at data range | Leaks probability mass outside data range unless a bounded kernel or reflection is used |
| Sparse columns | Degrades gracefully — empty bins get zero weight | Silverman bandwidth can over-smooth if the column is multi-modal or has outliers |
| Interpretability | Simple | Slightly less transparent |

**Practical recommendation for this use case:**

The columns here appear to be sparse biological measurements (peptide/clinical data). Many will be zero-inflated or have hard lower bounds (e.g. concentration cannot be negative). For those, a naive KDE will assign non-trivial probability to negative values, which histogram sampling avoids naturally.

A reasonable progression:
1. **Short term**: switch from median to uniform-within-bin sampling. Zero code complexity added, immediately fixes the discretisation artifact.
2. **Medium term**: use KDE with reflection at the boundary (or a log-transform for strictly positive columns) for columns that are dense enough to benefit from it. `statsmodels.nonparametric.kde.KDEUnivariate` with `fft=True` is fast and supports bandwidth selection.
3. **Skip** a full KDE for columns that are zero-inflated — model the zero-mass separately and fit KDE only on the positive part (a two-component mixture).

---

## Summary table

| # | Severity | Category | Issue |
|---|----------|----------|-------|
| 1 | Bug | Correctness | Bin edge double-counting corrupts medians |
| 2 | Bug | Safety | Mutating iterable inside loop body |
| 3 | Medium | Performance | O(n × bins) indexing instead of one `np.digitize` pass |
| 4 | Low | Style | `data.select(col).to_numpy().flatten()` vs `data[col].to_numpy()` |
| 5 | Low | Robustness | `np.isnan` fails on integer dtypes |
| 6 | Low | Consistency | Mixed list / ndarray internal state |
| 7 | Low | Robustness | No validation of `num_bins` |

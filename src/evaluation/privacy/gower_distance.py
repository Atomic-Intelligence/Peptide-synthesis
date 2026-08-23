"""Gower's Distance for mixed-type (numerical + categorical) data.

Gower's distance computes per-feature distances using type-appropriate
measures, then averages them:

- **Numerical features**: absolute difference normalised by the feature range
  (max - min), so each feature contributes equally regardless of scale.
- **Categorical features**: simple mismatch indicator (0 if equal, 1 otherwise).

The final distance is the mean of all per-feature distances, bounded in [0, 1].

This module provides a pairwise distance function compatible with
``sklearn.neighbors.NearestNeighbors(metric="precomputed")`` and a
``NearestNeighbors``-compatible wrapper for use in DCR estimation.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, List, Optional
from loguru import logger


class GowerDistanceCalculator:
    """Compute Gower's distance between records with mixed feature types.

    Unlike Euclidean distance on scaled+one-hot-encoded features, Gower's
    distance operates on the *original* feature values and handles numerical
    and categorical features natively.

    Parameters
    ----------
    categorical_columns :
        Column names that are categorical.  All other columns are treated as
        numerical.
    feature_groups :
        Optional mapping of group name → list of column names.  Each named
        group receives an equal share of the total weight (1 / n_groups),
        distributed evenly among its member features.  This prevents a large
        group (e.g. many peptide columns) from dominating the distance over a
        small but clinically important group.  Columns not listed in any group
        are collected into an implicit ``"__other__"`` group that receives the
        same budget as every named group.  When ``None`` (default) all features
        are weighted equally, reproducing the original Gower formula.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        self.categorical_columns: List[str] = categorical_columns or []
        self.feature_groups: Optional[Dict[str, List[str]]] = feature_groups
        self._numerical_cols: Optional[List[str]] = None
        self._categorical_cols: Optional[List[str]] = None
        self._ranges: Optional[np.ndarray] = None  # per numerical feature
        self._weights_num: Optional[np.ndarray] = None
        self._weights_cat: Optional[np.ndarray] = None
        self.fitted = False

    def _compute_weights(self, all_columns: List[str]) -> np.ndarray:
        """Return a normalised per-feature weight array for *all_columns*."""
        n = len(all_columns)
        if not self.feature_groups or n == 0:
            return np.ones(n) / n

        col_to_group: Dict[str, str] = {}
        for group_name, cols in self.feature_groups.items():
            for col in cols:
                col_to_group[col] = group_name

        # Ungrouped columns share an implicit group
        for col in all_columns:
            if col not in col_to_group:
                col_to_group[col] = "__other__"

        active_groups = sorted({col_to_group[c] for c in all_columns})
        n_groups = len(active_groups)
        group_budget = 1.0 / n_groups

        group_sizes = {g: sum(1 for c in all_columns if col_to_group[c] == g)
                       for g in active_groups}

        weights = np.array(
            [group_budget / group_sizes[col_to_group[c]] for c in all_columns],
            dtype=np.float64,
        )
        # Normalise to guard against floating-point drift
        weights /= weights.sum()
        return weights

    def fit(self, dataframe: pl.DataFrame) -> "GowerDistanceCalculator":
        """Learn feature ranges and per-feature weights from a reference dataset."""
        self._numerical_cols = [
            c for c in dataframe.columns if c not in self.categorical_columns
        ]
        self._categorical_cols = [
            c for c in dataframe.columns if c in self.categorical_columns
        ]

        if self._numerical_cols:
            num_array = dataframe.select(self._numerical_cols).to_numpy().astype(np.float64)
            col_min = np.nanmin(num_array, axis=0)
            col_max = np.nanmax(num_array, axis=0)
            self._ranges = col_max - col_min
            # Avoid division by zero for constant features
            self._ranges[self._ranges == 0] = 1.0
        else:
            self._ranges = np.array([])

        all_cols = self._numerical_cols + self._categorical_cols
        all_weights = self._compute_weights(all_cols)
        n_num = len(self._numerical_cols)
        self._weights_num = all_weights[:n_num]
        self._weights_cat = all_weights[n_num:]

        self.fitted = True

        if self.feature_groups:
            active_groups = sorted({
                next((g for g, cols in self.feature_groups.items() if c in cols), "__other__")
                for c in all_cols
            })
            group_info = ", ".join(
                f"{g}={sum(1 for c in all_cols if (next((gr for gr, cols in self.feature_groups.items() if c in cols), '__other__')) == g)}"
                for g in active_groups
            )
            logger.info(
                f"GowerDistanceCalculator fitted with group-balance weights "
                f"({len(active_groups)} groups: {group_info}); "
                f"{len(self._numerical_cols)} numerical, {len(self._categorical_cols)} categorical."
            )
        else:
            logger.info(
                f"GowerDistanceCalculator fitted: {len(self._numerical_cols)} numerical, "
                f"{len(self._categorical_cols)} categorical features (equal weights)."
            )
        return self

    def _to_arrays(
        self, dataframe: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a dataframe to separate numerical and categorical arrays."""
        if self._numerical_cols:
            num = dataframe.select(self._numerical_cols).to_numpy().astype(np.float64)
        else:
            num = np.empty((len(dataframe), 0), dtype=np.float64)

        if self._categorical_cols:
            cat = dataframe.select(self._categorical_cols).to_numpy()
        else:
            cat = np.empty((len(dataframe), 0), dtype=object)

        return num, cat

    def pairwise_distance(
        self,
        df_a: pl.DataFrame,
        df_b: pl.DataFrame,
    ) -> np.ndarray:
        """Compute the full pairwise Gower distance matrix.

        Returns
        -------
        dist : ndarray of shape (len(df_a), len(df_b))
            ``dist[i, j]`` is the Gower distance between record *i* of
            ``df_a`` and record *j* of ``df_b``.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before pairwise_distance().")

        num_a, cat_a = self._to_arrays(df_a)
        num_b, cat_b = self._to_arrays(df_b)

        n_a, n_b = len(num_a), len(num_b)
        n_num = num_a.shape[1]
        n_cat = cat_a.shape[1]
        n_features = n_num + n_cat

        if n_features == 0:
            return np.zeros((n_a, n_b))

        dist = np.zeros((n_a, n_b), dtype=np.float64)

        # Numerical contribution: w_j * |x_i - y_j| / range_j
        if n_num > 0:
            for j in range(n_num):
                diff = np.abs(num_a[:, j:j+1] - num_b[:, j].reshape(1, -1))
                dist += self._weights_num[j] * diff / self._ranges[j]

        # Categorical contribution: w_j * (0 if equal, 1 if different)
        if n_cat > 0:
            for j in range(n_cat):
                mismatch = (cat_a[:, j:j+1] != cat_b[:, j].reshape(1, -1)).astype(np.float64)
                dist += self._weights_cat[j] * mismatch

        # Weights are normalised to sum to 1, so dist is already in [0, 1].
        # Divide by actual sum to be robust against floating-point drift.
        total_weight = self._weights_num.sum() + self._weights_cat.sum()
        if total_weight > 0:
            dist /= total_weight
        return dist

    def nearest_neighbor_distances(
        self,
        df_query: pl.DataFrame,
        df_reference: pl.DataFrame,
        n_neighbors: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find nearest-neighbour distances and indices using Gower's distance.

        Parameters
        ----------
        df_query :
            Records to query (rows we want NN distances for).
        df_reference :
            Reference dataset to search against.
        n_neighbors :
            Number of nearest neighbours to return.

        Returns
        -------
        distances : ndarray of shape (len(df_query), n_neighbors)
        indices : ndarray of shape (len(df_query), n_neighbors)
        """
        dist_matrix = self.pairwise_distance(df_query, df_reference)

        if n_neighbors >= dist_matrix.shape[1]:
            sorted_idx = np.argsort(dist_matrix, axis=1)
            return np.take_along_axis(dist_matrix, sorted_idx, axis=1), sorted_idx

        # Use argpartition for efficiency when n_neighbors << n_reference
        part_idx = np.argpartition(dist_matrix, n_neighbors, axis=1)[:, :n_neighbors]
        part_dist = np.take_along_axis(dist_matrix, part_idx, axis=1)

        # Sort the top-k
        sort_order = np.argsort(part_dist, axis=1)
        indices = np.take_along_axis(part_idx, sort_order, axis=1)
        distances = np.take_along_axis(part_dist, sort_order, axis=1)

        return distances, indices

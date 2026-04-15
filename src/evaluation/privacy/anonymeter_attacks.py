"""Privacy attacks: singling out, linkability, attribute inference.

Custom implementations of the Giomi et al. (2023) privacy risk framework,
with no dependency on the ``anonymeter`` package.

All three attacks share the same risk formula:

    risk = max(0, R_attack − R_control) / max(1 − R_control, ε)

    R_attack  – adversary success rate using synthetic records as queries
    R_control – success rate using held-out real records (never seen by the
                generative model) as queries

Attacks
-------
  - Singling out (univariate): per-column exact-match queries; success when a
    synthetic record's value for some column is unique in the real dataset.
  - Singling out (multivariate): KNN gap-ratio approach on the full encoded
    feature space; a record is "singled out" when its nearest real neighbour
    is far closer than the second-nearest (d₂/d₁ > threshold).
  - Linkability: synthetic records used as bridges between two attribute views
    (A and B) of the real dataset; success when both views lead to the same
    real record.
  - Attribute inference: nearest-synthetic-neighbour imputation; success when
    the inferred value of a hidden column matches the true value.

References
----------
Giomi et al. (2023) "A Unified Framework for Quantifying Privacy Risk in
Synthetic Data", PoPETs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict
from loguru import logger
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from src.evaluation.privacy.preprocessing import FeatureProcessor


# ---------------------------------------------------------------------------
# Risk formula
# ---------------------------------------------------------------------------


def _risk(
    atk_k: int,
    atk_n: int,
    ctrl_k: int,
    ctrl_n: int,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """Privacy risk with Wilson-score 95 % CI.

    risk = max(0, R_atk − R_ctrl) / max(1 − R_ctrl, ε)
    """
    r_atk = atk_k / max(atk_n, 1)
    r_ctrl = ctrl_k / max(ctrl_n, 1)
    denom = max(1.0 - r_ctrl, 1e-10)
    value = min(1.0, max(0.0, r_atk - r_ctrl) / denom)

    z = stats.norm.ppf(1.0 - alpha / 2.0)

    def _wilson(k: int, n: int) -> Tuple[float, float]:
        if n == 0:
            return 0.0, 1.0
        p = k / n
        adj = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / adj
        margin = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / adj
        return max(0.0, center - margin), min(1.0, center + margin)

    a_lo, a_hi = _wilson(atk_k, atk_n)
    c_lo, c_hi = _wilson(ctrl_k, ctrl_n)
    ci = (
        max(0.0, min(1.0, (a_lo - c_hi) / denom)),
        max(0.0, min(1.0, (a_hi - c_lo) / denom)),
    )
    return value, ci


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class AnonymeterResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Singling out
    singling_out_univariate: Optional[float] = None
    singling_out_multivariate: Optional[float] = None
    singling_out_univariate_ci: Optional[Tuple[float, float]] = None
    singling_out_multivariate_ci: Optional[Tuple[float, float]] = None

    # Linkability
    linkability: Optional[float] = None
    linkability_ci: Optional[Tuple[float, float]] = None

    # Attribute inference (mean across targets + per-column breakdown)
    inference: Optional[float] = None
    inference_per_column: Optional[Dict[str, float]] = None

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.singling_out_univariate is not None:
            out["singling_out_univariate_risk"] = self.singling_out_univariate
        if self.singling_out_multivariate is not None:
            out["singling_out_multivariate_risk"] = self.singling_out_multivariate
        if self.linkability is not None:
            out["linkability_risk"] = self.linkability
        if self.inference is not None:
            out["attribute_inference_risk"] = self.inference
        if self.inference_per_column:
            for col, v in self.inference_per_column.items():
                safe = (
                    col.replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_")
                )
                out[f"attr_inference_{safe}_risk"] = v
        return out


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class AnonymeterAttacksEstimator:
    """Run singling out, linkability, and attribute inference.

    Parameters
    ----------
    categorical_columns :
        Columns to one-hot encode for distance-based attacks; used as default
        attribute-inference targets.
    holdout_fraction :
        Fraction of real data reserved as the control set.  Default 0.2.
    n_attacks :
        Number of attack queries per evaluator.  Default 2000.
    run_singling_out :
        Whether to run singling out (univariate + multivariate).  Default True.
    run_linkability :
        Whether to run the linkability attack.  Default True.
    run_attribute_inference :
        Whether to run attribute inference.  Default True.
    linkability_aux_cols :
        Tuple ``(columns_A, columns_B)`` for linkability.  When ``None``
        (default), columns are split 50/50 randomly.
    inference_target_cols :
        Columns the adversary tries to infer.  Defaults to ``categorical_columns``.
    n_jobs :
        Parallelism for NearestNeighbors.  Default -1 (all cores).
    gap_ratio_threshold :
        Multivariate singling-out threshold.  A record with
        d(2nd-NN) / d(1st-NN) > threshold is considered singled out.  Default 2.0.
    inference_tolerance :
        Relative tolerance for numerical inference success:
        |inferred − true| / (col_range + ε) < tol counts as correct.  Default 0.1.
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        holdout_fraction: float = 0.2,
        n_attacks: int = 2000,
        run_singling_out: bool = True,
        run_linkability: bool = True,
        run_attribute_inference: bool = True,
        linkability_aux_cols: Optional[Tuple[List[str], List[str]]] = None,
        inference_target_cols: Optional[List[str]] = None,
        n_jobs: int = -1,
        gap_ratio_threshold: float = 2.0,
        inference_tolerance: float = 0.1,
    ):
        self.categorical_columns = categorical_columns or []
        self.holdout_fraction = holdout_fraction
        self.n_attacks = n_attacks
        self.run_singling_out = run_singling_out
        self.run_linkability = run_linkability
        self.run_attribute_inference = run_attribute_inference
        self.linkability_aux_cols = linkability_aux_cols
        self.inference_target_cols = inference_target_cols
        self.n_jobs = n_jobs
        self.gap_ratio_threshold = gap_ratio_threshold
        self.inference_tolerance = inference_tolerance

        self._ori_df: Optional[pl.DataFrame] = None
        self._control_df: Optional[pl.DataFrame] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, real_dataframe: pl.DataFrame) -> "AnonymeterAttacksEstimator":
        """Split real data into ori (training proxy) and control (holdout)."""
        n = len(real_dataframe)
        n_holdout = max(1, int(n * self.holdout_fraction))
        rng = np.random.default_rng()
        idx = rng.permutation(n)
        self._control_df = real_dataframe[idx[:n_holdout].tolist()]
        self._ori_df = real_dataframe[idx[n_holdout:].tolist()]
        logger.info(
            f"AnonymeterAttacksEstimator fitted: "
            f"{len(self._ori_df)} ori / {len(self._control_df)} control records."
        )
        return self

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    def _resolve_linkability_cols(
        self, columns: List[str]
    ) -> Tuple[List[str], List[str]]:
        if self.linkability_aux_cols is not None:
            avail = set(columns)
            a = [c for c in self.linkability_aux_cols[0] if c in avail]
            b = [c for c in self.linkability_aux_cols[1] if c in avail]
            if a and b:
                return a, b
            logger.warning(
                "linkability_aux_cols has absent columns; falling back to random 50/50 split."
            )
        shuffled = list(columns)
        np.random.default_rng().shuffle(shuffled)
        half = max(1, len(shuffled) // 2)
        return shuffled[:half], shuffled[half:]

    def _resolve_inference_targets(self, columns: List[str]) -> List[str]:
        if self.inference_target_cols is not None:
            return [c for c in self.inference_target_cols if c in columns]
        return [c for c in self.categorical_columns if c in columns]

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _make_encoded(
        self,
        ori_df: pl.DataFrame,
        *others: pl.DataFrame,
        columns: List[str],
    ) -> Tuple[np.ndarray, ...]:
        """Fit a FeatureProcessor on ori[columns] and transform all DataFrames."""
        cat_cols = [c for c in self.categorical_columns if c in columns]
        fp = FeatureProcessor(scaler=RobustScaler(), categorical_columns=cat_cols)
        fp.fit(ori_df.select(columns))
        return tuple(fp.transform(df.select(columns)) for df in (ori_df, *others))

    # ------------------------------------------------------------------
    # Singling out – univariate
    # ------------------------------------------------------------------

    def _singling_out_univariate(
        self,
        ori_pd: pd.DataFrame,
        syn_pd: pd.DataFrame,
        ctrl_pd: pd.DataFrame,
        rng: np.random.Generator,
    ) -> Tuple[float, Tuple[float, float]]:
        """Univariate singling-out: single-column exact-match queries.

        A query record singles out a real person when any of its column values
        appears exactly once in the ori (training proxy) dataset.
        """
        singleton_sets: Dict[str, set] = {}
        for col in ori_pd.columns:
            vc = ori_pd[col].value_counts()
            singleton_sets[col] = set(vc[vc == 1].index)

        def _successes(queries: pd.DataFrame) -> int:
            hit = pd.Series(False, index=queries.index)
            for col, singletons in singleton_sets.items():
                if singletons:
                    hit |= queries[col].isin(singletons)
            return int(hit.sum())

        atk_idx = rng.choice(
            len(syn_pd), min(self.n_attacks, len(syn_pd)), replace=False
        )
        ctrl_idx = rng.choice(
            len(ctrl_pd), min(self.n_attacks, len(ctrl_pd)), replace=False
        )
        atk_k = _successes(syn_pd.iloc[atk_idx])
        atk_n = len(atk_idx)
        ctrl_k = _successes(ctrl_pd.iloc[ctrl_idx])
        ctrl_n = len(ctrl_idx)
        return _risk(atk_k, atk_n, ctrl_k, ctrl_n)

    # ------------------------------------------------------------------
    # Singling out – multivariate
    # ------------------------------------------------------------------

    def _singling_out_multivariate(
        self,
        ori_enc: np.ndarray,
        syn_enc: np.ndarray,
        ctrl_enc: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[float, Tuple[float, float]]:
        """Multivariate singling-out via KNN gap ratio.

        A record is considered singled out when its nearest real neighbour is
        significantly closer than the second-nearest:
            d(2nd-NN) / d(1st-NN) > gap_ratio_threshold
        """
        knn = NearestNeighbors(
            n_neighbors=2, n_jobs=self.n_jobs, algorithm="ball_tree"
        ).fit(ori_enc)

        def _gap_successes(queries: np.ndarray) -> int:
            dists, _ = knn.kneighbors(queries)
            d1, d2 = dists[:, 0], dists[:, 1]
            ratio = np.where(d1 > 1e-12, d2 / d1, 1.0)
            return int((ratio > self.gap_ratio_threshold).sum())

        atk_idx = rng.choice(
            len(syn_enc), min(self.n_attacks, len(syn_enc)), replace=False
        )
        ctrl_idx = rng.choice(
            len(ctrl_enc), min(self.n_attacks, len(ctrl_enc)), replace=False
        )
        atk_k = _gap_successes(syn_enc[atk_idx])
        atk_n = len(atk_idx)
        ctrl_k = _gap_successes(ctrl_enc[ctrl_idx])
        ctrl_n = len(ctrl_idx)
        return _risk(atk_k, atk_n, ctrl_k, ctrl_n)

    # ------------------------------------------------------------------
    # Linkability
    # ------------------------------------------------------------------

    def _linkability(
        self,
        ori_df: pl.DataFrame,
        syn_df: pl.DataFrame,
        ctrl_df: pl.DataFrame,
        cols_a: List[str],
        cols_b: List[str],
        rng: np.random.Generator,
    ) -> Tuple[float, Tuple[float, float]]:
        """Linkability attack.

        Each query record (synthetic or control) is used to probe the real
        dataset from two independent attribute views:

          1. Find the nearest ori record via A columns (quasi-identifiers).
          2. Find the nearest ori record via B columns (sensitive attributes).

        The attack succeeds if both queries converge on the same ori record,
        meaning the synthetic data acts as a bridge between the two views.
        """
        ori_a, syn_a, ctrl_a = self._make_encoded(
            ori_df, syn_df, ctrl_df, columns=cols_a
        )
        ori_b, syn_b, ctrl_b = self._make_encoded(
            ori_df, syn_df, ctrl_df, columns=cols_b
        )

        knn_a = NearestNeighbors(
            n_neighbors=1, n_jobs=self.n_jobs, algorithm="ball_tree"
        ).fit(ori_a)
        knn_b = NearestNeighbors(
            n_neighbors=1, n_jobs=self.n_jobs, algorithm="ball_tree"
        ).fit(ori_b)

        def _link_successes(q_a: np.ndarray, q_b: np.ndarray) -> int:
            _, nn_a = knn_a.kneighbors(q_a)
            _, nn_b = knn_b.kneighbors(q_b)
            return int(np.sum(nn_a[:, 0] == nn_b[:, 0]))

        atk_idx = rng.choice(
            len(syn_df), min(self.n_attacks, len(syn_df)), replace=False
        )
        ctrl_idx = rng.choice(
            len(ctrl_df), min(self.n_attacks, len(ctrl_df)), replace=False
        )
        atk_k = _link_successes(syn_a[atk_idx], syn_b[atk_idx])
        atk_n = len(atk_idx)
        ctrl_k = _link_successes(ctrl_a[ctrl_idx], ctrl_b[ctrl_idx])
        ctrl_n = len(ctrl_idx)
        return _risk(atk_k, atk_n, ctrl_k, ctrl_n)

    # ------------------------------------------------------------------
    # Attribute inference
    # ------------------------------------------------------------------

    def _attribute_inference_col(
        self,
        ori_df: pl.DataFrame,
        syn_df: pl.DataFrame,
        ctrl_df: pl.DataFrame,
        secret_col: str,
        rng: np.random.Generator,
    ) -> float:
        """Attribute inference for a single secret column.

        For each query record q (ori or ctrl), the adversary:
          1. Finds the nearest synthetic record s using all auxiliary columns.
          2. Infers q[secret] ≈ s[secret].
          3. Succeeds if the inferred value matches the true value (exact for
             categorical; within ``inference_tolerance`` of column range for
             numerical).

        Using ori records as attack queries and ctrl records as control.
        """
        aux_cols = [c for c in ori_df.columns if c != secret_col]
        if not aux_cols:
            return 0.0

        ori_aux, syn_aux, ctrl_aux = self._make_encoded(
            ori_df, syn_df, ctrl_df, columns=aux_cols
        )

        knn = NearestNeighbors(
            n_neighbors=1, n_jobs=self.n_jobs, algorithm="ball_tree"
        ).fit(syn_aux)

        is_cat = secret_col in self.categorical_columns
        ori_secret = ori_df[secret_col].to_numpy()
        ctrl_secret = ctrl_df[secret_col].to_numpy()
        syn_secret = syn_df[secret_col].to_numpy()
        col_range = float(np.ptp(ori_secret)) if not is_cat else 1.0

        def _infer_successes(q_enc: np.ndarray, true_vals: np.ndarray) -> int:
            _, nn = knn.kneighbors(q_enc)
            inferred = syn_secret[nn[:, 0]]
            if is_cat:
                return int(np.sum(inferred == true_vals))
            return int(
                np.sum(
                    np.abs(inferred - true_vals) / max(col_range, 1e-10)
                    < self.inference_tolerance
                )
            )

        atk_idx = rng.choice(
            len(ori_df), min(self.n_attacks, len(ori_df)), replace=False
        )
        ctrl_idx = rng.choice(
            len(ctrl_df), min(self.n_attacks, len(ctrl_df)), replace=False
        )
        atk_k = _infer_successes(ori_aux[atk_idx], ori_secret[atk_idx])
        atk_n = len(atk_idx)
        ctrl_k = _infer_successes(ctrl_aux[ctrl_idx], ctrl_secret[ctrl_idx])
        ctrl_n = len(ctrl_idx)
        value, _ = _risk(atk_k, atk_n, ctrl_k, ctrl_n)
        return value

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def estimate(self, synthetic_dataframe: pl.DataFrame) -> AnonymeterResults:
        """Run all enabled attacks and return results.

        Parameters
        ----------
        synthetic_dataframe :
            Synthetic dataset to evaluate.
        """
        if self._ori_df is None:
            raise RuntimeError("Call fit() before estimate().")

        rng = np.random.default_rng()

        common_cols = sorted(
            set(self._ori_df.columns)
            & set(synthetic_dataframe.columns)
            & set(self._control_df.columns)
        )
        ori_df = self._ori_df.select(common_cols)
        syn_df = synthetic_dataframe.select(common_cols)
        ctrl_df = self._control_df.select(common_cols)

        ori_pd = ori_df.to_pandas()
        syn_pd = syn_df.to_pandas()
        ctrl_pd = ctrl_df.to_pandas()

        results = AnonymeterResults()

        # --- Singling out ---
        if self.run_singling_out:
            try:
                logger.info("Running singling out (univariate)...")
                risk, ci = self._singling_out_univariate(ori_pd, syn_pd, ctrl_pd, rng)
                results.singling_out_univariate = risk
                results.singling_out_univariate_ci = ci
                logger.success(f"Singling out (univariate): {risk:.4f}  CI={ci}")
            except Exception as exc:
                logger.error(f"Singling out (univariate) failed — {exc}")

            try:
                logger.info("Running singling out (multivariate)...")
                ori_enc, syn_enc, ctrl_enc = self._make_encoded(
                    ori_df, syn_df, ctrl_df, columns=common_cols
                )
                risk, ci = self._singling_out_multivariate(
                    ori_enc, syn_enc, ctrl_enc, rng
                )
                results.singling_out_multivariate = risk
                results.singling_out_multivariate_ci = ci
                logger.success(f"Singling out (multivariate): {risk:.4f}  CI={ci}")
            except Exception as exc:
                logger.error(f"Singling out (multivariate) failed — {exc}")

        # --- Linkability ---
        if self.run_linkability:
            try:
                logger.info("Running linkability...")
                cols_a, cols_b = self._resolve_linkability_cols(common_cols)
                logger.info(f"Linkability: {len(cols_a)} A-cols, {len(cols_b)} B-cols.")
                risk, ci = self._linkability(
                    ori_df, syn_df, ctrl_df, cols_a, cols_b, rng
                )
                results.linkability = risk
                results.linkability_ci = ci
                logger.success(f"Linkability: {risk:.4f}  CI={ci}")
            except Exception as exc:
                logger.error(f"Linkability failed — {exc}")

        # --- Attribute inference ---
        if self.run_attribute_inference:
            target_cols = self._resolve_inference_targets(common_cols)
            if not target_cols:
                logger.warning(
                    "No inference target columns resolved; skipping attribute inference. "
                    "Pass inference_target_cols or categorical_columns to enable this attack."
                )
            else:
                inference_per_col: Dict[str, float] = {}
                for secret_col in target_cols:
                    try:
                        logger.info(
                            f"Running attribute inference for '{secret_col}'..."
                        )
                        v = self._attribute_inference_col(
                            ori_df, syn_df, ctrl_df, secret_col, rng
                        )
                        inference_per_col[secret_col] = v
                        logger.success(f"Attribute inference [{secret_col}]: {v:.4f}")
                    except Exception as exc:
                        logger.error(
                            f"Attribute inference [{secret_col}] failed — {exc}"
                        )
                if inference_per_col:
                    results.inference = float(np.mean(list(inference_per_col.values())))
                    results.inference_per_column = inference_per_col

        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        results: AnonymeterResults,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Bar chart of all risk values with CI error bars."""
        labels: List[str] = []
        values: List[float] = []
        yerr_low: List[float] = []
        yerr_high: List[float] = []

        def _add(label: str, val: float, ci: Optional[Tuple[float, float]]) -> None:
            labels.append(label)
            values.append(val)
            if ci is not None:
                yerr_low.append(max(0.0, val - ci[0]))
                yerr_high.append(max(0.0, ci[1] - val))
            else:
                yerr_low.append(0.0)
                yerr_high.append(0.0)

        if results.singling_out_univariate is not None:
            _add(
                "Singling Out\n(Univariate)",
                results.singling_out_univariate,
                results.singling_out_univariate_ci,
            )
        if results.singling_out_multivariate is not None:
            _add(
                "Singling Out\n(Multivariate)",
                results.singling_out_multivariate,
                results.singling_out_multivariate_ci,
            )
        if results.linkability is not None:
            _add("Linkability", results.linkability, results.linkability_ci)
        if results.inference_per_column:
            for col, v in results.inference_per_column.items():
                _add(f"Attr Inf\n[{col[:14]}]", v, None)

        if not labels:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.text(
                0.5,
                0.5,
                "No results",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        fig, ax = plt.subplots(figsize=(max(6, 2 * len(labels)), 5))
        x = np.arange(len(labels))
        ax.bar(
            x,
            values,
            color="steelblue",
            alpha=0.8,
            yerr=[yerr_low, yerr_high],
            capsize=5,
            error_kw={"elinewidth": 1.5},
        )
        ax.axhline(
            0.0, color="green", linestyle="--", linewidth=1, label="Perfect privacy (0)"
        )
        ax.axhline(
            1.0,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Maximum risk (1)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Privacy risk (lower is better)")
        ax.set_ylim(-0.15, 1.15)
        ax.set_title("Privacy Attacks")
        ax.legend(fontsize=8)
        fig.suptitle(
            "Singling Out · Linkability · Attribute Inference  |  Risk ∈ [0, 1], lower is better",
            fontsize=10,
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig

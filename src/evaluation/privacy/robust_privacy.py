"""RobustPrivacyReport — run the privacy suite across multiple representations.

A single privacy number computed in one feature space is fragile: it depends on
the dimensionality, the scaling, and the exact set of columns an attacker is
assumed to hold.  This orchestrator hardens the audit by re-running the full
:class:`~privacy_report.PrivacyReport` in several representations and reporting
how the risk moves between them:

1. **Variable subfeatures** — draw many random column subsets and report the
   *distribution* (mean / std / min / max / p5 / p95) of every privacy metric.
   This answers "how much does the measured risk depend on which features the
   attacker has?" rather than committing to one arbitrary feature set.
2. **Reduced dimensionality** — project into low-dimensional PCA and UMAP spaces
   before computing distances, controlling curse-of-dimensionality artefacts
   that inflate or flatten nearest-neighbour distances in the raw space.
3. **TabPFN embedding space** — measure privacy in the learned embedding space of
   a pre-trained TabPFN foundation model, where distance reflects the model's
   notion of record similarity rather than raw feature geometry.

The ``identity`` (raw) space is always included as the baseline.  Each
representation is fitted on the real data and applied to both frames, then the
existing :class:`PrivacyReport` (DCR, authenticity, re-identification,
anonymeter singling-out / linkability / attribute-inference, and membership
inference) runs on the transformed frames.  Every sub-metric is wrapped in
``PrivacyReport``'s own try/except, so a metric that is ill-defined in a given
space (e.g. anonymeter on continuous embeddings) is skipped without aborting the
report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from loguru import logger

from src.evaluation.privacy.privacy_report import PrivacyReport, PrivacyResults
from src.evaluation.privacy.representations import (
    IdentityRepresentation,
    PCARepresentation,
    Representation,
    RepresentationOutput,
    RepresentationUnavailable,
    SubfeatureRepresentation,
    TabPFNRepresentation,
    UMAPRepresentation,
)


# ---------------------------------------------------------------------------
# Results containers
# ---------------------------------------------------------------------------


@dataclass
class RepresentationPrivacy:
    """Privacy results for one fixed representation (a single feature space)."""

    name: str
    n_features: int
    results: PrivacyResults

    def summary(self) -> Dict[str, float]:
        return self.results.summary()


@dataclass
class SubfeaturePrivacy:
    """Aggregated privacy risk across many random column-subset draws.

    ``per_draw`` holds one flat metric dict per subset; ``distribution`` holds,
    for every metric, the mean/std/min/max/p5/p95 across draws.
    """

    subset_size: int
    n_draws: int
    per_draw: List[Dict[str, float]]
    distribution: Dict[str, Dict[str, float]]

    def summary(self) -> Dict[str, float]:
        """Flatten to ``subfeat_<metric>_<stat>`` for MLflow / CSV export."""
        out: Dict[str, float] = {}
        for metric, stats in self.distribution.items():
            base = metric.replace("privacy/", "")
            for stat, value in stats.items():
                out[f"subfeat_{base}_{stat}"] = value
        return out


@dataclass
class RobustPrivacyResults:
    """Full multi-representation privacy audit."""

    representations: Dict[str, RepresentationPrivacy] = field(default_factory=dict)
    subfeatures: Optional[SubfeaturePrivacy] = None
    skipped: Dict[str, str] = field(default_factory=dict)  # name -> reason

    def comparison_table(self) -> pl.DataFrame:
        """Long-format table: one row per (representation, metric, value).

        Includes the fixed representations plus, when present, the subfeature
        distribution as a ``subfeature (mean)`` pseudo-representation.
        """
        rows: List[Dict[str, Any]] = []
        for rep_name, rep in self.representations.items():
            for metric, value in rep.summary().items():
                rows.append(
                    {
                        "representation": rep_name,
                        "n_features": rep.n_features,
                        "metric": metric.replace("privacy/", ""),
                        "value": float(value) if _is_number(value) else np.nan,
                    }
                )
        if self.subfeatures is not None:
            for metric, stats in self.subfeatures.distribution.items():
                rows.append(
                    {
                        "representation": "subfeature (mean)",
                        "n_features": self.subfeatures.subset_size,
                        "metric": metric.replace("privacy/", ""),
                        "value": stats.get("mean", np.nan),
                    }
                )
        if not rows:
            return pl.DataFrame(
                schema={
                    "representation": pl.Utf8,
                    "n_features": pl.Int64,
                    "metric": pl.Utf8,
                    "value": pl.Float64,
                }
            )
        return pl.DataFrame(rows)

    def summary(self) -> Dict[str, float]:
        """Flatten everything to ``robust/<representation>/<metric>`` keys."""
        out: Dict[str, float] = {}
        for rep_name, rep in self.representations.items():
            for metric, value in rep.summary().items():
                if _is_number(value):
                    key = f"robust/{rep_name}/{metric.replace('privacy/', '')}"
                    out[key] = float(value)
        if self.subfeatures is not None:
            for metric, value in self.subfeatures.summary().items():
                out[f"robust/{metric}"] = float(value)
        return out


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class RobustPrivacyReport:
    """Run :class:`PrivacyReport` across identity, subfeature, PCA, UMAP and
    TabPFN representations and aggregate the results.

    Parameters
    ----------
    base_privacy_kwargs :
        Keyword arguments forwarded to every :class:`PrivacyReport` (e.g.
        ``run_dcr``, ``n_anonymeter_attacks``, thresholds).  ``categorical_columns``
        is overridden per representation, so any value passed here is ignored.
    categorical_columns :
        Categorical columns of the *raw* data.  Projection / embedding spaces
        report no categoricals; subfeature draws intersect this with the subset.
    run_identity, run_subfeatures, run_pca, run_umap, run_tabpfn :
        Toggles for each representation.
    subfeature_subset_size, subfeature_fraction :
        Subset size for the subfeature draws, as an absolute count or a fraction
        of the available columns (fraction wins when both are set).
    subfeature_n_draws :
        Number of random subsets to draw.
    pca_components, umap_components :
        Target dimensionality of the projections.
    tabpfn_target_col :
        Supervised target column for the TabPFN embedding.
    tabpfn_model_path, tabpfn_device, tabpfn_n_estimators :
        TabPFN backend configuration (see :class:`TabPFNRepresentation`).
    seed :
        Base seed; subfeature draw ``i`` uses ``seed + i``.
    """

    def __init__(
        self,
        base_privacy_kwargs: Optional[Dict[str, Any]] = None,
        categorical_columns: Optional[Sequence[str]] = None,
        run_identity: bool = True,
        run_subfeatures: bool = True,
        run_pca: bool = True,
        run_umap: bool = True,
        run_tabpfn: bool = True,
        subfeature_subset_size: int = 10,
        subfeature_fraction: Optional[float] = None,
        subfeature_n_draws: int = 20,
        pca_components: int = 10,
        umap_components: int = 5,
        tabpfn_target_col: str = "event_type",
        tabpfn_model_path: str = "auto",
        tabpfn_device: str = "auto",
        tabpfn_n_estimators: int = 4,
        seed: int = 0,
    ):
        self.base_privacy_kwargs = dict(base_privacy_kwargs or {})
        # categorical_columns is representation-specific; never inherit it.
        self.base_privacy_kwargs.pop("categorical_columns", None)
        self.categorical_columns = list(categorical_columns or [])
        self.run_identity = run_identity
        self.run_subfeatures = run_subfeatures
        self.run_pca = run_pca
        self.run_umap = run_umap
        self.run_tabpfn = run_tabpfn
        self.subfeature_subset_size = subfeature_subset_size
        self.subfeature_fraction = subfeature_fraction
        self.subfeature_n_draws = subfeature_n_draws
        self.pca_components = pca_components
        self.umap_components = umap_components
        self.tabpfn_target_col = tabpfn_target_col
        self.tabpfn_model_path = tabpfn_model_path
        self.tabpfn_device = tabpfn_device
        self.tabpfn_n_estimators = tabpfn_n_estimators
        self.seed = seed

    # ------------------------------------------------------------------

    def _run_privacy(
        self, output: RepresentationOutput
    ) -> PrivacyResults:
        """Run a full PrivacyReport on one transformed representation."""
        report = PrivacyReport(
            categorical_columns=output.categorical_columns,
            **self.base_privacy_kwargs,
        )
        return report.run(output.real_df, output.synth_df)

    def _run_fixed(
        self,
        rep: Representation,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        real_target: Optional[np.ndarray],
    ) -> Optional[RepresentationPrivacy]:
        """Transform with ``rep`` and run privacy; return None if unavailable."""
        try:
            output = rep.fit_transform(real_df, synth_df, real_target=real_target)
        except RepresentationUnavailable as exc:
            logger.warning(f"RobustPrivacy: skipping '{rep.name}' — {exc}")
            self._skipped[rep.name] = str(exc)
            return None
        except Exception as exc:
            logger.error(f"RobustPrivacy: representation '{rep.name}' failed — {exc}")
            self._skipped[rep.name] = str(exc)
            return None

        logger.info(
            f"RobustPrivacy: running privacy report in '{rep.name}' space "
            f"({output.n_features} features)."
        )
        results = self._run_privacy(output)
        return RepresentationPrivacy(
            name=rep.name, n_features=output.n_features, results=results
        )

    # ------------------------------------------------------------------

    def run(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        target: Optional[np.ndarray] = None,
        target_col: Optional[str] = None,
    ) -> RobustPrivacyResults:
        """Run the privacy suite across all enabled representations.

        Parameters
        ----------
        real_df, synth_df :
            Feature frames.  If they still contain the TabPFN target column it is
            split out automatically; otherwise pass ``target`` explicitly.
        target :
            Optional supervised target array aligned to ``real_df`` rows, used by
            the TabPFN representation.
        target_col :
            Name of the target column to split out of the frames for TabPFN.
            Defaults to ``tabpfn_target_col``.
        """
        self._skipped: Dict[str, str] = {}
        results = RobustPrivacyResults()

        target_col = target_col or self.tabpfn_target_col

        # Split the target column out of the feature frames if present.
        real_feat, synth_feat = real_df, synth_df
        real_target = target
        if target_col in real_df.columns:
            if real_target is None:
                real_target = real_df[target_col].to_numpy()
            real_feat = real_df.drop(target_col)
            synth_feat = (
                synth_df.drop(target_col)
                if target_col in synth_df.columns
                else synth_df
            )

        feature_columns = list(real_feat.columns)
        cats = [c for c in self.categorical_columns if c in feature_columns]

        # --- Fixed representations ---------------------------------------
        fixed: List[Representation] = []
        if self.run_identity:
            fixed.append(IdentityRepresentation(categorical_columns=cats))
        if self.run_pca:
            fixed.append(
                PCARepresentation(
                    n_components=self.pca_components, categorical_columns=cats
                )
            )
        if self.run_umap:
            fixed.append(
                UMAPRepresentation(
                    n_components=self.umap_components,
                    categorical_columns=cats,
                    seed=self.seed,
                )
            )
        if self.run_tabpfn:
            fixed.append(
                TabPFNRepresentation(
                    target_col=target_col,
                    categorical_columns=cats,
                    model_path=self.tabpfn_model_path,
                    device=self.tabpfn_device,
                    n_estimators=self.tabpfn_n_estimators,
                    seed=self.seed,
                )
            )

        for rep in fixed:
            rep_privacy = self._run_fixed(rep, real_feat, synth_feat, real_target)
            if rep_privacy is not None:
                results.representations[rep.name] = rep_privacy

        # --- Variable subfeature subsets ---------------------------------
        if self.run_subfeatures and feature_columns:
            results.subfeatures = self._run_subfeatures(
                real_feat, synth_feat, feature_columns, cats
            )

        results.skipped = dict(self._skipped)
        logger.success(
            f"RobustPrivacy complete: {len(results.representations)} representations, "
            f"{'subfeature dist. computed' if results.subfeatures else 'no subfeatures'}, "
            f"{len(results.skipped)} skipped."
        )
        return results

    # ------------------------------------------------------------------

    def _resolve_subset_size(self, n_columns: int) -> int:
        if self.subfeature_fraction is not None:
            size = int(round(self.subfeature_fraction * n_columns))
        else:
            size = self.subfeature_subset_size
        return int(np.clip(size, 1, n_columns))

    def _run_subfeatures(
        self,
        real_feat: pl.DataFrame,
        synth_feat: pl.DataFrame,
        feature_columns: List[str],
        cats: List[str],
    ) -> Optional[SubfeaturePrivacy]:
        subset_size = self._resolve_subset_size(len(feature_columns))
        per_draw: List[Dict[str, float]] = []

        logger.info(
            f"RobustPrivacy: drawing {self.subfeature_n_draws} random subsets of "
            f"{subset_size}/{len(feature_columns)} features."
        )
        for i in range(self.subfeature_n_draws):
            rep = SubfeatureRepresentation(
                all_columns=feature_columns,
                subset_size=subset_size,
                seed=self.seed + i,
                categorical_columns=cats,
            )
            try:
                output = rep.fit_transform(real_feat, synth_feat)
                res = self._run_privacy(output)
            except Exception as exc:
                logger.warning(
                    f"RobustPrivacy: subfeature draw {i} failed — {exc}"
                )
                continue
            summary = {k: v for k, v in res.summary().items() if _is_number(v)}
            per_draw.append(summary)

        if not per_draw:
            logger.warning("RobustPrivacy: no successful subfeature draws.")
            return None

        distribution = _aggregate_draws(per_draw)
        return SubfeaturePrivacy(
            subset_size=subset_size,
            n_draws=len(per_draw),
            per_draw=per_draw,
            distribution=distribution,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        results: RobustPrivacyResults,
        metrics: Optional[Sequence[str]] = None,
        save_path: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """Grouped bar chart of key metrics across representations.

        Bars show the fixed representations; the subfeature draws are overlaid as
        a mean bar with a p5–p95 whisker so their spread is visible next to the
        point estimates.
        """
        table = results.comparison_table()
        if table.is_empty():
            logger.warning("RobustPrivacy.plot: no metrics to plot.")
            return None

        default_metrics = [
            "dcr_median_dcr_ratio",
            "dcr_privacy_at_risk",
            "reid_reidentification_rate",
        ]
        available = set(table["metric"].unique().to_list())
        if metrics is None:
            metrics = [m for m in default_metrics if m in available]
            if not metrics:
                metrics = sorted(available)[:6]

        fixed_reps = list(results.representations.keys())
        has_subfeat = results.subfeatures is not None
        rep_labels = fixed_reps + (["subfeature"] if has_subfeat else [])

        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            1, n_metrics, figsize=(max(4 * n_metrics, 5), 4), squeeze=False
        )
        axes = axes[0]

        x = np.arange(len(rep_labels))
        for ax, metric in zip(axes, metrics):
            heights, lo_err, hi_err = [], [], []
            for rep in fixed_reps:
                rp = results.representations[rep]
                val = rp.summary().get(f"privacy/{metric}", np.nan)
                heights.append(float(val) if _is_number(val) else np.nan)
                lo_err.append(0.0)
                hi_err.append(0.0)
            if has_subfeat:
                stats = results.subfeatures.distribution.get(f"privacy/{metric}", {})
                mean = stats.get("mean", np.nan)
                heights.append(mean)
                lo_err.append(max(0.0, mean - stats.get("p5", mean)))
                hi_err.append(max(0.0, stats.get("p95", mean) - mean))

            colors = ["steelblue"] * len(fixed_reps) + (
                ["darkorange"] if has_subfeat else []
            )
            ax.bar(
                x, heights, yerr=[lo_err, hi_err], capsize=4,
                color=colors, alpha=0.85, edgecolor="white",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(rep_labels, rotation=30, ha="right", fontsize=8)
            ax.set_title(metric, fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            "Privacy risk across representations "
            "(subfeature bar shows mean ± p5–p95)",
            fontsize=11,
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig


def _aggregate_draws(
    per_draw: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Compute mean/std/min/max/p5/p95 for each metric across draws."""
    all_metrics = sorted({k for d in per_draw for k in d})
    distribution: Dict[str, Dict[str, float]] = {}
    for metric in all_metrics:
        values = np.array(
            [d[metric] for d in per_draw if metric in d], dtype=np.float64
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        distribution[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "n": float(values.size),
        }
    return distribution

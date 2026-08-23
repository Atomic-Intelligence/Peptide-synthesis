"""Alternative feature representations for robust privacy evaluation.

Every distance-based privacy metric in this package (DCR, re-identification,
authenticity, DCR-signal MIA) reduces to *nearest-neighbour distances in some
feature space*.  The privacy risk you measure is therefore only as trustworthy
as the single feature space you happen to compute it in — a curse-of-
dimensionality artefact, an accident of feature scaling, or a lucky/unlucky
choice of which columns the attacker is assumed to hold can all move the numbers
substantially.

This module makes the *representation* an explicit, swappable object so the same
privacy report can be re-run across several spaces and the results compared:

- :class:`IdentityRepresentation` — the raw feature space (baseline).
- :class:`SubfeatureRepresentation` — a random subset of the columns; drawing
  many subsets yields a *distribution* of privacy risk rather than a single
  point estimate (see :class:`~robust_privacy.RobustPrivacyReport`).
- :class:`PCARepresentation` — a low-dimensional linear projection.
- :class:`UMAPRepresentation` — a low-dimensional non-linear projection.
- :class:`TabPFNRepresentation` — the learned embedding space of a pre-trained
  TabPFN foundation model, using a supervised target column.

Each representation is *fitted on the real data* and then applied to both the
real and synthetic frames, guaranteeing that both live in the same space so the
downstream nearest-neighbour distances are meaningful.  Optional dependencies
(``umap-learn``, ``tabpfn``) are imported lazily; a representation whose backend
is unavailable raises :class:`RepresentationUnavailable`, which the orchestrator
catches and skips.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import polars as pl
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.evaluation.privacy.preprocessing import FeatureProcessor, Scaler


class RepresentationUnavailable(RuntimeError):
    """Raised when a representation's backend (dependency, model, target) is missing.

    The orchestrator treats this as "skip this representation" rather than a hard
    failure, so an environment without ``umap-learn`` or ``tabpfn`` still produces
    a report for every other space.
    """


@dataclass
class RepresentationOutput:
    """A pair of transformed frames living in a common feature space.

    Attributes
    ----------
    real_df, synth_df :
        The transformed real and synthetic frames (same columns, same space).
    categorical_columns :
        Columns of the transformed frames that remain categorical.  Empty for
        projection/embedding spaces, which are fully continuous.
    n_features :
        Number of columns in the transformed space (convenience for logging).
    """

    real_df: pl.DataFrame
    synth_df: pl.DataFrame
    categorical_columns: List[str]

    @property
    def n_features(self) -> int:
        return self.real_df.width


class Representation(ABC):
    """Base class: fit a transform on real data, apply it to real and synthetic.

    Subclasses implement :meth:`fit_transform`.  A representation may consume an
    optional supervised target (aligned to the *rows* of the real / synthetic
    frames) — only :class:`TabPFNRepresentation` uses it, but the signature is
    shared so the orchestrator can call every representation uniformly.
    """

    #: Short identifier used in result keys, log lines, and plot labels.
    name: str = "identity"

    @abstractmethod
    def fit_transform(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        *,
        real_target: Optional[np.ndarray] = None,
        synth_target: Optional[np.ndarray] = None,
    ) -> RepresentationOutput:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class IdentityRepresentation(Representation):
    """Pass the frames through unchanged — the raw-feature baseline space."""

    name = "identity"

    def __init__(self, categorical_columns: Optional[Sequence[str]] = None):
        self.categorical_columns = list(categorical_columns or [])

    def fit_transform(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        *,
        real_target: Optional[np.ndarray] = None,
        synth_target: Optional[np.ndarray] = None,
    ) -> RepresentationOutput:
        cats = [c for c in self.categorical_columns if c in real_df.columns]
        return RepresentationOutput(real_df, synth_df, cats)


# ---------------------------------------------------------------------------
# Random subfeature subset
# ---------------------------------------------------------------------------


class SubfeatureRepresentation(Representation):
    """Restrict both frames to a fixed random subset of the columns.

    A single instance holds *one* drawn subset; the orchestrator constructs many
    instances with different seeds to build a distribution of privacy risk over
    the space of attacker-held feature subsets.

    Parameters
    ----------
    all_columns :
        The full column universe to sample from.
    subset_size :
        Number of columns in the subset.  Clamped to ``[1, len(all_columns)]``.
    seed :
        Seed for the column draw (deterministic per instance).
    categorical_columns :
        Columns considered categorical; the transformed ``categorical_columns``
        is the intersection of this list with the drawn subset.
    """

    name = "subfeature"

    def __init__(
        self,
        all_columns: Sequence[str],
        subset_size: int,
        seed: int,
        categorical_columns: Optional[Sequence[str]] = None,
    ):
        self.all_columns = list(all_columns)
        self.subset_size = int(np.clip(subset_size, 1, len(self.all_columns)))
        self.seed = seed
        self.categorical_columns = list(categorical_columns or [])
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(
            len(self.all_columns), size=self.subset_size, replace=False
        )
        self.selected_columns = [self.all_columns[i] for i in sorted(chosen_idx)]

    def fit_transform(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        *,
        real_target: Optional[np.ndarray] = None,
        synth_target: Optional[np.ndarray] = None,
    ) -> RepresentationOutput:
        cols = [c for c in self.selected_columns if c in real_df.columns]
        cats = [c for c in self.categorical_columns if c in cols]
        return RepresentationOutput(
            real_df.select(cols), synth_df.select(cols), cats
        )


# ---------------------------------------------------------------------------
# Linear / non-linear projections
# ---------------------------------------------------------------------------


class _ProjectionRepresentation(Representation):
    """Shared plumbing for projection spaces (scale + one-hot, then reduce).

    Numerical features are scaled and categoricals one-hot encoded via a
    :class:`FeatureProcessor` fitted on the real data; the resulting matrix is
    fed to a dimensionality reducer (also fitted on real) and both frames are
    projected.  Output columns are ``<name>_1 … <name>_k`` and fully continuous.
    """

    def __init__(
        self,
        n_components: int,
        categorical_columns: Optional[Sequence[str]] = None,
        scaler: Optional[Scaler] = None,
    ):
        self.n_components = int(n_components)
        self.categorical_columns = list(categorical_columns or [])
        self._scaler = scaler if scaler is not None else StandardScaler()

    def _make_reducer(self, n_components: int, n_samples: int):
        raise NotImplementedError

    def fit_transform(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        *,
        real_target: Optional[np.ndarray] = None,
        synth_target: Optional[np.ndarray] = None,
    ) -> RepresentationOutput:
        cats = [c for c in self.categorical_columns if c in real_df.columns]
        fp = FeatureProcessor(scaler=self._scaler, categorical_columns=cats)
        real_mat = fp.fit_transform(real_df)
        synth_mat = fp.transform(synth_df)

        max_components = min(real_mat.shape[0], real_mat.shape[1])
        n_components = int(np.clip(self.n_components, 1, max_components))
        if n_components < self.n_components:
            logger.warning(
                f"{self.name}: requested {self.n_components} components but only "
                f"{n_components} are feasible for a {real_mat.shape} matrix."
            )

        reducer = self._make_reducer(n_components, real_mat.shape[0])
        real_proj = np.asarray(reducer.fit_transform(real_mat))
        synth_proj = np.asarray(reducer.transform(synth_mat))

        col_names = [f"{self.name}_{i + 1}" for i in range(real_proj.shape[1])]
        real_out = pl.DataFrame(real_proj, schema=col_names)
        synth_out = pl.DataFrame(synth_proj, schema=col_names)
        return RepresentationOutput(real_out, synth_out, categorical_columns=[])


class PCARepresentation(_ProjectionRepresentation):
    """Linear PCA projection.  Preserves global structure and absolute scale,
    the safest reducer for distance-threshold privacy metrics."""

    name = "pca"

    def _make_reducer(self, n_components: int, n_samples: int):
        return PCA(n_components=n_components, random_state=0)


class UMAPRepresentation(_ProjectionRepresentation):
    """Non-linear UMAP projection.  Captures manifold structure but distorts
    absolute distances (it optimises local topology), so DCR / re-id thresholds
    in this space measure a different, more perceptual, notion of closeness.

    ``umap-learn`` is imported lazily; absence raises
    :class:`RepresentationUnavailable`.
    """

    name = "umap"

    def __init__(
        self,
        n_components: int,
        categorical_columns: Optional[Sequence[str]] = None,
        scaler: Optional[Scaler] = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        seed: int = 0,
    ):
        super().__init__(n_components, categorical_columns, scaler)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.seed = seed

    def _make_reducer(self, n_components: int, n_samples: int):
        try:
            import umap  # noqa: WPS433 (lazy optional dependency)
        except Exception as exc:  # pragma: no cover - env dependent
            raise RepresentationUnavailable(
                f"UMAP representation requires 'umap-learn' — {exc}"
            ) from exc
        # UMAP requires n_neighbors < n_samples.
        n_neighbors = int(np.clip(self.n_neighbors, 2, max(2, n_samples - 1)))
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            random_state=self.seed,
        )


# ---------------------------------------------------------------------------
# TabPFN embedding space
# ---------------------------------------------------------------------------


class TabPFNRepresentation(Representation):
    """Embed both frames with a pre-trained TabPFN foundation model.

    A :class:`TabPFNClassifier` is fitted on ``(real_features, real_target)`` and
    its internal per-row embeddings are extracted for both the real and synthetic
    frames via ``get_embeddings``.  This projects records into a learned space
    that encodes cross-feature interactions the model considers predictive of the
    target — a representation neither hand-crafted scaling nor linear projection
    can reproduce.

    The embedding tensor has shape ``(n_estimators, n_rows, embed_dim)``; the
    estimator axis is mean-pooled to a ``(n_rows, embed_dim)`` matrix.

    Parameters
    ----------
    target_col :
        Name of the supervised target column.  It is dropped from the feature
        set before embedding.  A target with fewer than two classes in the real
        data makes the representation unavailable.
    categorical_columns :
        Recorded for completeness; TabPFN performs its own preprocessing, so the
        raw numeric feature matrix is passed through.
    model_path :
        Path to a local TabPFN checkpoint, or ``"auto"`` to let TabPFN resolve
        one.  Falls back to the ``TABPFN_MODEL_PATH`` environment variable when
        ``"auto"``.  A missing/ungated model raises
        :class:`RepresentationUnavailable`.
    device :
        Torch device string (``"auto"``, ``"cpu"``, ``"cuda"``).
    n_estimators :
        Number of TabPFN ensemble members (kept small for speed; embeddings are
        pooled across them).
    """

    name = "tabpfn"

    def __init__(
        self,
        target_col: str,
        categorical_columns: Optional[Sequence[str]] = None,
        model_path: str = "auto",
        device: str = "auto",
        n_estimators: int = 4,
        seed: int = 0,
    ):
        self.target_col = target_col
        self.categorical_columns = list(categorical_columns or [])
        env_path = os.environ.get("TABPFN_MODEL_PATH")
        self.model_path = env_path if (model_path == "auto" and env_path) else model_path
        self.device = device
        self.n_estimators = n_estimators
        self.seed = seed

    def _feature_columns(self, df: pl.DataFrame) -> List[str]:
        return [c for c in df.columns if c != self.target_col]

    def fit_transform(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        *,
        real_target: Optional[np.ndarray] = None,
        synth_target: Optional[np.ndarray] = None,
    ) -> RepresentationOutput:
        try:
            from tabpfn import TabPFNClassifier  # noqa: WPS433
        except Exception as exc:  # pragma: no cover - env dependent
            raise RepresentationUnavailable(
                f"TabPFN representation requires 'tabpfn' — {exc}"
            ) from exc

        # Resolve target: prefer explicit arrays, else pull the column out.
        y_real = real_target
        if y_real is None and self.target_col in real_df.columns:
            y_real = real_df[self.target_col].to_numpy()
        if y_real is None:
            raise RepresentationUnavailable(
                f"TabPFN representation needs target '{self.target_col}' "
                "(pass real_target or include the column)."
            )
        if len(np.unique(y_real)) < 2:
            raise RepresentationUnavailable(
                f"TabPFN target '{self.target_col}' has <2 classes in the real "
                "data; cannot fit a classifier for embeddings."
            )

        feat_cols = self._feature_columns(real_df)
        x_real = real_df.select(feat_cols).to_numpy().astype(np.float64)
        x_synth = synth_df.select(
            [c for c in feat_cols if c in synth_df.columns]
        ).to_numpy().astype(np.float64)

        try:
            clf = TabPFNClassifier(
                device=self.device,
                model_path=self.model_path,
                n_estimators=self.n_estimators,
                random_state=self.seed,
                ignore_pretraining_limits=True,
            )
            clf.fit(x_real, y_real)
            emb_real = np.asarray(clf.get_embeddings(x_real, data_source="test"))
            emb_synth = np.asarray(clf.get_embeddings(x_synth, data_source="test"))
        except Exception as exc:  # pragma: no cover - env/model dependent
            raise RepresentationUnavailable(
                f"TabPFN embedding extraction failed — {exc}"
            ) from exc

        # (n_estimators, n_rows, embed_dim) -> mean-pool the estimator axis.
        if emb_real.ndim == 3:
            emb_real = emb_real.mean(axis=0)
            emb_synth = emb_synth.mean(axis=0)

        col_names = [f"tabpfn_{i + 1}" for i in range(emb_real.shape[1])]
        real_out = pl.DataFrame(emb_real, schema=col_names)
        synth_out = pl.DataFrame(emb_synth, schema=col_names)
        return RepresentationOutput(real_out, synth_out, categorical_columns=[])

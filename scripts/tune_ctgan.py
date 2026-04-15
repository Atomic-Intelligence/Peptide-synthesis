"""CT-GAN Hyperparameter Tuning
================================
Uses Optuna (TPE sampler) to search the CT-GAN hyperparameter space.
Each trial trains a fresh CTGANSynthetizationModel, generates synthetic
data, and evaluates it with a lightweight FidelityReport.  Results are
logged to MLflow; the best trial parameters are printed and saved as a
JSON artifact at the end.

The script reuses the standard pipeline config (Hydra) for all data-loading
settings.  Tuning-specific knobs can be overridden from the CLI with the
``++`` prefix syntax.

Usage
-----
    # from repo root
    python scripts/tune_ctgan.py

    # common overrides
    python scripts/tune_ctgan.py ++n_trials=50
    python scripts/tune_ctgan.py ++n_trials=30 ++n_synthetic_samples=300 event=hf
    python scripts/tune_ctgan.py ++tuning_experiment=my_exp ++study_name=my_study
    python scripts/tune_ctgan.py ++n_jobs=2   # parallel Optuna workers

Objective
---------
Minimise the mean Kolmogorov-Smirnov statistic across all marginal distributions
(lower = more faithful synthetic data):

    score = mean_ks

where ``mean_ks`` ∈ [0, 1] is the average per-feature KS statistic (0 = perfect).

Tuning parameters (override with ``++name=value``)
----------------------------------------------------
n_trials              int   30      Number of Optuna trials
n_synthetic_samples   int   500     Synthetic rows generated per trial for eval
tuning_experiment     str   ctgan_tuning   MLflow experiment name
study_name            str   ctgan_hparam_tuning   Optuna study name
n_jobs                int   1       Parallel Optuna workers (-1 = all cores)
pruning               bool  True    Enable Optuna MedianPruner
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so ``src.*`` imports work regardless of
# the directory the script is launched from.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlflow
import optuna
import polars as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.mlflow_utils import start_or_connect_mlflow_server
from src.models.ctgan.gan_interface import CTGANSynthetizationModel
from src.models.synthetization_model_interface import (
    DatasetMetadata,
    MlFlowTrainingRunInfo,
)

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

#: Architectural widths tried for generator / discriminator layers.
_DIM_CHOICES = [64, 128, 256, 512]

#: Batch sizes explored during search.
_BATCH_CHOICES = [100, 200, 500, 1000]

#: PAC parameter controls how the discriminator groups samples.
_PAC_CHOICES = [4, 8, 10, 16]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_real_data(cfg: DictConfig) -> tuple[pl.DataFrame, DatasetMetadata]:
    """Load and preprocess the real dataset identically to run_pipeline.py."""
    data_processor_partial = instantiate(cfg.data_processor, _partial_=True)
    df = pl.read_csv(cfg.paths.real_dataset_path)
    data_processor = data_processor_partial(dfs=[df])

    if cfg.sampled_patients_num > 0:
        data_processor = data_processor.sample_patients_with_preserved_ratios(
            n_patients=cfg.sampled_patients_num,
        )

    real_dataset, _ = (
        data_processor.filter_peptides(non_zero_threshold=cfg.non_zero_threshold)
        .split_event_control(event=cfg.event)
        .get_processed_data()
    )[0]

    patient_ids = (
        real_dataset[cfg.primary_key].unique().to_list()
        if cfg.primary_key in real_dataset.columns
        else []
    )
    patient_ids = [str(pid) for pid in patient_ids]
    real_dataset = real_dataset.drop(cfg.primary_key)

    peptide_ids = [col for col in real_dataset.columns if "peptide" in col.lower()]
    metadata = DatasetMetadata(
        peptide_ids=peptide_ids,
        patient_ids=patient_ids,
        full_dataset_path=cfg.paths.real_dataset_path,
        n_rows=len(patient_ids),
    )
    return real_dataset, metadata


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _build_gan_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample a complete set of CT-GAN hyperparameters for one trial."""
    gen_width = trial.suggest_categorical("generator_dim_size", _DIM_CHOICES)
    gen_depth = trial.suggest_int("generator_depth", 1, 3)
    dis_width = trial.suggest_categorical("discriminator_dim_size", _DIM_CHOICES)
    dis_depth = trial.suggest_int("discriminator_depth", 1, 3)

    return {
        "epochs": trial.suggest_int("epochs", 50, 300, step=50),
        "batch_size": trial.suggest_categorical("batch_size", _BATCH_CHOICES),
        "generator_dim": [gen_width] * gen_depth,
        "discriminator_dim": [dis_width] * dis_depth,
        "generator_lr": trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True),
        "discriminator_lr": trial.suggest_float(
            "discriminator_lr", 1e-5, 1e-3, log=True
        ),
        "discriminator_steps": trial.suggest_int("discriminator_steps", 1, 5),
        "pac": trial.suggest_categorical("pac", _PAC_CHOICES),
        "log_frequency": True,
        "verbose": False,
        "cuda": True,
    }


def make_objective(
    real_data: pl.DataFrame,
    metadata: DatasetMetadata,
    tuning_experiment: str,
    n_synthetic_samples: int,
):
    """Return a closure that Optuna calls for each trial."""
    # Lazy import to keep startup fast.
    from src.evaluation.fidelity.fidelity_report import FidelityReport

    # Drop string/categorical columns once so we don't repeat it every trial.
    _str_dtypes = (pl.Utf8, pl.String, pl.Categorical)
    _str_cols = [c for c in real_data.columns if real_data[c].dtype in _str_dtypes]
    real_eval = real_data.drop(_str_cols) if _str_cols else real_data

    def objective(trial: optuna.Trial) -> float:
        gan_params = _build_gan_params(trial)
        run_name = f"trial_{trial.number:04d}"

        ml_flow_info = MlFlowTrainingRunInfo(
            experiment_name=tuning_experiment,
            run_name=run_name,
        )
        model = CTGANSynthetizationModel(
            ml_flow_info=ml_flow_info,
            gan_params=OmegaConf.create(gan_params),
        )

        try:
            model.fit(real_data, metadata)
        except Exception as exc:
            logger.error(f"Trial {trial.number} — training failed: {exc}")
            raise optuna.exceptions.TrialPruned() from exc

        try:
            synthetic_data = model.generate(n_synthetic_patients=n_synthetic_samples)
        except Exception as exc:
            logger.error(f"Trial {trial.number} — generation failed: {exc}")
            raise optuna.exceptions.TrialPruned() from exc

        # Align columns and drop string cols.
        shared_cols = [c for c in real_eval.columns if c in synthetic_data.columns]
        if not shared_cols:
            logger.error(
                f"Trial {trial.number} — no shared columns between real and synthetic."
            )
            raise optuna.exceptions.TrialPruned()

        synth_eval = synthetic_data.select(shared_cols)
        real_aligned = real_eval.select(shared_cols)

        # Lightweight fidelity evaluation — skip slow bootstrap and classifier modules.
        fidelity_report = FidelityReport(
            run_corr_uncertainty=False,
            run_sparse_peptide=False,
            run_classifier=False,
        )
        try:
            results = fidelity_report.run(real_aligned, synth_eval)
        except Exception as exc:
            logger.error(f"Trial {trial.number} — evaluation failed: {exc}")
            raise optuna.exceptions.TrialPruned() from exc

        metrics = results.summary()

        # Objective: mean KS statistic across all marginal distributions.
        #   mean_ks ∈ [0, 1] — 0 = perfect marginal fidelity
        ks = float(metrics.get("fidelity/mean_ks_statistic", 1.0))
        score = ks

        # Log the composite score and key metrics as MLflow params on the trial run
        # (the training run already logged most params via CTGANSynthetizationModel.fit).
        try:
            exp = mlflow.get_experiment_by_name(tuning_experiment)
            if exp is not None:
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{model.ml_flow_info.run_name}'",
                    max_results=1,
                    output_format="pandas",
                )
                if not runs.empty:
                    run_id = runs.iloc[0].run_id
                    with mlflow.start_run(run_id=run_id):
                        mlflow.log_metric("tune/mean_ks", ks)
                        mlflow.log_param("tune/trial_number", trial.number)
        except Exception as log_exc:
            logger.warning(f"Could not log composite score to MLflow: {log_exc}")

        logger.info(
            f"Trial {trial.number:04d} | score={score:.4f} "
            f"ks={ks:.4f} | params={gan_params}"
        )
        return score

    return objective


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # Parse simple CLI overrides before Hydra initialises
    # ------------------------------------------------------------------
    # Tuning params are extracted from sys.argv (``++key=val`` syntax)
    # and removed so they don't confuse Hydra.
    tuning_keys = {
        "n_trials": 30,
        "n_synthetic_samples": 500,
        "tuning_experiment": "ctgan_tuning",
        "study_name": "ctgan_hparam_tuning",
        "n_jobs": 1,
        "pruning": True,
    }

    remaining_argv: list[str] = []
    overrides: dict[str, Any] = {}
    for arg in sys.argv[1:]:
        matched = False
        for key in tuning_keys:
            prefix = f"++{key}="
            if arg.startswith(prefix):
                raw = arg[len(prefix) :]
                default = tuning_keys[key]
                if isinstance(default, bool):
                    overrides[key] = raw.lower() not in ("false", "0", "no")
                elif isinstance(default, int):
                    overrides[key] = int(raw)
                else:
                    overrides[key] = raw
                matched = True
                break
        if not matched:
            remaining_argv.append(arg)

    tuning_cfg = {**tuning_keys, **overrides}
    sys.argv = [sys.argv[0]] + remaining_argv

    # ------------------------------------------------------------------
    # Load pipeline config via Hydra compose API
    # ------------------------------------------------------------------
    config_dir = str(_REPO_ROOT / "configs" / "training_and_inference_pipeline")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(
            config_name="pipeline",
            overrides=remaining_argv,
        )

    # ------------------------------------------------------------------
    # MLflow server
    # ------------------------------------------------------------------
    shutdown_hook = start_or_connect_mlflow_server(cfg.ml_flow_tracking_uri)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    logger.info("Loading real dataset…")
    real_data, metadata = load_real_data(cfg)
    logger.success(f"Dataset loaded — shape: {real_data.shape}")

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        if tuning_cfg["pruning"]
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=tuning_cfg["study_name"],
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
    )

    objective_fn = make_objective(
        real_data=real_data,
        metadata=metadata,
        tuning_experiment=tuning_cfg["tuning_experiment"],
        n_synthetic_samples=tuning_cfg["n_synthetic_samples"],
    )

    logger.info(
        f"Starting Optuna search — "
        f"n_trials={tuning_cfg['n_trials']}, "
        f"n_jobs={tuning_cfg['n_jobs']}, "
        f"experiment={tuning_cfg['tuning_experiment']}"
    )

    from tqdm import tqdm

    with tqdm(total=tuning_cfg["n_trials"], desc="Tuning CT-GAN") as pbar:

        def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            pbar.update(1)
            pbar.set_postfix({"best": f"{study.best_value:.4f}"})

        study.optimize(
            objective_fn,
            n_trials=tuning_cfg["n_trials"],
            n_jobs=tuning_cfg["n_jobs"],
            callbacks=[_callback],
            show_progress_bar=False,
        )

    # ------------------------------------------------------------------
    # Report best trial
    # ------------------------------------------------------------------
    best = study.best_trial
    best_gan_params = _build_gan_params_from_frozen(best)

    logger.success(f"Best trial: #{best.number}  score={best.value:.4f}")
    logger.success(f"Best Optuna params:\n{json.dumps(best.params, indent=2)}")
    logger.success(
        f"Best gan_params (ready for gan.yaml):\n{json.dumps(best_gan_params, indent=2)}"
    )

    # Save best params as a JSON artifact in a dedicated MLflow run.
    try:
        mlflow.set_experiment(tuning_cfg["tuning_experiment"])
        with mlflow.start_run(run_name="best_params_summary"):
            mlflow.log_metric("best_composite_score", best.value)
            mlflow.log_params({f"best/{k}": v for k, v in best.params.items()})
            with tempfile.TemporaryDirectory() as tmp:
                summary_path = Path(tmp) / "best_ctgan_params.json"
                summary = {
                    "trial_number": best.number,
                    "composite_score": best.value,
                    "optuna_params": best.params,
                    "gan_params": best_gan_params,
                }
                summary_path.write_text(json.dumps(summary, indent=2))
                mlflow.log_artifact(str(summary_path), "tuning_summary")
        logger.success("Best params saved to MLflow.")
    except Exception as exc:
        logger.warning(f"Could not save best params to MLflow: {exc}")

    shutdown_hook()


def _build_gan_params_from_frozen(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Reconstruct the full gan_params dict from a completed Optuna trial."""
    p = trial.params
    return {
        "epochs": p["epochs"],
        "batch_size": p["batch_size"],
        "generator_dim": [p["generator_dim_size"]] * p["generator_depth"],
        "discriminator_dim": [p["discriminator_dim_size"]] * p["discriminator_depth"],
        "generator_lr": p["generator_lr"],
        "discriminator_lr": p["discriminator_lr"],
        "discriminator_steps": p["discriminator_steps"],
        "pac": p["pac"],
        "log_frequency": True,
        "verbose": True,
        "cuda": True,
    }


if __name__ == "__main__":
    main()

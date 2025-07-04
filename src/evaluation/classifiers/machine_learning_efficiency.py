import logging

import polars as pl
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO)


def train_on_real_estimate_on_synthetic(
    classifier: SVC | Pipeline,
    event_type: str,
    real_df: pl.DataFrame,
    synthetic_df: pl.DataFrame,
    peptide_columns: list[str] = None,
    patient_id_subset: list[str] = None,  # todo handle this again
):
    logger.info(f"Building {event_type} classifier")

    real_df = real_df.fill_nan(0.0)
    synthetic_df = synthetic_df.fill_nan(0.0)

    real_df = real_df.filter(
        (pl.col("event_type") == event_type) | (pl.col("event_type") == "no_event")
    )

    real_df = real_df.with_columns(
        pl.col("event_type")
        .map_elements(lambda x: 1.0 if x != "no_event" else 0.0, return_dtype=float)
        .alias("event_label")
    )

    X_train = real_df.drop(["event_type", "event_label"]).to_numpy()
    y_train = real_df.select("event_label").to_numpy()

    # NOTE: Možda treba korisiti .ravel() na X_train i y_train -> Dakle dodati još jednu dimenziju X_train i/ili y_train
    classifier.fit(X_train, y_train.ravel())
    prediction_real = classifier.predict(X_train)
    probability_real = classifier.predict_proba(X_train)
    svm_score_real = classifier.decision_function(X_train)
    
    f1_real = f1_score(y_train, prediction_real)


    synthetic_df = synthetic_df.filter(
        (pl.col("event_type") == event_type) | (pl.col("event_type") == "no_event")
    )

    synthetic_df = synthetic_df.with_columns(
        pl.col("event_type")
        .map_elements(lambda x: 1.0 if x != "no_event" else 0.0, return_dtype=float)
        .alias("event_label")
    )

    X_eval = synthetic_df.drop(["event_type", "event_label"]).to_numpy()
    y_eval = synthetic_df.select("event_label").to_numpy()

    prediction_synth = classifier.predict(X_eval)
    probability_synth = classifier.predict_proba(X_eval)
    svm_score_synth = classifier.decision_function(X_eval)
    
    f1_synth = f1_score(y_eval, prediction_synth)


    report_dict = {
        "real": {
            "predicted_class": prediction_real.flatten(),
            "predicted_prob": probability_real.flatten(),
            "svm_score": svm_score_real.flatten(),
            "f1": f1_real,
        },
        "synthetic": {
            "predicted_class": prediction_synth.flatten(),
            "predicted_prob": probability_synth.flatten(),
            "svm_score": svm_score_synth.flatten(),
            "f1": f1_synth,
        },
    }

    logger.success(f"{event_type} classifier done!")

    return report_dict


def train_on_synthetic_test_on_real(
    classifier: SVC | Pipeline,
    event_type: str,
    real_df: pl.DataFrame,
    synthetic_df: pl.DataFrame,
    peptide_columns: list[str] = None,
):
    logger.info(f"Building {event_type} synthetic data SVM")

    real_df = real_df.fill_nan(0.0)
    synthetic_df = synthetic_df.fill_nan(0.0)

    real_df = real_df.filter(
        (pl.col("event_type") == event_type) | (pl.col("event_type") == "no_event")
    )

    real_df = real_df.with_columns(
        pl.col("event_type")
        .map_elements(lambda x: 1.0 if x != "no_event" else 0.0, return_dtype=float)
        .alias("event_label")
    )

    synthetic_df = synthetic_df.filter(
        (pl.col("event_type") == event_type) | (pl.col("event_type") == "no_event")
    )

    synthetic_df = synthetic_df.with_columns(
        pl.col("event_type")
        .map_elements(lambda x: 1.0 if x != "no_event" else 0.0, return_dtype=float)
        .alias("event_label")
    )

    X_train = synthetic_df.drop(["event_type", "event_label"]).to_numpy()
    y_train = synthetic_df.select("event_label").to_numpy()

    classifier.fit(X_train, y_train.ravel())

    prediction_synth = classifier.predict(X_train)
    probability_synth = classifier.predict_proba(X_train)
    svm_score_synth = classifier.decision_function(X_train)

    f1_synth = f1_score(y_train, prediction_synth)

    X_eval = real_df.drop(["event_type", "event_label"]).to_numpy()
    y_eval = real_df.select("event_label").to_numpy()

    prediction_real = classifier.predict(X_eval)
    probability_real = classifier.predict_proba(X_eval)
    svm_score_real = classifier.decision_function(X_eval)

    f1_real = f1_score(y_eval, prediction_real)

    report_dict = {
        "real": {
            "predicted_class": prediction_real.flatten(),
            "predicted_prob": probability_real.flatten(),
            "svm_score": svm_score_real.flatten(),
            "f1": f1_real,
        },
        "synthetic": {
            "predicted_class": prediction_synth.flatten(),
            "predicted_prob": probability_synth.flatten(),
            "svm_score": svm_score_synth.flatten(),
            "f1": f1_synth,
        },
    }

    logger.success(f"{event_type} synthetic SVM done!")

    return report_dict

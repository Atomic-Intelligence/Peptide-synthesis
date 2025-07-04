from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import mlflow
import numpy as np
from scipy.stats import mannwhitneyu, false_discovery_control
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score


from v0.src import get_peptide_columns


def svm_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict[str, list],
    event_type: str,
    cv: int = 5,
    verbose: int = 2,
    n_jobs: int = 16,
) -> GridSearchCV:
    """
    Performs GridSearchCV for SVM with specified parameters.

    Args:
        X (numpy.ndarray or pandas.DataFrame): Feature matrix.
        y (numpy.ndarray or pandas.Series): Target vector.
        param_grid (dict): Dictionary of parameters to search over.
        cv (int): Number of cross-validation folds.
        verbose (int): Verbosity level.
        n_jobs (int): Number of jobs to run in parallel (-1 means using all processors).

    Returns:
        sklearn.model_selection._search.GridSearchCV: Fitted GridSearchCV object.
    """
    # NOTE: Alternatively, use RobustScaler() instead of StandardScaler()
    classifier = Pipeline(steps=[("scaler", RobustScaler()), ("svm", SVC())])

    logger.info(
        f"Initializing grid search for event: {event_type} with classifier{classifier}"
    )
    grid_search = GridSearchCV(
        classifier, param_grid, cv=cv, verbose=verbose, n_jobs=n_jobs, scoring="f1"
    )
    logger.info(f"Performing grid search...")
    grid_search.fit(X, y)
    logger.success(f"Grid search done for {event_type} patients!")
    return grid_search


def evaluate_grid_search(
    grid_search: GridSearchCV, X_test: np.ndarray, y_test: np.ndarray, event_type: str
) -> dict:
    """
    Evaluates the best estimator from GridSearchCV.

    Args:
        grid_search (sklearn.model_selection._search.GridSearchCV): Fitted GridSearchCV object.
        X_test (numpy.ndarray or pandas.DataFrame): Test feature matrix.
        y_test (numpy.ndarray or pandas.Series): Test target vector.

    Returns:
        None. Prints the best parameters and classification report.
    """
    logger.success(
        f"Grid search for event {event_type} resulted with best parameters: {grid_search.best_params_}"
    )
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mlflow.log_metric(f"{event_type}/f1_score", f1_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    mlflow.log_figure(fig, f"{event_type}/confusion_matrix.png")
    logger.info(report)

    mlflow.log_dict(report, f"{event_type}/best_model_performance.json")
    mlflow.log_dict(grid_search.best_params_, f"{event_type}/best_params.json")
    return report


def prepare_data(
    full_dataframe: pl.DataFrame, event_type: str, zero_threshold: float = 0.4
) -> tuple[np.ndarray]:
    all_peptide_columns = get_peptide_columns(full_dataframe)
    peptide_df = (
        full_dataframe.select(all_peptide_columns + ["event_type"])
        .fill_nan(0.0)
        .fill_null(0.0)
    )
    peptide_df = peptide_df.filter(pl.col("event_type").is_in(["no_event", event_type]))
    zero_percentages = peptide_df.select(
        [(pl.col(col) == 0).sum() / pl.count() for col in all_peptide_columns]
    ).to_dicts()[0]

    cols_to_keep = [
        col
        for col, zero_percentage in zero_percentages.items()
        if zero_percentage <= zero_threshold
    ]

    no_event_rows = (
        peptide_df.select(cols_to_keep + ["event_type"])
        .filter(pl.col("event_type") == "no_event")
        .drop("event_type")
        .to_numpy()
    )
    event_rows = (
        peptide_df.select(cols_to_keep + ["event_type"])
        .filter(pl.col("event_type") != "no_event")
        .drop("event_type")
        .to_numpy()
    )
    _, pvalues = mannwhitneyu(no_event_rows, event_rows)
    pvalues = false_discovery_control(pvalues)

    cols_to_keep = [col for col, pval in zip(cols_to_keep, pvalues) if pval < 0.05]
    mlflow.log_dict(
        {"num_peptides": len(cols_to_keep), "peptide_ids": cols_to_keep},
        f"{event_type}/selected_peptide_columns.json",
    )
    peptide_array: np.ndarray = peptide_df.select(cols_to_keep).to_numpy()
    event_array: np.ndarray = (
        peptide_df.with_columns(
            val=pl.when(event_type="no_event").then(0.0).otherwise(1.0)
        )
        .select("val")
        .to_numpy()
    )
    logger.info(f"Peptide shape {peptide_array.shape}, Event shape {event_array.shape}")
    return peptide_array, event_array.ravel()


def main(param_grid: dict[str, list], data_path: Path | str) -> None:
    mlflow.log_dict(param_grid, "param_grid.json")
    events = {"ckd": "Chronic-Kidney-Disease", "hf": "Heart-Failure"}
    full_dataframe = pl.read_csv(data_path)
    for event, event_display_name in events.items():
        X, y = prepare_data(full_dataframe, event_type=event)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
        )
        grid_search = svm_grid_search(
            X_train, y_train, param_grid, event_type=event_display_name
        )
        logger.info(
            f"Starting grid search evaluation for {event_display_name} classifier."
        )
        evaluate_grid_search(grid_search, X_test, y_test, event_type=event_display_name)


# Example usage with the Iris dataset:
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://10.100.111.210:44555")
    try:
        mlflow.set_experiment("Peptide_SVM_Grid_Search")
    except:
        mlflow.create_experiment("Peptide_SVM_Grid_Search")
        mlflow.set_experiment("Peptide_SVM_Grid_Search")

    data_path = Path("/data1/prostrat-ai/data/peptide_and_clinical_data_v2.csv")

    param_grid = {
        "scaler": [
            StandardScaler(),
            RobustScaler(),
            QuantileTransformer(output_distribution="uniform"),
            QuantileTransformer(output_distribution="normal"),
        ],
        "svm__C": [0.1, 1, 10, 100],
        "svm__kernel": ["linear", "rbf", "poly"],
        "svm__gamma": ["scale", "auto", 0.1, 1],
        "svm__degree": [2, 3, 4],  # only relevant for poly kernel
        "svm__class_weight": [None, "balanced"],
    }

    with mlflow.start_run() as run:
        main(param_grid=param_grid, data_path=data_path)

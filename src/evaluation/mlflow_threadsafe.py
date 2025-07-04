import mlflow
import threading
from functools import wraps
from multiprocessing import RLock

# Create a lock for MLflow operations
mlflow_lock = RLock()


def mlflow_safe(func):
    """
    Decorator to make MLflow logging functions thread-safe by using a mutex lock.
    Can be applied to any function that interacts with MLflow.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with mlflow_lock:
            return func(*args, **kwargs)

    return wrapper


# Example usage for common MLflow logging functions
@mlflow_safe
def safe_log_param(key, value):
    mlflow.log_param(key, value)


@mlflow_safe
def safe_log_metric(key, value, step=None):
    mlflow.log_metric(key, value, step=step)


@mlflow_safe
def safe_log_artifact(local_path, artifact_path=None):
    mlflow.log_artifact(local_path, artifact_path)


@mlflow_safe
def safe_log_dict(dictionary, artifact_file):
    mlflow.log_dict(dictionary, artifact_file)


@mlflow_safe
def safe_log_figure(figure, artifact_file):
    mlflow.log_figure(figure, artifact_file)


# For starting and ending runs
@mlflow_safe
def safe_start_run(
    run_id=None, experiment_id=None, run_name=None, nested=False, tags=None
):
    return mlflow.start_run(
        run_id=run_id,
        experiment_id=experiment_id,
        run_name=run_name,
        nested=nested,
        tags=tags,
    )


# You can use this as a context manager
class SafeMlflowRun:
    def __init__(
        self, run_id=None, experiment_id=None, run_name=None, nested=False, tags=None
    ):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.run_name = run_name
        self.nested = nested
        self.tags = tags

    def __enter__(self):
        with mlflow_lock:
            self.run = mlflow.start_run(
                run_id=self.run_id,
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                nested=self.nested,
                tags=self.tags,
            )
            return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        with mlflow_lock:
            mlflow.end_run()

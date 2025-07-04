import hydra
from loguru import logger
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def make_svm_scaler_pipeline(steps: dict[str, any]) -> Pipeline:
    """
    Creates a scikit-learn Pipeline from a dictionary configuration.
    Args:
        steps: A dictionary where keys are step names and values are the
               configurations for each step. Order is preserved.

    Returns:
        A scikit-learn Pipeline object with the configured steps.

    Example:
        steps = {
            "scaler": {"_target_": "sklearn.preprocessing.StandardScaler"},
            "svm": {"_target_": "sklearn.svm.SVC", "C": 1.0}
        }

        pipeline = pipeline_factory(steps)
        # Returns Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=1.0))])
    """
    # Convert dictionary to list of (name, object) tuples
    pipeline_steps = []
    for name, step_instance in steps.items():
        # Instantiate the component using Hydra
        logger.info(f"Instantiating classifier step: {name}: {step_instance}")
        # step_instance = hydra.utils.instantiate(config)
        pipeline_steps.append((name, step_instance))

    return Pipeline(steps=pipeline_steps)

from typing import Callable
import mlflow
import requests
from src.logger import setup_logger

import subprocess

logger = setup_logger()


LOCAL_MLFLOW_SERVER_HOST = "127.0.0.1"
LOCAL_MLFLOW_SERVER_PORT = "5000"
LOCAL_MLFLOW_SERVER_URI = f"http://{LOCAL_MLFLOW_SERVER_HOST}:{LOCAL_MLFLOW_SERVER_PORT}"


def start_or_connect_mlflow_server(uri: str) -> Callable[[], None]:
    try:
        _ = requests.get(f"{uri}/api/health")
        logger.info(f"You can view your experiments at {uri}")
        return lambda: None
        
        
    except Exception:
        logger.info(
            f"You can view your experiments at {LOCAL_MLFLOW_SERVER_URI}."
        )

        subprocess.Popen(
            ["mlflow", "server", "--host", LOCAL_MLFLOW_SERVER_HOST, "--port", LOCAL_MLFLOW_SERVER_PORT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        mlflow.set_tracking_uri(LOCAL_MLFLOW_SERVER_URI)
        return lambda: subprocess.run(f"lsof -t -i :{LOCAL_MLFLOW_SERVER_PORT} | xargs kill -9", shell=True)
"""Shared MLflow configuration for DagsHub experiment tracking."""

import os

import mlflow


DEFAULT_DAGSHUB_OWNER = "cescidy"
DEFAULT_DAGSHUB_REPOSITORY = "diabetes-prediction"


def configure_mlflow() -> None:
    """Point MLflow at the configured DagsHub repository and experiment.

    MLFLOW_TRACKING_URI can override the default DagsHub endpoint for local or
    alternative tracking servers. Authentication is supplied through
    MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD.
    """
    owner = os.getenv("DAGSHUB_REPO_OWNER", DEFAULT_DAGSHUB_OWNER)
    repository = os.getenv("DAGSHUB_REPO_NAME", DEFAULT_DAGSHUB_REPOSITORY)
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"https://dagshub.com/{owner}/{repository}.mlflow",
    )
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "diabetes-predictor")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

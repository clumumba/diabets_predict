"""Evaluate the trained classifier against the held-out test data."""

import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from diabetesPredictor import logger
from diabetesPredictor.config.configuration import metrics_path, model_path, test_data


def evaluate_model(model, data: pd.DataFrame) -> dict[str, float]:
    if "Outcome" not in data.columns:
        raise ValueError("Test data must contain an 'Outcome' column")
    features, target = data.drop(columns="Outcome"), data["Outcome"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }


def main() -> None:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    metrics = evaluate_model(model, pd.read_csv(test_data))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Evaluation metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()

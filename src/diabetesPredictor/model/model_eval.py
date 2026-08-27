"""Evaluate the trained classifier against the held-out test data."""

import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from diabetesPredictor import logger
from diabetesPredictor.config.configuration import metrics_path, model_path, x_test_data, y_test_data


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def main() -> None:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    X_test = pd.read_csv(x_test_data)
    y_test = pd.read_csv(y_test_data).squeeze("columns")
    metrics = evaluate_model(model, X_test, y_test) 
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Evaluation metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()

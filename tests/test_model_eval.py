"""Tests for model evaluation module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from diabetesPredictor.model.model_eval import evaluate_model


class TestEvaluateModel:
    """Test model evaluation functionality."""

    def test_evaluate_model_success(self, sample_diabetes_data):
        """Test successful model evaluation."""
        X_test = sample_diabetes_data.drop("Outcome", axis=1)
        y_test = sample_diabetes_data["Outcome"]

        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7],
            [0.95, 0.05], [0.1, 0.9], [0.85, 0.15], [0.4, 0.6]
        ])

        metrics = evaluate_model(mock_model, X_test, y_test)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_evaluate_model_returns_floats(self, sample_diabetes_data):
        """Test that all metrics are returned as floats."""
        X_test = sample_diabetes_data.drop("Outcome", axis=1)
        y_test = sample_diabetes_data["Outcome"]

        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7],
            [0.95, 0.05], [0.1, 0.9], [0.85, 0.15], [0.4, 0.6]
        ])

        metrics = evaluate_model(mock_model, X_test, y_test)

        for metric_value in metrics.values():
            assert isinstance(metric_value, float)

    def test_evaluate_model_metrics_in_valid_range(self, sample_diabetes_data):
        """Test that metrics are in valid ranges."""
        X_test = sample_diabetes_data.drop("Outcome", axis=1)
        y_test = sample_diabetes_data["Outcome"]

        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7],
            [0.95, 0.05], [0.1, 0.9], [0.85, 0.15], [0.4, 0.6]
        ])

        metrics = evaluate_model(mock_model, X_test, y_test)

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_evaluate_model_perfect_predictions(self, sample_diabetes_data):
        """Test evaluation with perfect predictions."""
        X_test = sample_diabetes_data.drop("Outcome", axis=1)
        y_test = sample_diabetes_data["Outcome"]

        mock_model = Mock()
        mock_model.predict.return_value = y_test.values
        mock_model.predict_proba.return_value = np.array([
            [0.9, 0.1] if y == 0 else [0.1, 0.9] for y in y_test
        ])

        metrics = evaluate_model(mock_model, X_test, y_test)

        assert metrics["accuracy"] == 1.0

    def test_evaluate_model_all_wrong_predictions(self, sample_diabetes_data):
        """Test evaluation with all wrong predictions."""
        X_test = sample_diabetes_data.drop("Outcome", axis=1)
        y_test = sample_diabetes_data["Outcome"]

        mock_model = Mock()
        mock_model.predict.return_value = 1 - y_test.values
        mock_model.predict_proba.return_value = np.array([
            [0.1, 0.9] if y == 0 else [0.9, 0.1] for y in y_test
        ])

        metrics = evaluate_model(mock_model, X_test, y_test)

        assert metrics["accuracy"] == 0.0

    def test_evaluate_model_binary_classification(self, sample_diabetes_data):
        """Test that metrics are appropriate for binary classification."""
        X_test = sample_diabetes_data.drop("Outcome", axis=1)
        y_test = sample_diabetes_data["Outcome"]

        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7],
            [0.95, 0.05], [0.1, 0.9], [0.85, 0.15], [0.4, 0.6]
        ])

        metrics = evaluate_model(mock_model, X_test, y_test)

        required_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        assert all(metric in metrics for metric in required_metrics)


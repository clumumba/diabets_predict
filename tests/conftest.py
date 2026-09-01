"""Pytest configuration and fixtures."""

import sys
import pytest
import pandas as pd
from pathlib import Path

# Add src directory to Python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def sample_diabetes_data():
    """Create sample diabetes dataset for testing."""
    return pd.DataFrame({
        "Pregnancies": [1, 2, 3, 1, 0, 2, 1, 3],
        "Glucose": [100, 120, 110, 90, 85, 130, 105, 140],
        "BloodPressure": [70, 80, 75, 65, 60, 85, 72, 88],
        "SkinThickness": [20, 25, 22, 18, 15, 28, 20, 30],
        "Insulin": [0, 100, 50, 0, 0, 120, 30, 150],
        "BMI": [25.0, 27.5, 26.0, 24.0, 22.5, 28.0, 26.5, 32.0],
        "DiabetesPedigreeFunction": [0.5, 0.6, 0.55, 0.45, 0.4, 0.7, 0.52, 0.8],
        "Age": [30, 35, 28, 25, 22, 40, 33, 45],
        "Outcome": [0, 1, 0, 0, 0, 1, 0, 1],
    })


@pytest.fixture
def sample_diabetes_data_no_outcome():
    """Create sample diabetes dataset without Outcome column."""
    return pd.DataFrame({
        "Pregnancies": [1, 2, 3],
        "Glucose": [100, 120, 110],
        "BloodPressure": [70, 80, 75],
        "SkinThickness": [20, 25, 22],
        "Insulin": [0, 100, 50],
        "BMI": [25.0, 27.5, 26.0],
        "DiabetesPedigreeFunction": [0.5, 0.6, 0.55],
        "Age": [30, 35, 28],
    })


@pytest.fixture
def mock_model():
    """Create a mock sklearn model."""
    from unittest.mock import Mock
    model = Mock()
    model.predict.return_value = [0, 1, 0]
    model.predict_proba.return_value = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
    return model

"""Tests for data preparation module."""

import pytest
import pandas as pd
import yaml
from pathlib import Path

from diabetesPredictor.data.data_prep import load_data, load_params, split


class TestLoadData:
    """Test data loading in data_prep module."""

    def test_load_data_success(self, tmp_path, sample_diabetes_data):
        """Test successful data loading."""
        csv_file = tmp_path / "data.csv"
        sample_diabetes_data.to_csv(csv_file, index=False)

        loaded = load_data(csv_file)

        assert isinstance(loaded, pd.DataFrame)
        assert loaded.shape == sample_diabetes_data.shape
        pd.testing.assert_frame_equal(loaded, sample_diabetes_data)

    def test_load_data_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_data(Path("/non/existent/file.csv"))


class TestLoadParams:
    """Test parameter loading from YAML."""

    def test_load_params_success(self, tmp_path):
        """Test successful parameter loading."""
        params_file = tmp_path / "params.yml"
        params_content = {
            "data_prep": {
                "test_size": 0.2,
                "random_state": 42,
            }
        }
        with open(params_file, "w") as f:
            yaml.dump(params_content, f)

        params = load_params(params_file)

        assert params["test_size"] == 0.2
        assert params["random_state"] == 42

    def test_load_params_missing_file(self):
        """Test error when params file doesn't exist."""
        with pytest.raises(RuntimeError, match="Invalid data-preparation parameters"):
            load_params(Path("/non/existent/params.yml"))

    def test_load_params_invalid_yaml(self, tmp_path):
        """Test error with invalid YAML syntax."""
        params_file = tmp_path / "invalid.yml"
        params_file.write_text("invalid: {yaml: [content:")

        with pytest.raises(RuntimeError, match="Invalid data-preparation parameters"):
            load_params(params_file)

    def test_load_params_missing_data_prep_section(self, tmp_path):
        """Test error when data_prep section is missing."""
        params_file = tmp_path / "params.yml"
        params_content = {"other_section": {"key": "value"}}
        with open(params_file, "w") as f:
            yaml.dump(params_content, f)

        with pytest.raises(RuntimeError, match="Invalid data-preparation parameters"):
            load_params(params_file)

    def test_load_params_type_conversion(self, tmp_path):
        """Test that parameters are converted to correct types."""
        params_file = tmp_path / "params.yml"
        params_content = {
            "data_prep": {
                "test_size": "0.25",
                "random_state": "123",
            }
        }
        with open(params_file, "w") as f:
            yaml.dump(params_content, f)

        params = load_params(params_file)

        assert isinstance(params["test_size"], float)
        assert isinstance(params["random_state"], int)
        assert params["test_size"] == 0.25
        assert params["random_state"] == 123


class TestSplit:
    """Test train/test split functionality."""

    def test_split_success(self, sample_diabetes_data):
        """Test successful data split."""
        X_train, X_test, y_train, y_test = split(sample_diabetes_data, test_size=0.25, random_state=42)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

        assert "Outcome" not in X_train.columns
        assert "Outcome" not in X_test.columns

        assert len(X_train) + len(X_test) == len(sample_diabetes_data)
        assert len(y_train) + len(y_test) == len(sample_diabetes_data)

    def test_split_test_size(self, sample_diabetes_data):
        """Test that split respects test_size parameter."""
        X_train, X_test, y_train, y_test = split(sample_diabetes_data, test_size=0.3, random_state=42)

        total_size = len(sample_diabetes_data)
        expected_test_size_approx = int(total_size * 0.3)

        # Allow some tolerance due to rounding
        assert abs(len(X_test) - expected_test_size_approx) <= 1
        assert len(X_train) + len(X_test) == total_size

    def test_split_reproducibility(self, sample_diabetes_data):
        """Test that split is reproducible with same random_state."""
        X_train1, X_test1, y_train1, y_test1 = split(sample_diabetes_data, test_size=0.2, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split(sample_diabetes_data, test_size=0.2, random_state=42)

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)

    def test_split_missing_outcome_column(self, sample_diabetes_data_no_outcome):
        """Test error when Outcome column is missing."""
        with pytest.raises(ValueError, match="Outcome"):
            split(sample_diabetes_data_no_outcome, test_size=0.2, random_state=42)

    def test_split_larger_dataset(self):
        """Test split with larger dataset for more predictable sizes."""
        large_data = pd.DataFrame({
            "col1": range(100),
            "col2": range(100, 200),
            "Outcome": [i % 2 for i in range(100)],
        })

        X_train, X_test, y_train, y_test = split(large_data, test_size=0.2, random_state=42)

        assert len(X_test) == 20
        assert len(X_train) == 80


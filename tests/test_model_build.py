"""Tests for model building module."""

import pytest
import pickle
import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from diabetesPredictor.model.model_build import load_params, load_data, train_model, save_model


class TestLoadParams:
    """Test parameter loading for model training."""

    def test_load_params_success(self, tmp_path):
        """Test successful model parameter loading."""
        params_file = tmp_path / "params.yml"
        params_content = {
            "model": {
                "n_estimators": 100,
                "random_state": 42,
            }
        }
        with open(params_file, "w") as f:
            yaml.dump(params_content, f)

        params = load_params(params_file)

        assert params["n_estimators"] == 100
        assert params["random_state"] == 42

    def test_load_params_missing_file(self):
        """Test error when params file doesn't exist."""
        with pytest.raises(RuntimeError, match="Invalid model parameters"):
            load_params(Path("/non/existent/params.yml"))

    def test_load_params_invalid_yaml(self, tmp_path):
        """Test error with invalid YAML syntax."""
        params_file = tmp_path / "invalid.yml"
        params_file.write_text("invalid: {yaml: [content:")

        with pytest.raises(RuntimeError, match="Invalid model parameters"):
            load_params(params_file)

    def test_load_params_missing_model_section(self, tmp_path):
        """Test error when model section is missing."""
        params_file = tmp_path / "params.yml"
        params_content = {"other_section": {"key": "value"}}
        with open(params_file, "w") as f:
            yaml.dump(params_content, f)

        with pytest.raises(RuntimeError, match="Invalid model parameters"):
            load_params(params_file)

    def test_load_params_type_conversion(self, tmp_path):
        """Test that parameters are converted to correct types."""
        params_file = tmp_path / "params.yml"
        params_content = {
            "model": {
                "n_estimators": "50",
                "random_state": "99",
            }
        }
        with open(params_file, "w") as f:
            yaml.dump(params_content, f)

        params = load_params(params_file)

        assert isinstance(params["n_estimators"], int)
        assert isinstance(params["random_state"], int)
        assert params["n_estimators"] == 50
        assert params["random_state"] == 99


class TestLoadData:
    """Test data loading for model training."""

    def test_load_data_success(self, tmp_path):
        """Test successful data loading."""
        csv_file = tmp_path / "train_data.csv"
        test_data = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
        })
        test_data.to_csv(csv_file, index=False)

        loaded = load_data(csv_file)

        assert isinstance(loaded, pd.DataFrame)
        pd.testing.assert_frame_equal(loaded, test_data)

    def test_load_data_file_not_found(self):
        """Test error when data file doesn't exist."""
        with pytest.raises(RuntimeError, match="Error loading data"):
            load_data(Path("/non/existent/data.csv"))

    def test_load_data_permission_error(self, tmp_path):
        """Test error with permission denied."""
        csv_file = tmp_path / "test.csv"
        test_data = pd.DataFrame({"col1": [1, 2, 3]})
        test_data.to_csv(csv_file, index=False)
        
        # Try to load from a directory instead of a file
        with pytest.raises(RuntimeError, match="Error loading data"):
            load_data(tmp_path)


class TestTrainModel:
    """Test model training functionality."""

    def test_train_model_success(self, sample_diabetes_data):
        """Test successful model training."""
        X = sample_diabetes_data.drop("Outcome", axis=1)
        y = sample_diabetes_data["Outcome"]

        model = train_model(X, y, n_estimators=10, random_state=42)

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 10
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_train_model_makes_predictions(self, sample_diabetes_data):
        """Test that trained model can make predictions."""
        X = sample_diabetes_data.drop("Outcome", axis=1)
        y = sample_diabetes_data["Outcome"]

        model = train_model(X, y, n_estimators=10, random_state=42)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)

    def test_train_model_predict_proba_shape(self, sample_diabetes_data):
        """Test that trained model can output probabilities with correct shape."""
        X = sample_diabetes_data.drop("Outcome", axis=1)
        y = sample_diabetes_data["Outcome"]

        model = train_model(X, y, n_estimators=10, random_state=42)
        probas = model.predict_proba(X)

        assert probas.shape[0] == len(y)
        assert probas.shape[1] == 2

    def test_train_model_reproducibility(self, sample_diabetes_data):
        """Test that model training is reproducible with same random_state."""
        X = sample_diabetes_data.drop("Outcome", axis=1)
        y = sample_diabetes_data["Outcome"]

        model1 = train_model(X, y, n_estimators=5, random_state=99)
        model2 = train_model(X, y, n_estimators=5, random_state=99)

        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        assert all(pred1 == pred2)


class TestSaveModel:
    """Test model saving functionality."""

    def test_save_model_success(self, tmp_path, sample_diabetes_data):
        """Test successful model saving."""
        X = sample_diabetes_data.drop("Outcome", axis=1)
        y = sample_diabetes_data["Outcome"]
        model = train_model(X, y, n_estimators=5, random_state=42)

        model_file = tmp_path / "models" / "test_model.pkl"
        save_model(model, model_file)

        assert model_file.exists()

    def test_save_model_creates_parent_dirs(self, tmp_path, sample_diabetes_data):
        """Test that parent directories are created."""
        X = sample_diabetes_data.drop("Outcome", axis=1)
        y = sample_diabetes_data["Outcome"]
        model = train_model(X, y, n_estimators=5, random_state=42)

        nested_path = tmp_path / "a" / "b" / "c" / "model.pkl"
        save_model(model, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_model_can_be_loaded(self, tmp_path, sample_diabetes_data):
        """Test that saved model can be loaded and used."""
        X = sample_diabetes_data.drop("Outcome", axis=1)
        y = sample_diabetes_data["Outcome"]
        original_model = train_model(X, y, n_estimators=5, random_state=42)

        model_file = tmp_path / "model.pkl"
        save_model(original_model, model_file)

        with open(model_file, "rb") as f:
            loaded_model = pickle.load(f)

        original_pred = original_model.predict(X)
        loaded_pred = loaded_model.predict(X)

        assert all(original_pred == loaded_pred)


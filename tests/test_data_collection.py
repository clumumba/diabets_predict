"""Tests for data collection module."""

import pandas as pd
import pytest
from pathlib import Path

from diabetesPredictor.data.data_collection import load_data, save_data


class TestLoadData:
    """Test data loading functionality."""

    def test_load_data_success(self, tmp_path):
        """Test successful data loading from CSV."""
        csv_file = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            "Pregnancies": [1, 2, 3],
            "Glucose": [100, 120, 110],
            "BloodPressure": [70, 80, 75],
        })
        test_data.to_csv(csv_file, index=False)

        loaded_data = load_data(csv_file)
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == (3, 3)
        assert list(loaded_data.columns) == ["Pregnancies", "Glucose", "BloodPressure"]

    def test_load_data_file_not_found(self):
        """Test error handling when file doesn't exist."""
        non_existent_path = Path("/non/existent/path/data.csv")
        with pytest.raises(RuntimeError, match="Failed to load raw data"):
            load_data(non_existent_path)

    def test_load_data_permission_error(self, tmp_path):
        """Test error handling when file cannot be read."""
        csv_file = tmp_path / "test.csv"
        test_data = pd.DataFrame({"col1": [1, 2, 3]})
        test_data.to_csv(csv_file, index=False)
        
        # Try to load from a directory instead of a file
        with pytest.raises(RuntimeError, match="Failed to load raw data"):
            load_data(tmp_path)


class TestSaveData:
    """Test data saving functionality."""

    def test_save_data_success(self, tmp_path):
        """Test successful data saving to CSV."""
        output_file = tmp_path / "output" / "saved_data.csv"
        test_data = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        })

        result = save_data(test_data, output_file)

        assert result == output_file
        assert output_file.exists()
        loaded = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(loaded, test_data)

    def test_save_data_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        nested_path = tmp_path / "a" / "b" / "c" / "data.csv"
        test_data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        save_data(test_data, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_data_with_large_dataframe(self, tmp_path):
        """Test saving a large DataFrame."""
        output_file = tmp_path / "large.csv"
        large_df = pd.DataFrame({
            "col1": range(10000),
            "col2": range(10000, 20000),
        })

        save_data(large_df, output_file)

        assert output_file.exists()
        loaded = pd.read_csv(output_file)
        assert len(loaded) == 10000

    def test_save_data_preserves_columns(self, tmp_path):
        """Test that columns are preserved during save/load cycle."""
        output_file = tmp_path / "typed_data.csv"
        test_data = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        })

        save_data(test_data, output_file)
        loaded = pd.read_csv(output_file)

        assert list(loaded.columns) == ["int_col", "float_col", "str_col"]
        assert len(loaded) == 3


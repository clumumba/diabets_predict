"""Tests for FastAPI endpoints."""

import pytest
from pydantic import ValidationError


class TestDiabetesFeatures:
    """Test the DiabetesFeatures validation model."""

    def test_valid_features(self):
        """Test valid diabetes features."""
        from main import DiabetesFeatures
        
        features = DiabetesFeatures(
            Pregnancies=1,
            Glucose=100,
            BloodPressure=70,
            SkinThickness=20,
            Insulin=0,
            BMI=25.0,
            DiabetesPedigreeFunction=0.5,
            Age=30,
        )
        
        assert features.Pregnancies == 1
        assert features.Glucose == 100
        assert features.Age == 30

    def test_missing_required_field(self):
        """Test validation fails with missing field."""
        from main import DiabetesFeatures
        
        with pytest.raises(ValidationError):
            DiabetesFeatures(
                Pregnancies=1,
                Glucose=100,
                # Missing other required fields
            )

    def test_negative_value_rejected(self):
        """Test validation rejects negative values."""
        from main import DiabetesFeatures
        
        with pytest.raises(ValidationError):
            DiabetesFeatures(
                Pregnancies=-1,
                Glucose=100,
                BloodPressure=70,
                SkinThickness=20,
                Insulin=0,
                BMI=25.0,
                DiabetesPedigreeFunction=0.5,
                Age=30,
            )

    def test_invalid_type(self):
        """Test validation rejects invalid types."""
        from main import DiabetesFeatures
        
        with pytest.raises(ValidationError):
            DiabetesFeatures(
                Pregnancies="not_a_number",
                Glucose=100,
                BloodPressure=70,
                SkinThickness=20,
                Insulin=0,
                BMI=25.0,
                DiabetesPedigreeFunction=0.5,
                Age=30,
            )

    def test_all_zero_values(self):
        """Test validation accepts all zero values."""
        from main import DiabetesFeatures
        
        features = DiabetesFeatures(
            Pregnancies=0,
            Glucose=0,
            BloodPressure=0,
            SkinThickness=0,
            Insulin=0,
            BMI=0.0,
            DiabetesPedigreeFunction=0.0,
            Age=0,
        )
        
        assert features.Pregnancies == 0
        assert features.BMI == 0.0

    def test_large_values(self):
        """Test validation accepts large values."""
        from main import DiabetesFeatures
        
        features = DiabetesFeatures(
            Pregnancies=17,
            Glucose=199,
            BloodPressure=122,
            SkinThickness=99,
            Insulin=846,
            BMI=67.1,
            DiabetesPedigreeFunction=2.42,
            Age=81,
        )
        
        assert features.Pregnancies == 17
        assert features.Glucose == 199


class TestFeatureColumns:
    """Test FEATURE_COLUMNS constant."""

    def test_feature_columns_count(self):
        """Test that FEATURE_COLUMNS has correct count."""
        from main import FEATURE_COLUMNS
        
        assert len(FEATURE_COLUMNS) == 8

    def test_feature_columns_names(self):
        """Test that FEATURE_COLUMNS has expected names."""
        from main import FEATURE_COLUMNS
        
        expected = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        assert FEATURE_COLUMNS == expected

    def test_feature_columns_type(self):
        """Test that FEATURE_COLUMNS is a list."""
        from main import FEATURE_COLUMNS
        
        assert isinstance(FEATURE_COLUMNS, list)
        assert all(isinstance(col, str) for col in FEATURE_COLUMNS)


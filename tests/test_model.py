"""Automated unit tests for Machine Learning Dataset, Inference, and Validation."""

import pytest
import numpy as np
import pandas as pd

from ml.dataset import (
    FEATURE_NAMES,
    TARGET_NAME,
    generate_synthetic_data,
    get_performance_category,
)
from ml.predictor import ModelManager


def test_synthetic_data_generation():
    """Verify generated dataset contains correct schema, row count, and non-null values."""
    df = generate_synthetic_data(n_samples=100, random_seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    for feat in FEATURE_NAMES:
        assert feat in df.columns
    assert TARGET_NAME in df.columns
    assert df.isnull().sum().sum() == 0


def test_performance_category_thresholds():
    """Verify performance category thresholds match defined academic standards."""
    dist = get_performance_category(92.0)
    assert dist["name"] == "Distinction"

    first = get_performance_category(75.0)
    assert first["name"] == "First Class"

    second = get_performance_category(60.0)
    assert second["name"] == "Second Class"

    risk = get_performance_category(42.0)
    assert risk["name"] == "At Risk"


def test_model_manager_loading():
    """Verify ModelManager loads serialized champion model and metadata successfully."""
    manager = ModelManager()
    assert manager.is_loaded is True
    assert manager.model is not None
    assert manager.metadata is not None
    assert "champion_metrics" in manager.metadata


def test_valid_prediction():
    """Verify end-to-end inference for a standard valid student profile."""
    manager = ModelManager()
    payload = {
        "student_id": "STU-TEST-001",
        "student_name": "Test Candidate",
        "study_hours": 25.0,
        "attendance": 88.0,
        "previous_score": 78.0,
        "assignment_score": 85.0,
        "sleep_hours": 7.5,
        "tutoring_sessions": 2,
    }

    result = manager.predict(payload)
    assert result["success"] is True
    assert 0.0 <= result["predicted_score"] <= 100.0
    assert "category" in result
    assert "confidence_interval" in result
    assert result["confidence_interval"]["lower"] <= result["predicted_score"]
    assert result["confidence_interval"]["upper"] >= result["predicted_score"]
    assert len(result["factor_impacts"]) == len(FEATURE_NAMES)
    assert len(result["recommendations"]) > 0


def test_out_of_bounds_rejection():
    """Verify validation rejects inputs beyond permitted physical/academic boundaries."""
    manager = ModelManager()
    invalid_payload = {
        "study_hours": 80.0,  # Max is 45
        "attendance": 120.0,  # Max is 100
        "previous_score": 75.0,
        "assignment_score": 80.0,
        "sleep_hours": 7.0,
        "tutoring_sessions": 1,
    }

    result = manager.predict(invalid_payload)
    assert result["success"] is False
    assert len(result["errors"]) >= 2

"""Automated integration tests for Flask web routes and API endpoints."""

import json
import pytest
from app import create_app


@pytest.fixture
def client():
    """Create test client for Flask application."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    """Verify landing page renders successfully with hero headline."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Predict Performance." in response.data
    assert b"Empower Potential." in response.data


def test_dashboard_route(client):
    """Verify dashboard page loads with metrics and charts."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Academic Performance Overview" in response.data
    assert b"Total Predictions" in response.data


def test_predict_page_route(client):
    """Verify prediction studio page loads with form controls."""
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Student Examination Performance Predictor" in response.data
    assert b"study_hours" in response.data


def test_analytics_route(client):
    """Verify analytics page loads with model comparisons and benchmark charts."""
    response = client.get("/analytics")
    assert response.status_code == 200
    assert b"Academic Performance Analytics" in response.data
    assert b"Regression Algorithm Benchmark" in response.data


def test_history_route(client):
    """Verify history registry page loads with records table."""
    response = client.get("/history")
    assert response.status_code == 200
    assert b"Student Prediction Records" in response.data


def test_model_insights_route(client):
    """Verify model architecture & transparency page loads."""
    response = client.get("/model-insights")
    assert response.status_code == 200
    assert b"Model Specifications &amp; Methodology" in response.data or b"Model Specifications & Methodology" in response.data


def test_about_route(client):
    """Verify about page loads with project architecture."""
    response = client.get("/about")
    assert response.status_code == 200
    assert b"Project Architecture &amp; Methodology" in response.data or b"Project Architecture & Methodology" in response.data


def test_api_predict_success(client):
    """Verify AJAX prediction API accepts payload and returns inference results."""
    payload = {
        "student_id": "STU-API-001",
        "student_name": "API Candidate",
        "study_hours": 28.0,
        "attendance": 90.0,
        "previous_score": 82.0,
        "assignment_score": 88.0,
        "sleep_hours": 8.0,
        "tutoring_sessions": 2,
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "predicted_score" in data
    assert "category" in data
    assert "confidence_interval" in data
    assert "factor_impacts" in data
    assert "recommendations" in data
    assert "record_id" in data


def test_api_predict_validation_failure(client):
    """Verify AJAX prediction API rejects out-of-range payload with HTTP 400."""
    payload = {
        "student_id": "STU-ERR-001",
        "study_hours": 999.0,  # Invalid
        "attendance": -5.0,    # Invalid
        "previous_score": 50.0,
        "assignment_score": 50.0,
        "sleep_hours": 7.0,
        "tutoring_sessions": 0,
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert len(data["errors"]) > 0


def test_export_csv_endpoint(client):
    """Verify CSV export endpoint serves attachment with text/csv header."""
    response = client.get("/export/csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["Content-Type"]
    assert b"Student ID" in response.data
    assert b"Predicted Score" in response.data

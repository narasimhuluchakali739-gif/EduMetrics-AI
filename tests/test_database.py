"""Automated unit tests for SQLite database persistence, search, aggregation, and export."""

import os
import pytest
import database


def test_init_db():
    """Verify database initialization creates tables and seeds sample demo data."""
    database.init_db(seed_demo=True)
    stats = database.get_dashboard_metrics()
    assert stats["total_count"] >= 6
    assert stats["avg_score"] > 0


def test_save_and_retrieve_prediction():
    """Verify saving a new prediction record and retrieving it by primary key."""
    inputs = {
        "study_hours": 30.0,
        "attendance": 92.0,
        "previous_score": 85.0,
        "assignment_score": 90.0,
        "sleep_hours": 8.0,
        "tutoring_sessions": 2,
    }
    result = {
        "predicted_score": 88.5,
        "category": {"name": "Distinction"},
        "confidence_interval": {"lower": 85.4, "upper": 91.6},
        "factor_impacts": [],
        "recommendations": [{"type": "positive", "badge": "Excellence", "title": "Good Job", "text": "Keep up"}],
    }

    record_id = database.save_prediction(
        student_id="STU-UNIT-999",
        student_name="Unit Tester",
        inputs=inputs,
        result=result,
    )
    assert record_id > 0

    record = database.get_prediction_by_id(record_id)
    assert record is not None
    assert record["student_id"] == "STU-UNIT-999"
    assert record["student_name"] == "Unit Tester"
    assert record["predicted_score"] == 88.5
    assert record["performance_category"] == "Distinction"
    assert len(record["recommendations"]) == 1


def test_search_and_filter_predictions():
    """Verify search by student ID and category filtering."""
    rows, count = database.get_predictions(search="STU-UNIT-999")
    assert count >= 1
    assert rows[0]["student_id"] == "STU-UNIT-999"

    dist_rows, dist_count = database.get_predictions(category="Distinction")
    assert dist_count >= 1
    for r in dist_rows:
        assert r["performance_category"] == "Distinction"


def test_delete_prediction():
    """Verify deleting a record removes it from the database."""
    inputs = {
        "study_hours": 15.0,
        "attendance": 70.0,
        "previous_score": 55.0,
        "assignment_score": 60.0,
        "sleep_hours": 6.0,
        "tutoring_sessions": 0,
    }
    result = {
        "predicted_score": 58.0,
        "category": {"name": "Second Class"},
        "confidence_interval": {"lower": 54.9, "upper": 61.1},
        "factor_impacts": [],
        "recommendations": [],
    }
    record_id = database.save_prediction("STU-DEL-001", "To Delete", inputs, result)
    assert database.get_prediction_by_id(record_id) is not None

    deleted = database.delete_prediction(record_id)
    assert deleted is True
    assert database.get_prediction_by_id(record_id) is None


def test_export_csv():
    """Verify CSV export contains header columns and valid row formatting."""
    csv_str = database.export_predictions_csv()
    assert "Record ID" in csv_str
    assert "Student ID" in csv_str
    assert "Predicted Score" in csv_str
    lines = csv_str.strip().split("\n")
    assert len(lines) >= 2  # Header + at least 1 record

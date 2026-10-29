"""Database persistence module using SQLite.

Manages prediction history, search filtering, pagination, statistics aggregation,
CSV export, and sample demo records for executive dashboards.
"""

import csv
import io
import json
import os
import sqlite3
from datetime import datetime
from typing import Optional


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "predictions.db")


def get_db_connection() -> sqlite3.Connection:
    """Create and return a configured SQLite database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(seed_demo: bool = True):
    """Initialize SQLite database tables and seed with demonstration records if empty."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            study_hours REAL NOT NULL,
            attendance REAL NOT NULL,
            previous_score REAL NOT NULL,
            assignment_score REAL NOT NULL,
            sleep_hours REAL NOT NULL,
            tutoring_sessions INTEGER NOT NULL,
            predicted_score REAL NOT NULL,
            performance_category TEXT NOT NULL,
            confidence_lower REAL NOT NULL,
            confidence_upper REAL NOT NULL,
            factor_impacts_json TEXT,
            recommendations_json TEXT,
            is_demo INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Check if empty, and seed sample student records if desired
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]

    if count == 0 and seed_demo:
        print("Seeding demonstration prediction records into database...")
        demo_students = [
            {
                "student_id": "STU-2026-101",
                "student_name": "Aarav Sharma",
                "study_hours": 32.5,
                "attendance": 94.0,
                "previous_score": 88.0,
                "assignment_score": 96.0,
                "sleep_hours": 7.5,
                "tutoring_sessions": 2,
                "predicted_score": 91.8,
                "performance_category": "Distinction",
                "confidence_lower": 86.8,
                "confidence_upper": 96.8,
                "is_demo": 1,
            },
            {
                "student_id": "STU-2026-102",
                "student_name": "Priya Nair",
                "study_hours": 24.0,
                "attendance": 85.0,
                "previous_score": 76.5,
                "assignment_score": 82.0,
                "sleep_hours": 7.0,
                "tutoring_sessions": 1,
                "predicted_score": 78.4,
                "performance_category": "First Class",
                "confidence_lower": 73.4,
                "confidence_upper": 83.4,
                "is_demo": 1,
            },
            {
                "student_id": "STU-2026-103",
                "student_name": "Rohan Patel",
                "study_hours": 14.0,
                "attendance": 68.0,
                "previous_score": 52.0,
                "assignment_score": 62.0,
                "sleep_hours": 5.5,
                "tutoring_sessions": 0,
                "predicted_score": 54.2,
                "performance_category": "Second Class",
                "confidence_lower": 49.2,
                "confidence_upper": 59.2,
                "is_demo": 1,
            },
            {
                "student_id": "STU-2026-104",
                "student_name": "Ananya Roy",
                "study_hours": 9.5,
                "attendance": 58.0,
                "previous_score": 44.0,
                "assignment_score": 50.0,
                "sleep_hours": 5.0,
                "tutoring_sessions": 0,
                "predicted_score": 42.6,
                "performance_category": "At Risk",
                "confidence_lower": 37.6,
                "confidence_upper": 47.6,
                "is_demo": 1,
            },
            {
                "student_id": "STU-2026-105",
                "student_name": "Vikram Singh",
                "study_hours": 28.0,
                "attendance": 90.0,
                "previous_score": 82.0,
                "assignment_score": 88.0,
                "sleep_hours": 8.0,
                "tutoring_sessions": 3,
                "predicted_score": 84.7,
                "performance_category": "First Class",
                "confidence_lower": 79.7,
                "confidence_upper": 89.7,
                "is_demo": 1,
            },
            {
                "student_id": "STU-2026-106",
                "student_name": "Sneha Mukherjee",
                "study_hours": 36.0,
                "attendance": 96.0,
                "previous_score": 92.5,
                "assignment_score": 98.0,
                "sleep_hours": 7.5,
                "tutoring_sessions": 4,
                "predicted_score": 95.2,
                "performance_category": "Distinction",
                "confidence_lower": 90.2,
                "confidence_upper": 100.0,
                "is_demo": 1,
            },
        ]

        for s in demo_students:
            cursor.execute("""
                INSERT INTO predictions (
                    student_id, student_name, study_hours, attendance, previous_score,
                    assignment_score, sleep_hours, tutoring_sessions, predicted_score,
                    performance_category, confidence_lower, confidence_upper,
                    factor_impacts_json, recommendations_json, is_demo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                s["student_id"],
                s["student_name"],
                s["study_hours"],
                s["attendance"],
                s["previous_score"],
                s["assignment_score"],
                s["sleep_hours"],
                s["tutoring_sessions"],
                s["predicted_score"],
                s["performance_category"],
                s["confidence_lower"],
                s["confidence_upper"],
                json.dumps([]),
                json.dumps([]),
                s["is_demo"],
            ))

    conn.commit()
    conn.close()


def save_prediction(student_id: str, student_name: str, inputs: dict, result: dict) -> int:
    """Save an inference prediction record into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            student_id, student_name, study_hours, attendance, previous_score,
            assignment_score, sleep_hours, tutoring_sessions, predicted_score,
            performance_category, confidence_lower, confidence_upper,
            factor_impacts_json, recommendations_json, is_demo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    """, (
        student_id.strip(),
        (student_name or "Anonymous Student").strip(),
        float(inputs["study_hours"]),
        float(inputs["attendance"]),
        float(inputs["previous_score"]),
        float(inputs["assignment_score"]),
        float(inputs["sleep_hours"]),
        int(inputs["tutoring_sessions"]),
        float(result["predicted_score"]),
        result["category"]["name"],
        float(result["confidence_interval"]["lower"]),
        float(result["confidence_interval"]["upper"]),
        json.dumps(result.get("factor_impacts", [])),
        json.dumps(result.get("recommendations", [])),
    ))

    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_predictions(
    search: Optional[str] = None,
    category: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list, int]:
    """Retrieve filtered, sorted, and paginated prediction records."""
    conn = get_db_connection()
    cursor = conn.cursor()

    valid_sorts = {
        "created_at": "created_at",
        "student_id": "student_id",
        "student_name": "student_name",
        "predicted_score": "predicted_score",
        "attendance": "attendance",
        "study_hours": "study_hours",
    }
    sort_col = valid_sorts.get(sort_by, "created_at")
    sort_dir = "ASC" if sort_order.lower() == "asc" else "DESC"

    where_clauses = []
    params = []

    if search:
        where_clauses.append("(student_id LIKE ? OR student_name LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    if category and category != "all":
        where_clauses.append("performance_category = ?")
        params.append(category)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Count query
    count_query = f"SELECT COUNT(*) FROM predictions {where_sql}"
    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0]

    # Select query
    data_query = f"""
        SELECT * FROM predictions
        {where_sql}
        ORDER BY {sort_col} {sort_dir}
        LIMIT ? OFFSET ?
    """
    cursor.execute(data_query, params + [limit, offset])
    rows = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return rows, total_count


def get_prediction_by_id(record_id: int) -> Optional[dict]:
    """Fetch a single prediction record by primary key."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    data = dict(row)
    try:
        data["factor_impacts"] = json.loads(data.get("factor_impacts_json") or "[]")
    except Exception:
        data["factor_impacts"] = []
    try:
        data["recommendations"] = json.loads(data.get("recommendations_json") or "[]")
    except Exception:
        data["recommendations"] = []
    return data


def delete_prediction(record_id: int) -> bool:
    """Delete a prediction record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def get_dashboard_metrics() -> dict:
    """Calculate aggregated metrics and distributions for the dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*), AVG(predicted_score), MIN(predicted_score), MAX(predicted_score) FROM predictions")
    count, avg_score, min_score, max_score = cursor.fetchone()

    if count == 0:
        conn.close()
        return {
            "total_count": 0,
            "avg_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "categories": {"Distinction": 0, "First Class": 0, "Second Class": 0, "At Risk": 0},
            "recent_predictions": [],
        }

    # Category breakdown
    cursor.execute("""
        SELECT performance_category, COUNT(*) as cat_count
        FROM predictions
        GROUP BY performance_category
    """)
    cat_counts = {"Distinction": 0, "First Class": 0, "Second Class": 0, "At Risk": 0}
    for row in cursor.fetchall():
        if row["performance_category"] in cat_counts:
            cat_counts[row["performance_category"]] = row["cat_count"]

    # Recent 5 records
    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5")
    recent = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {
        "total_count": count,
        "avg_score": round(avg_score, 1) if avg_score else 0.0,
        "min_score": round(min_score, 1) if min_score else 0.0,
        "max_score": round(max_score, 1) if max_score else 0.0,
        "categories": cat_counts,
        "recent_predictions": recent,
    }


def export_predictions_csv() -> str:
    """Generate a CSV string containing all prediction history records."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, student_id, student_name, study_hours, attendance, previous_score,
               assignment_score, sleep_hours, tutoring_sessions, predicted_score,
               performance_category, confidence_lower, confidence_upper, created_at
        FROM predictions
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Record ID",
        "Student ID",
        "Student Name",
        "Study Hours/Wk",
        "Attendance %",
        "Previous Marks",
        "Assignment Marks",
        "Sleep Hours/Day",
        "Tutoring Sessions",
        "Predicted Score",
        "Performance Category",
        "CI Lower",
        "CI Upper",
        "Predicted Date",
    ])

    for r in rows:
        writer.writerow([
            r["id"],
            r["student_id"],
            r["student_name"],
            r["study_hours"],
            r["attendance"],
            r["previous_score"],
            r["assignment_score"],
            r["sleep_hours"],
            r["tutoring_sessions"],
            r["predicted_score"],
            r["performance_category"],
            r["confidence_lower"],
            r["confidence_upper"],
            r["created_at"],
        ])

    return output.getvalue()

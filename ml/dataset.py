"""Synthetic Student Performance Dataset Generator.

Generates a realistic, statistically grounded academic dataset reflecting
real-world educational dynamics (attendance thresholds, diminishing study returns,
sleep fatigue, assignment correlation, and previous academic foundation).
"""

import os
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "sleep_hours",
    "tutoring_sessions",
]

TARGET_NAME = "final_score"

FEATURE_DEFINITIONS = {
    "study_hours": {
        "label": "Study Hours per Week",
        "description": "Weekly dedicated academic study and revision time (hours).",
        "min": 5.0,
        "max": 45.0,
        "step": 0.5,
        "unit": "hrs/wk",
        "default": 20.0,
    },
    "attendance": {
        "label": "Attendance Rate",
        "description": "Percentage of lecture and lab classes attended.",
        "min": 50.0,
        "max": 100.0,
        "step": 1.0,
        "unit": "%",
        "default": 85.0,
    },
    "previous_score": {
        "label": "Previous Exam Score",
        "description": "Score achieved in the previous semester or mid-term examination.",
        "min": 35.0,
        "max": 100.0,
        "step": 0.5,
        "unit": "marks",
        "default": 72.0,
    },
    "assignment_score": {
        "label": "Assignment Completion Score",
        "description": "Average marks across continuous assessments and lab assignments.",
        "min": 40.0,
        "max": 100.0,
        "step": 0.5,
        "unit": "marks",
        "default": 78.0,
    },
    "sleep_hours": {
        "label": "Average Sleep Duration",
        "description": "Average daily rest and sleep time (hours).",
        "min": 4.0,
        "max": 10.0,
        "step": 0.5,
        "unit": "hrs/day",
        "default": 7.5,
    },
    "tutoring_sessions": {
        "label": "Tutoring / Mentoring Sessions",
        "description": "Number of academic mentoring or peer-tutoring sessions attended per month.",
        "min": 0,
        "max": 8,
        "step": 1,
        "unit": "sessions/mo",
        "default": 2,
    },
}

CATEGORIES = [
    {"name": "Distinction", "min": 85.0, "max": 100.0, "color": "emerald", "badge": "success"},
    {"name": "First Class", "min": 70.0, "max": 84.99, "color": "indigo", "badge": "primary"},
    {"name": "Second Class", "min": 50.0, "max": 69.99, "color": "amber", "badge": "warning"},
    {"name": "At Risk", "min": 0.0, "max": 49.99, "color": "rose", "badge": "danger"},
]


def generate_synthetic_data(n_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """Generate realistic academic performance samples with non-linear correlations."""
    rng = np.random.default_rng(random_seed)

    # 1. Previous score: Beta distribution shifted to realistic [35, 100]
    raw_prev = rng.beta(a=4.5, b=2.8, size=n_samples)
    previous_score = 35.0 + raw_prev * 65.0

    # 2. Study hours: Correlated with prior motivation/ability plus variance
    study_hours_base = 8.0 + (previous_score - 35.0) * 0.32 + rng.normal(0, 4.0, n_samples)
    study_hours = np.clip(study_hours_base, 5.0, 45.0)

    # 3. Attendance: Positively correlated with previous score and study habits
    attendance_base = 55.0 + (previous_score - 35.0) * 0.42 + (study_hours - 5.0) * 0.3 + rng.normal(0, 5.5, n_samples)
    attendance = np.clip(attendance_base, 50.0, 100.0)

    # 4. Assignment score: Heavily driven by attendance and study hours
    assignment_base = 35.0 + 0.38 * previous_score + 0.4 * attendance + 0.35 * study_hours + rng.normal(0, 4.0, n_samples)
    assignment_score = np.clip(assignment_base, 40.0, 100.0)

    # 5. Sleep hours: Slightly negatively correlated with extreme study hours
    sleep_base = 7.8 - (study_hours - 20.0) * 0.04 + rng.normal(0, 0.9, n_samples)
    sleep_hours = np.clip(sleep_base, 4.0, 10.0)

    # 6. Tutoring sessions: Students seeking help (poorer previous scores or high strivers)
    tutoring_prob = 1.0 / (1.0 + np.exp(0.04 * (previous_score - 65.0)))
    tutoring_sessions = rng.poisson(lam=1.5 + tutoring_prob * 2.5, size=n_samples)
    tutoring_sessions = np.clip(tutoring_sessions, 0, 8)

    # Calculate final score with educational dynamics:
    # 1) Diminishing returns on study hours (log scale beyond 25 hours)
    study_effect = 16.0 * np.log1p((study_hours - 5.0) / 7.0)

    # 2) Attendance effect with severe penalty below 75%
    attendance_norm = (attendance - 50.0) / 50.0
    attendance_penalty = np.where(attendance < 75.0, (75.0 - attendance) * 0.35, 0.0)
    attendance_effect = 18.0 * attendance_norm - attendance_penalty

    # 3) Previous score baseline anchor (30% weight)
    prev_effect = 0.30 * previous_score

    # 4) Assignment mastery (18% weight)
    assign_effect = 0.18 * assignment_score

    # 5) Sleep effect: Peak cognitive performance at 7.0 - 8.5 hours
    sleep_penalty = np.where(sleep_hours < 6.0, (6.0 - sleep_hours) * 2.2, 0.0)
    sleep_optimal = np.where((sleep_hours >= 7.0) & (sleep_hours <= 8.5), 2.5, 0.0)
    sleep_effect = sleep_optimal - sleep_penalty

    # 6) Tutoring benefit: +1.2 pts per session up to 5 sessions
    tutoring_effect = np.minimum(tutoring_sessions, 5) * 1.3

    # Baseline intercept and noise
    noise = rng.normal(0, 2.8, n_samples)
    raw_final = (
        12.0
        + study_effect
        + attendance_effect
        + prev_effect
        + assign_effect
        + sleep_effect
        + tutoring_effect
        + noise
    )

    final_score = np.clip(raw_final, 0.0, 100.0)

    df = pd.DataFrame({
        "study_hours": np.round(study_hours, 1),
        "attendance": np.round(attendance, 1),
        "previous_score": np.round(previous_score, 1),
        "assignment_score": np.round(assignment_score, 1),
        "sleep_hours": np.round(sleep_hours, 1),
        "tutoring_sessions": tutoring_sessions.astype(int),
        "final_score": np.round(final_score, 1),
    })

    return df


def save_dataset(filepath: str, n_samples: int = 2500) -> pd.DataFrame:
    """Generate and save the academic dataset to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = generate_synthetic_data(n_samples=n_samples)
    df.to_csv(filepath, index=False)
    return df


def get_performance_category(score: float) -> dict:
    """Return the academic performance category classification for a score."""
    for cat in CATEGORIES:
        if cat["min"] <= score <= cat["max"]:
            return cat
    if score >= 100.0:
        return CATEGORIES[0]
    return CATEGORIES[-1]

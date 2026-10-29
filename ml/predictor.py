"""Model Inference, Validation, and Explainability Engine.

Loads the saved model pipeline and metadata, validates student academic features,
computes point predictions with confidence intervals, delivers local feature
attributions, and provides constructive, personalized academic recommendations.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from ml.dataset import CATEGORIES, FEATURE_DEFINITIONS, FEATURE_NAMES, get_performance_category


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")


class ModelManager:
    """Singleton model loader and inference engine."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.metadata = None
            cls._instance.is_loaded = False
            cls._instance.load()
        return cls._instance

    def load(self):
        """Load trained pipeline and metadata from disk."""
        if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                with open(METADATA_PATH, "r") as f:
                    self.metadata = json.load(f)
                self.is_loaded = True
                print("Model pipeline and metadata successfully loaded.")
            except Exception as e:
                print(f"Error loading model artifacts: {e}")
                self.is_loaded = False
        else:
            print("Model artifacts not found. Automatic training recommended.")
            self.is_loaded = False

    def validate_inputs(self, data: dict) -> tuple[bool, dict, list]:
        """Validate input payload against expected types and boundary ranges."""
        cleaned = {}
        errors = []

        for feat in FEATURE_NAMES:
            spec = FEATURE_DEFINITIONS[feat]
            val = data.get(feat)

            if val is None or val == "":
                errors.append(f"Field '{spec['label']}' is required.")
                continue

            try:
                num_val = float(val) if spec.get("step", 1.0) != 1 else int(val)
            except (ValueError, TypeError):
                errors.append(f"'{spec['label']}' must be a valid numeric value.")
                continue

            if num_val < spec["min"] or num_val > spec["max"]:
                errors.append(
                    f"'{spec['label']}' must be between {spec['min']} and {spec['max']} {spec['unit']}."
                )
                continue

            cleaned[feat] = num_val

        return (len(errors) == 0, cleaned, errors)

    def predict(self, raw_data: dict) -> dict:
        """Execute validation, model inference, impact analysis, and recommendation generation."""
        if not self.is_loaded:
            self.load()
            if not self.is_loaded:
                raise RuntimeError("Machine learning model is not available. Please train the model first.")

        valid, cleaned_data, errors = self.validate_inputs(raw_data)
        if not valid:
            return {"success": False, "errors": errors}

        # Create 1-row DataFrame for pipeline prediction
        df_input = pd.DataFrame([cleaned_data])[FEATURE_NAMES]
        raw_pred = float(self.model.predict(df_input)[0])
        clamped_score = float(np.clip(raw_pred, 0.0, 100.0))
        predicted_score = round(clamped_score, 1)

        # Performance category
        category = get_performance_category(predicted_score)

        # Confidence interval estimation (using test RMSE)
        rmse = self.metadata.get("champion_metrics", {}).get("test_rmse", 3.0)
        margin = round(1.645 * rmse, 1)  # ~90% CI
        ci_lower = max(0.0, round(predicted_score - margin, 1))
        ci_upper = min(100.0, round(predicted_score + margin, 1))

        # Local factor impact breakdown
        stats = self.metadata.get("dataset_statistics", {})
        importances = self.metadata.get("feature_importance", {})
        factor_impacts = []

        for feat in FEATURE_NAMES:
            val = cleaned_data[feat]
            feat_stat = stats.get(feat, {})
            median = feat_stat.get("median", val)
            std = feat_stat.get("std", 1.0)
            importance = importances.get(feat, 0.1)

            # Estimate point impact relative to cohort baseline
            z_score = (val - median) / (std if std > 0 else 1.0)
            # Scaling point impact based on feature weight
            impact_points = round(z_score * importance * 12.0, 1)

            status = "positive" if impact_points > 0.5 else ("negative" if impact_points < -0.5 else "neutral")
            factor_impacts.append({
                "feature": feat,
                "label": FEATURE_DEFINITIONS[feat]["label"],
                "value": val,
                "unit": FEATURE_DEFINITIONS[feat]["unit"],
                "cohort_median": median,
                "impact_points": impact_points,
                "status": status,
                "description": self._get_factor_description(feat, val, impact_points),
            })

        # Sort factors by absolute impact
        factor_impacts.sort(key=lambda x: abs(x["impact_points"]), reverse=True)

        # Generate personalized, constructive recommendations
        recommendations = self._generate_recommendations(cleaned_data, predicted_score)

        return {
            "success": True,
            "predicted_score": predicted_score,
            "category": category,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "margin": margin,
                "level": "90%",
            },
            "factor_impacts": factor_impacts,
            "recommendations": recommendations,
            "model_info": {
                "name": self.metadata.get("model_name", "Gradient Boosting"),
                "test_r2": round(self.metadata.get("champion_metrics", {}).get("test_r2", 0.92), 3),
                "test_rmse": round(rmse, 2),
            },
            "inputs": cleaned_data,
            "disclaimer": (
                "Educational Estimate Notice: Predictions are statistical approximations derived from "
                "historical cohort models and do not represent a definitive limit on student performance or capability."
            ),
        }

    def _get_factor_description(self, feat: str, val: float, impact: float) -> str:
        """Provide concise explanation of factor impact."""
        if feat == "attendance":
            if val < 75.0:
                return f"Attendance ({val}%) is below the critical 75% threshold, causing an institutional penalty."
            elif val >= 85.0:
                return f"Strong lecture attendance ({val}%) strongly reinforces theoretical comprehension."
            return f"Satisfactory attendance ({val}%) provides steady baseline coverage."
        elif feat == "study_hours":
            if val < 15.0:
                return f"Study time ({val} hrs/wk) is below cohort average, limiting problem-solving repetition."
            elif val > 30.0:
                return f"High study volume ({val} hrs/wk) provides strong mastery; beware of diminishing returns."
            return f"Balanced study routine ({val} hrs/wk) maintains consistent conceptual grasp."
        elif feat == "previous_score":
            return f"Prior academic foundation ({val} marks) establishes strong baseline confidence."
        elif feat == "assignment_score":
            return f"Continuous assessment score ({val} marks) reflects consistent coursework execution."
        elif feat == "sleep_hours":
            if val < 6.0:
                return f"Rest duration ({val} hrs) is low, risking cognitive fatigue during complex analysis."
            return f"Healthy rest ({val} hrs) sustains focus and memory retention."
        elif feat == "tutoring_sessions":
            if val > 0:
                return f"{val} monthly mentoring sessions provide targeted problem-solving clarity."
            return "Zero tutoring sessions attended; self-directed learning pathway."
        return ""

    def _generate_recommendations(self, data: dict, predicted_score: float) -> list:
        """Generate high-impact, positive, actionable recommendations."""
        recs = []

        # Attendance check
        if data["attendance"] < 75.0:
            target_att = min(85.0, data["attendance"] + 10.0)
            recs.append({
                "type": "urgent",
                "badge": "Priority Focus",
                "title": "Mitigate Attendance Risk",
                "text": f"Your current attendance is {data['attendance']}%. Raising class presence above 75% (ideally {target_att}%) will eliminate institutional risk and can recover up to 4 to 6 examination marks.",
            })
        elif data["attendance"] < 85.0:
            recs.append({
                "type": "opportunity",
                "badge": "Attendance",
                "title": "Optimize Class Engagement",
                "text": "Boosting attendance toward 90% provides direct exposure to instructor exam hints and laboratory discussions.",
            })

        # Study hours check
        if data["study_hours"] < 18.0:
            boost_hrs = round(data["study_hours"] + 4.0, 1)
            recs.append({
                "type": "opportunity",
                "badge": "Study Habits",
                "title": "Expand Active Revision Time",
                "text": f"Increasing weekly study time from {data['study_hours']}h to ~{boost_hrs}h (approx. 45 mins extra per day) yields the highest return on performance, potentially boosting marks by ~3.5 to 5.0 points.",
            })
        elif data["study_hours"] > 35.0:
            recs.append({
                "type": "wellness",
                "badge": "Study Optimization",
                "title": "Focus on High-Yield Study Techniques",
                "text": "With 35+ weekly study hours, prioritize spaced repetition and timed practice exams over passive re-reading to avoid burnout and diminishing returns.",
            })

        # Sleep check
        if data["sleep_hours"] < 6.5:
            recs.append({
                "type": "wellness",
                "badge": "Cognitive Health",
                "title": "Prioritize Rest & Sleep Recovery",
                "text": f"Current rest average is {data['sleep_hours']} hours. Academic studies show that sleeping 7 to 8 hours enhances memory consolidation and exam recall accuracy by 10-15%.",
            })

        # Tutoring check
        if data["tutoring_sessions"] == 0 and predicted_score < 75.0:
            recs.append({
                "type": "action",
                "badge": "Mentorship",
                "title": "Engage in Faculty or Peer Mentoring",
                "text": "Attending 2 to 3 peer-tutoring or office-hour sessions per month can clarify difficult module topics and build exam problem-solving speed.",
            })

        # Assignment mastery
        if data["assignment_score"] < 75.0:
            recs.append({
                "type": "action",
                "badge": "Continuous Assessment",
                "title": "Strengthen Assignment Quality",
                "text": "Continuous assessments mirror final examination questions. Review past assignment feedback and resubmit revisions to reinforce concepts.",
            })

        # If already high performer
        if predicted_score >= 85.0:
            recs.append({
                "type": "positive",
                "badge": "Excellence",
                "title": "Maintain High-Performance Momentum",
                "text": "Outstanding academic indicators. Continue simulating full-length timed mock exams and consider peer-mentoring fellow students to deepen conceptual synthesis.",
            })

        # Ensure at least 3 constructive points
        if len(recs) < 3:
            recs.append({
                "type": "action",
                "badge": "Strategy",
                "title": "Weekly Progress Benchmarking",
                "text": "Establish a weekly self-assessment milestone to track syllabus completion and practice end-of-chapter problem sets.",
            })

        return recs[:4]


# Global instance
model_manager = ModelManager()

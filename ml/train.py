"""Machine Learning Training and Benchmark Pipeline.

Trains, cross-validates, and evaluates candidate regression models for
student academic performance estimation. Compares Linear, Ridge, Random Forest,
and Gradient Boosting models, saving the champion model pipeline and evaluation charts.
"""

import json
import os
from datetime import datetime
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.dataset import (
    CATEGORIES,
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    TARGET_NAME,
    generate_synthetic_data,
    save_dataset,
)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "student_model.joblib")
METADATA_SAVE_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def load_or_create_data() -> pd.DataFrame:
    """Load existing dataset or generate new synthetic dataset if absent."""
    if os.path.exists(DATA_PATH):
        print(f"Loading dataset from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"Dataset not found at {DATA_PATH}. Generating synthetic cohort...")
        df = save_dataset(DATA_PATH, n_samples=2500)
        print(f"Generated {len(df)} records.")
    return df


def train_and_evaluate_models():
    """Execute training pipeline, compare algorithms, generate plots and save champion."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_or_create_data()

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    candidate_models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]),
        "Random Forest": Pipeline([
            ("regressor", RandomForestRegressor(
                n_estimators=120, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("regressor", GradientBoostingRegressor(
                n_estimators=140, learning_rate=0.07, max_depth=4, random_state=42
            )),
        ]),
    }

    results = {}
    fitted_pipelines = {}

    print("\n--- Model Benchmark & Cross-Validation ---")
    for name, pipeline in candidate_models.items():
        # 5-fold cross validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
        
        # Fit on full training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on holdout test set
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std": float(np.std(cv_scores)),
            "test_r2": float(r2),
            "test_mae": float(mae),
            "test_rmse": float(rmse),
        }

        print(f"[{name}] Test R²: {r2:.4f} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | 5-Fold CV R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

    # Select champion based on highest test R2
    champion_name = max(results.keys(), key=lambda k: results[k]["test_r2"])
    champion_pipeline = fitted_pipelines[champion_name]
    print(f"\nChampion Model Selected: {champion_name} (R² = {results[champion_name]['test_r2']:.4f})")

    # Save champion model
    joblib.dump(champion_pipeline, MODEL_SAVE_PATH)
    print(f"Saved champion pipeline to: {MODEL_SAVE_PATH}")

    # Calculate feature importances
    regressor = champion_pipeline.named_steps["regressor"]
    feature_importance_dict = {}
    if hasattr(regressor, "feature_importances_"):
        raw_importances = regressor.feature_importances_
        for feat, imp in zip(FEATURE_NAMES, raw_importances):
            feature_importance_dict[feat] = float(imp)
    elif hasattr(regressor, "coef_"):
        raw_coefs = np.abs(regressor.coef_)
        total = np.sum(raw_coefs)
        for feat, coef in zip(FEATURE_NAMES, raw_coefs / total):
            feature_importance_dict[feat] = float(coef)

    # Sort feature importances descending
    feature_importance_dict = dict(
        sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
    )

    # Calculate dataset statistics for contextual interpretation
    stats = {}
    for col in FEATURE_NAMES + [TARGET_NAME]:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "q25": float(df[col].quantile(0.25)),
            "q75": float(df[col].quantile(0.75)),
        }

    # Generate metadata dictionary
    metadata = {
        "model_name": champion_name,
        "champion_algorithm": str(type(regressor).__name__),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": FEATURE_NAMES,
        "feature_definitions": FEATURE_DEFINITIONS,
        "target": TARGET_NAME,
        "categories": CATEGORIES,
        "benchmark_comparison": results,
        "champion_metrics": results[champion_name],
        "feature_importance": feature_importance_dict,
        "dataset_statistics": stats,
    }

    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {METADATA_SAVE_PATH}")

    # Generate evaluation visualizations
    generate_evaluation_charts(results, champion_pipeline, X_test, y_test, feature_importance_dict)

    return metadata


def generate_evaluation_charts(results, champion_pipeline, X_test, y_test, feature_importance_dict):
    """Generate high-resolution evaluation charts for analytics and reports."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    primary_color = "#4f46e5"
    secondary_color = "#3b82f6"
    accent_color = "#10b981"
    text_color = "#1e293b"

    # 1. Model Comparison Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    names = list(results.keys())
    r2_scores = [results[n]["test_r2"] for n in names]
    cv_means = [results[n]["cv_r2_mean"] for n in names]
    cv_stds = [results[n]["cv_r2_std"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    rects1 = ax.bar(x - width/2, r2_scores, width, label="Test R²", color="#4f46e5", alpha=0.9, edgecolor="none")
    rects2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds, capsize=4, label="5-Fold CV R²", color="#38bdf8", alpha=0.9, edgecolor="none")

    ax.set_ylabel("R² Score (Higher is better)", fontsize=11, fontweight="bold", color=text_color)
    ax.set_title("Machine Learning Algorithm Comparison", fontsize=13, fontweight="bold", pad=15, color=text_color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight="bold", color=text_color)
    ax.set_ylim(0.70, 1.0)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#4f46e5")

    plt.tight_layout()
    comp_path = os.path.join(REPORTS_DIR, "model_comparison.png")
    fig.savefig(comp_path)
    plt.close(fig)
    print(f"Saved: {comp_path}")

    # 2. Feature Importance Chart
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    feats = list(feature_importance_dict.keys())[::-1]
    imps = [feature_importance_dict[f] for f in feats]
    readable_feats = [FEATURE_DEFINITIONS.get(f, {}).get("label", f) for f in feats]

    bars = ax.barh(readable_feats, imps, color="#6366f1", height=0.6, edgecolor="none")
    ax.set_xlabel("Relative Predictive Importance (Normalized)", fontsize=11, fontweight="bold", color=text_color)
    ax.set_title("Feature Contribution to Academic Performance", fontsize=13, fontweight="bold", pad=15, color=text_color)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width*100:.1f}%",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold", color="#4338ca")

    plt.tight_layout()
    feat_path = os.path.join(REPORTS_DIR, "feature_importance.png")
    fig.savefig(feat_path)
    plt.close(fig)
    print(f"Saved: {feat_path}")

    # 3. Actual vs Predicted Plot
    y_pred = champion_pipeline.predict(X_test)
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    ax.scatter(y_test, y_pred, alpha=0.55, color="#3b82f6", edgecolors="none", s=28, label="Test Students")
    
    # 45-degree reference line
    lims = [min(min(y_test), min(y_pred)) - 2, max(max(y_test), max(y_pred)) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.8, label="Ideal 1:1 Identity Line", alpha=0.85)

    ax.set_xlabel("Actual Exam Marks", fontsize=11, fontweight="bold", color=text_color)
    ax.set_ylabel("Predicted Exam Marks", fontsize=11, fontweight="bold", color=text_color)
    ax.set_title("Champion Model: Actual vs Predicted Performance", fontsize=13, fontweight="bold", pad=15, color=text_color)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    act_path = os.path.join(REPORTS_DIR, "actual_vs_predicted.png")
    fig.savefig(act_path)
    plt.close(fig)
    print(f"Saved: {act_path}")

    # 4. Residuals Distribution Plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    ax.hist(residuals, bins=30, color="#10b981", alpha=0.75, edgecolor="#ffffff")
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, label="Zero Error Line")
    ax.set_xlabel("Prediction Error / Residual (Actual - Predicted)", fontsize=11, fontweight="bold", color=text_color)
    ax.set_ylabel("Student Count", fontsize=11, fontweight="bold", color=text_color)
    ax.set_title("Champion Model Residual Distribution", fontsize=13, fontweight="bold", pad=15, color=text_color)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    res_path = os.path.join(REPORTS_DIR, "residuals_distribution.png")
    fig.savefig(res_path)
    plt.close(fig)
    print(f"Saved: {res_path}")


if __name__ == "__main__":
    train_and_evaluate_models()

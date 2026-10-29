# EduMetrics AI: Enterprise Student Performance Prediction & Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-emerald.svg)]()
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passing-brightgreen.svg)]()

> A production-grade, full-stack Machine Learning web application engineered for student performance estimation, educational cohort analytics, and pedagogical intervention advisory. Designed for university portfolios, academic presentations, and real-world deployment.

---

## 1. Project Overview & Architectural Highlights

EduMetrics AI upgrades traditional college ML projects into an industry-grade SaaS platform inspired by modern interfaces like **Linear**, **Vercel**, and **Stripe**. 

### Key Capabilities:
- **Champion Gradient Boosting ML Model:** Outperforms linear baselines with a **Holdout Test $R^2 = 0.922$**, **RMSE = &plusmn;3.06 marks**, and **5-Fold Cross-Validation $R^2 = 0.925$**.
- **Local Feature Attribution:** Explains how each student's indicators (attendance, study hours, previous scores, continuous assessment, sleep, and tutoring) impact their final score relative to cohort medians.
- **Actionable Guidance Engine:** Generates positive, non-judgmental recommendations prioritizing interventions that deliver the highest ROI on academic marks.
- **Interactive Executive Dashboard:** Features real-time KPI metrics, performance tier distributions (Distinction, First Class, Second Class, At-Risk), and recent prediction tracking.
- **Deep-Dive Cohort Analytics:** Interactive Chart.js visualizations covering regression benchmarks, feature importances, study volume curves, and error residual distributions.
- **Persistent SQLite History & CSV Export:** Search, filter by tier, sort, view individualized printable assessment reports, delete records with confirmation, or export entire datasets to CSV.
- **Accessible & Responsive Design System:** Seamless Dark/Light theme switching, synchronized range sliders, keyboard accessibility, and mobile navigation drawer.

---

## 2. Machine Learning Pipeline & Empirical Benchmarks

The model was trained, benchmarked, and evaluated on a validated 2,500-sample academic cohort incorporating realistic non-linearities (diminishing study returns, institutional attendance risk thresholds below 75%, and sleep fatigue penalties).

### Benchmark Results (5-Fold Cross Validation):

| Algorithm | Algorithm Type | Holdout Test $R^2$ | 5-Fold CV Mean $R^2$ (&plusmn;Std) | MAE (Marks) | RMSE (Marks) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | Ensemble Decision Trees | **0.9219** | **0.9248 (&plusmn;0.0020)** | **2.415** | **3.058** | **Selected Champion** |
| **Random Forest** | Bagging Ensemble Trees | 0.9056 | 0.9158 (&plusmn;0.0019) | 2.660 | 3.362 | Candidate |
| **Ridge Regression** | L2 Regularized Linear | 0.9012 | 0.9109 (&plusmn;0.0058) | 2.758 | 3.439 | Baseline |
| **Linear Regression** | Ordinary Least Squares | 0.9012 | 0.9109 (&plusmn;0.0058) | 2.758 | 3.439 | Baseline |

### Feature Importance Weights:
1. **Attendance Rate (`attendance`):** 33.4% &mdash; Lecture and lab presence; critical risk threshold below 75%.
2. **Assignment Mastery (`assignment_score`):** 27.4% &mdash; Continuous coursework performance.
3. **Previous Exam Score (`previous_score`):** 18.7% &mdash; Baseline cognitive foundation anchor.
4. **Weekly Study Volume (`study_hours`):** 17.0% &mdash; Dedicated revision time with log-linear diminishing returns.
5. **Tutoring & Mentorship (`tutoring_sessions`):** 2.3% &mdash; Targeted remediation sessions.
6. **Sleep & Cognitive Rest (`sleep_hours`):** 1.3% &mdash; Rest duration; fatigue penalty when &lt;6 hours.

---

## 3. Project Directory Structure

```
student-performance-prediction/
│
├── app.py                      # Flask application factory, routes, and REST API
├── database.py                 # SQLite persistence layer, queries, aggregation, CSV export
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation and setup guide
│
├── ml/                         # Machine learning engineering package
│   ├── dataset.py              # Synthetic cohort generator and schema definition
│   ├── train.py                # Model training, comparison, evaluation & plot generator
│   └── predictor.py            # Singleton inference engine, impact attribution & recommendations
│
├── data/                       # Datasets & persistent database
│   ├── DATASET_CARD.md         # Data card documenting pedagogical assumptions & bounds
│   ├── student_performance_data.csv # 2,500-sample benchmark dataset
│   └── predictions.db          # SQLite persistent database
│
├── models/                     # Serialized artifacts
│   ├── student_model.joblib    # Serialized champion pipeline
│   └── model_metadata.json     # Hyperparameters, evaluation metrics, and feature weights
│
├── reports/                    # High-resolution evaluation charts
│   ├── model_comparison.png    # 4-algorithm benchmark bar chart
│   ├── feature_importance.png  # Feature importance plot
│   ├── actual_vs_predicted.png # Scatter plot with 1:1 identity line
│   └── residuals_distribution.png # Error residual histogram
│
├── static/                     # Frontend static assets
│   ├── css/
│   │   └── style.css           # Premium SaaS design system (tokens, themes, cards, forms)
│   └── js/
│       └── main.js             # Theme switcher, input synchronizers, modals, presets, toasts
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Global layout, sticky navbar, theme toggle, footer
│   ├── index.html              # High-conversion SaaS landing page
│   ├── dashboard.html          # Executive analytics dashboard
│   ├── predict.html            # AI inference studio with live gauge and attributions
│   ├── analytics.html          # Deep cohort charts and model benchmarks
│   ├── history.html            # Searchable, filterable history registry
│   ├── report.html             # Printable & PDF-ready assessment advisory docket
│   ├── model_insights.html     # Model transparency and ethical AI statements
│   └── about.html              # B.Tech AIML portfolio presentation and architecture
│
└── tests/                      # Automated pytest test suite (20/20 passing)
    ├── test_model.py           # ML inference, boundaries, and dataset tests
    ├── test_database.py        # SQLite CRUD and CSV export tests
    └── test_routes.py          # HTTP routes, API endpoints, and validation tests
```

---

## 4. Setup & Launch Instructions (Windows PowerShell)

Follow these exact steps from the project root directory (`student-performance-prediction`):

### Step 1: Open PowerShell in Project Directory
```powershell
cd "c:\Users\Narasimha\OneDrive\my projects\flask project\student-performance-prediction"
```

### Step 2: Activate the Virtual Environment
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Step 3: Verify Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 4: Run Automated Tests
Execute the full test suite to confirm 100% test passage:
```powershell
python -m pytest -v
```

### Step 5: (Optional) Re-train the Machine Learning Models
To re-evaluate candidate models and regenerate evaluation charts:
```powershell
python -m ml.train
```

### Step 6: Launch the Web Application
```powershell
python app.py
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 5. Navigating the Platform

- **Landing Page (`/`):** Hero section, preview card, feature breakdown, how-it-works workflow, and model accuracy cards.
- **Executive Dashboard (`/dashboard`):** Real-time prediction counts, average score KPIs, performance tier doughnut chart, and recent evaluations.
- **Prediction Studio (`/predict`):** Select a quick preset (e.g., *High Achiever*, *Balanced*, *Borderline*, or *At Risk*), adjust sliders, and compute instant predictions with animated score meters and local factor attributions.
- **Cohort Analytics (`/analytics`):** Interactive Chart.js visualizations illustrating algorithm comparisons, feature weights, attendance curves, and error distributions.
- **Records History (`/history`):** Search by student name/roll number, filter by tier, sort columns, view individualized printable reports, or delete records.
- **Printable Assessment Report (`/report/<id>`):** PDF-ready advisory report with institutional branding, candidate profile, score breakdown, and advisor sign-off.
- **Model Insights (`/model-insights`):** Technical audit of hyperparameter choices, 5-fold cross-validation metrics, and ethical AI scope.
- **About Project (`/about`):** Full-stack architecture documentation and academic capstone overview.

---

## 6. Ethical AI & Pedagogical Safeguards

1. **Non-Deterministic Guidance:** Predictions are statistical estimates with 90% confidence bands (&plusmn;3.1 marks); they do not dictate or constrain individual potential.
2. **Fairness by Design:** Sensitive demographic and socioeconomic attributes are strictly excluded to safeguard against demographic bias.
3. **Constructive Advisory:** Algorithmic outputs deliver positive, actionable intervention recommendations rather than punitive assessments.
4. **Human-in-the-Loop:** Designed as an advisory decision-support tool for academic counselors, mentors, and faculty advisors.

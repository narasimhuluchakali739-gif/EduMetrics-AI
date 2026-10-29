# Dataset Card: Academic Performance Benchmark Dataset

## 1. Overview
- **Dataset Name:** Student Performance Benchmark Dataset (Synthetic Demonstration Cohort)
- **Target Variable:** `final_score` (Continuous, 0.0 to 100.0 marks)
- **Sample Size:** 2,500 student records
- **Primary Domain:** Higher Education / Academic Analytics / Machine Learning for Educational Data Mining (EDM)

## 2. Feature Schema & Academic Grounding

| Feature | Type | Range | Description & Pedagogical Rationale |
| :--- | :--- | :--- | :--- |
| `study_hours` | Numeric (Continuous) | 5.0 - 45.0 hrs/wk | Weekly dedicated revision and active study time outside classroom lectures. Reflects diminishing returns past 35h. |
| `attendance` | Numeric (Percentage) | 50.0 - 100.0 % | Percentage of theoretical and laboratory sessions attended. Institutional guidelines impose penalties below 75%. |
| `previous_score` | Numeric (Marks) | 35.0 - 100.0 | Cumulative score from the preceding semester/term examination. Functions as cognitive baseline anchor. |
| `assignment_score` | Numeric (Marks) | 40.0 - 100.0 | Mean score across continuous assessments, problem sets, and lab evaluations. Strong indicator of consistent work habits. |
| `sleep_hours` | Numeric (Continuous) | 4.0 - 10.0 hrs/day | Average daily sleep duration. Incorporates cognitive fatigue penalty below 6 hours and peak efficiency at 7.0–8.5 hours. |
| `tutoring_sessions` | Integer | 0 - 8 sessions/mo | Number of academic mentoring, tutoring, or faculty problem-solving office hour sessions attended per month. |
| `final_score` | Numeric (Target) | 0.0 - 100.0 marks | Final summative examination score. |

## 3. Academic Performance Tiers
- **Distinction (Exemplary):** 85.0 – 100.0 marks
- **First Class (Proficient):** 70.0 – 84.9 marks
- **Second Class (Satisfactory):** 50.0 – 69.9 marks
- **At Risk (Needs Support):** 0.0 – 49.9 marks

## 4. Ethical AI & Limitations
- **Synthetic Data Disclaimer:** This cohort is synthetically generated using empirical distributions derived from published educational data mining research. It serves to demonstrate statistical modeling, machine learning pipelines, and predictive dashboards.
- **Non-Deterministic Nature:** Predictions represent point estimates with uncertainty intervals, not absolute determinations of a student's innate intellectual capacity.
- **Constructive Guidance:** Outputs must always be accompanied by positive, actionable recommendations rather than punitive assessments.

# 🏥 MedAdherence AI

An AI-powered diabetes medication adherence prediction and monitoring system. The app uses a Random Forest classifier trained on 100,000 patient records to predict patient risk tiers, surface actionable clinical insights, and provide an AI chatbot assistant — all through an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Data](#data)
- [Tech Stack](#tech-stack)

---

## Overview

MedAdherence AI helps clinicians and healthcare analysts identify diabetic patients at risk of poor medication adherence. It classifies patients into **High**, **Medium**, and **Low** risk tiers using clinical and behavioral features, then surfaces personalised recommendations and trends through a polished web interface.

---

## ✨ Features

- **Risk Prediction** — Classifies patients into High / Medium / Low risk tiers using a Random Forest model with cross-validated accuracy reporting.
- **Interactive Dashboard** — KPI cards, patient cards with risk badges, and rich Plotly charts (distributions, HbA1c trends, BMI breakdowns, feature importance, etc.).
- **AI Chatbot Assistant** — Intent-based assistant that answers natural-language questions about the dataset (adherence rates, missed doses, model performance, recommendations).
- **Data Upload** — Upload your own CSV; the app validates the schema and runs predictions on the fly.
- **PDF / Excel Export** — Download filtered patient lists and reports directly from the UI.
- **SHAP Explanations** — Feature importance visualisation backed by SHAP values.

---

## 📁 Project Structure

```
AI-Adherence__v3/
├── main.py                          # CLI entry point: load → clean → train
├── data.py                          # Data generation / augmentation utilities
├── generate_data.py                 # Script to generate synthetic patient data
├── requirements.txt                 # Pinned Python dependencies
│
├── src/
│   ├── config.py                    # DATA_PATH and global constants
│   ├── data_loader.py               # CSV ingestion
│   ├── preprocessing.py             # Cleaning, schema validation, feature engineering
│   └── model.py                     # Random Forest training, evaluation, saving
│
├── app/
│   ├── streamlit_app.py             # Main Streamlit dashboard
│   ├── streamlit_app_backup.py      # Backup / previous version
│   └── chatbot.py                   # Intent-based AI assistant
│
├── models/
│   ├── random_forest.pkl            # Trained model (joblib)
│   ├── feature_importance.pkl       # Feature importance array
│   ├── metrics.pkl                  # Saved evaluation metrics dict
│   └── target_mapping.pkl           # Label encoding map
│
└── data/
    └── diabetes_adherence_enhanced_100k.csv   # Primary training dataset (100k rows)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AI-Adherence.git
cd AI-Adherence

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The `venv/` folder is included in the repo for convenience but can be safely deleted and recreated with the steps above.

---

## 🖥️ Usage

### Train the model (CLI)

Trains the Random Forest on the dataset and saves artifacts to `models/`:

```bash
python main.py
```

### Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Upload your own data

From the sidebar in the dashboard, upload a CSV file. The app will validate the schema, run predictions, and refresh all charts automatically.

---

## 🤖 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| Estimators | 200 |
| Max depth | 12 |
| Class weighting | Balanced |
| Train / test split | 80 / 20 (stratified) |
| Cross-validation | 3-fold StratifiedKFold |
| Target variable | `Patient_Risk_Tier` (High / Medium / Low) |

Saved artifacts in `models/`:

- `random_forest.pkl` — trained model
- `feature_importance.pkl` — feature importance scores
- `metrics.pkl` — accuracy, AUC, classification report
- `target_mapping.pkl` — label-to-index mapping

---

## 📊 Data

The primary dataset (`data/diabetes_adherence_enhanced_100k.csv`) contains **100,000 synthetic patient records** with features such as:

- Demographics: Age, BMI, Gender
- Clinical: HbA1c, Fasting Glucose, Blood Pressure
- Behavioural: Missed Doses, Doctor Visit Frequency, Adherence Rate
- Outcome: `Patient_Risk_Tier`, `Health_Improvement_Score`

To regenerate or customise the dataset:

```bash
python generate_data.py
```

---

## 🛠️ Tech Stack

| Layer | Libraries |
|-------|-----------|
| ML | scikit-learn, SHAP, NumPy, pandas |
| Visualisation | Plotly, Matplotlib, Seaborn, Altair |
| Dashboard | Streamlit |
| AI Assistant | Anthropic SDK (claude-sonnet), intent-regex fallback |
| Export | fpdf2, openpyxl |
| Numerical | SciPy, statsmodels, Numba |



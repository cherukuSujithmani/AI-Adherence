import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPTIONAL_COLS = {
    "Adherence_Month1_Pct", "Adherence_Month3_Pct", "Adherence_Month6_Pct",
    "HbA1c_Followup", "Fasting_Glucose_Baseline_mg_dL", "Fasting_Glucose_Followup_mg_dL",
    "Health_Improvement_Score", "Recovery_Time_Days", "Missed_Doses_Per_Month",
    "Doctor_Visit_Frequency", "Age", "BMI", "Gender", "Diabetes_Type",
    "Medication_Adherence", "Adherence_Trend", "Socioeconomic_Status", "Education_Level",
}

def validate_schema(df):
    """Returns (ok, message, missing_optional_cols). Never blocks on optional cols."""
    if len(df) < 2:
        return False, "Dataset too small (minimum 2 rows required)", []
    if "HbA1c_Baseline" not in df.columns:
        return False, "Missing required column: HbA1c_Baseline", ["HbA1c_Baseline"]
    missing_optional = sorted(OPTIONAL_COLS - set(df.columns))
    return True, "OK", missing_optional


def clean_data(df):
    logger.info("Starting data cleaning. Shape: %s", df.shape)

    drop_cols = ["Patient_ID","Symptoms","Drug_Allergies","Genetic_Disorders","Reason_For_Non_Adherence","Enrollment_Date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df = df.ffill().bfill()
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Adherence_Avg: from monthly pct columns OR from text label
    adherence_months = [c for c in ["Adherence_Month1_Pct","Adherence_Month3_Pct","Adherence_Month6_Pct"] if c in df.columns]
    if adherence_months:
        df["Adherence_Avg"] = df[adherence_months].mean(axis=1)
    elif "Medication_Adherence" in df.columns:
        adh_map = {"High Adherent": 88, "Moderately Adherent": 63, "Low Adherent": 35}
        df["Adherence_Avg"] = df["Medication_Adherence"].map(adh_map).fillna(60)

    if "HbA1c_Baseline" in df.columns and "HbA1c_Followup" in df.columns:
        df["HbA1c_Delta"] = df["HbA1c_Baseline"] - df["HbA1c_Followup"]

    if "Fasting_Glucose_Baseline_mg_dL" in df.columns and "Fasting_Glucose_Followup_mg_dL" in df.columns:
        df["Glucose_Delta"] = df["Fasting_Glucose_Baseline_mg_dL"] - df["Fasting_Glucose_Followup_mg_dL"]

    if "Missed_Doses_Per_Month" in df.columns and "Adherence_Avg" in df.columns:
        df["Adherence_Risk_Score"] = (
            df["Missed_Doses_Per_Month"] / (df["Adherence_Avg"].replace(0, np.nan) / 100)
        ).fillna(0).round(4)

    numeric_target = [
        "Age","BMI","Missed_Doses_Per_Month","Doctor_Visit_Frequency",
        "HbA1c_Baseline","HbA1c_Followup","HbA1c_Delta",
        "Fasting_Glucose_Baseline_mg_dL","Fasting_Glucose_Followup_mg_dL","Glucose_Delta",
        "Health_Improvement_Score","Recovery_Time_Days",
        "Adherence_Avg","Adherence_Risk_Score","Patient_Risk_Tier",
    ]
    categorical_cols = ["Gender","Diabetes_Type","Medication_Adherence","Adherence_Trend","Socioeconomic_Status","Education_Level"]

    keep = [c for c in numeric_target if c in df.columns]
    keep += [c for c in categorical_cols if c in df.columns]
    df = df[[c for c in keep if c in df.columns]]

    cats_present = [c for c in categorical_cols if c in df.columns]
    if cats_present:
        df = pd.get_dummies(df, columns=cats_present, drop_first=True)

    for c in df.select_dtypes(include="number").columns:
        if c in ("Patient_Risk_Tier","Health_Improvement_Score"):
            continue
        q1, q3 = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(q1, q3)

    logger.info("Cleaning complete. Shape: %s", df.shape)
    return df
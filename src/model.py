import pandas as pd
import numpy as np
import os
import joblib
import logging

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    logger.info("Starting model training…")

    if "Patient_Risk_Tier" not in df.columns:
        raise ValueError("Target column 'Patient_Risk_Tier' not found.")

    y = df["Patient_Risk_Tier"]
    X = df.drop(columns=["Patient_Risk_Tier", "Health_Improvement_Score"], errors="ignore")
    X = X.select_dtypes(include="number")

    # Remove constant / all-NaN columns
    X = X.loc[:, X.nunique() > 1].dropna(axis=1, how="all")

    y_encoded, y_mapping = pd.factorize(y)

    logger.info("Target classes: %s", dict(enumerate(y_mapping)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Cross-val score (3-fold for speed on large data)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy", n_jobs=-1)

    logger.info("Test accuracy: %.4f", acc)
    logger.info("CV accuracy: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=y_mapping))

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)

    # ── AUC (multi-class) ─────────────────────────────────────────────────
    try:
        y_prob = model.predict_proba(X_test)
        y_test_bin = label_binarize(y_test, classes=list(range(len(y_mapping))))
        if y_test_bin.shape[1] > 1:
            auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
            logger.info("ROC-AUC (macro): %.4f", auc)
    except Exception:
        auc = None

    # ── Feature importance ────────────────────────────────────────────────
    feat_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # ── Save artefacts ────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")
    joblib.dump(y_mapping, "models/target_mapping.pkl")
    joblib.dump(feat_importance, "models/feature_importance.pkl")
    joblib.dump(
        {
            "accuracy": float(acc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "auc": float(auc) if auc else None,
            "confusion_matrix": cm.tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "classes": list(y_mapping),
        },
        "models/metrics.pkl",
    )

    logger.info("All artefacts saved to models/")
    return model
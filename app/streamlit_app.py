import os, sys, io, logging
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_data, validate_schema
from app.chatbot import build_stats, respond

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="MedAdherence AI", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.kpi-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
.kpi-label { font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value { font-size: 32px; font-weight: 700; color: #0f172a; line-height: 1; }
.kpi-sub { font-size: 12px; color: #94a3b8; margin-top: 4px; }

.patient-card {
    background: white;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.risk-badge-High { background: #fef2f2; color: #dc2626; border: 1.5px solid #fca5a5; border-radius: 20px; padding: 4px 14px; font-weight: 600; font-size: 13px; }
.risk-badge-Medium { background: #fffbeb; color: #d97706; border: 1.5px solid #fcd34d; border-radius: 20px; padding: 4px 14px; font-weight: 600; font-size: 13px; }
.risk-badge-Low { background: #f0fdf4; color: #16a34a; border: 1.5px solid #86efac; border-radius: 20px; padding: 4px 14px; font-weight: 600; font-size: 13px; }

.insight-card {
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    border-left: 4px solid;
}
.insight-warning { background: #fffbeb; border-color: #f59e0b; }
.insight-danger  { background: #fef2f2; border-color: #ef4444; }
.insight-success { background: #f0fdf4; border-color: #22c55e; }
.insight-info    { background: #eff6ff; border-color: #3b82f6; }

.missing-badge {
    display: inline-block;
    background: #f1f5f9;
    color: #475569;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    margin: 3px;
    font-family: monospace;
}
.section-header {
    font-size: 13px; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin: 20px 0 10px; border-bottom: 1px solid #f1f5f9; padding-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
CHART_BASE = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    margin=dict(t=50, b=40, l=55, r=20),
    yaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb", showline=True),
    xaxis=dict(linecolor="#e5e7eb", showline=True),
)

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model    = joblib.load("models/random_forest.pkl")
    mapping  = joblib.load("models/target_mapping.pkl")
    metrics  = joblib.load("models/metrics.pkl") if os.path.exists("models/metrics.pkl") else {}
    feat_imp = joblib.load("models/feature_importance.pkl") if os.path.exists("models/feature_importance.pkl") else None
    return model, mapping, metrics, feat_imp

@st.cache_data(show_spinner="Reading file…")
def process_file(file_bytes, file_name):
    buf = io.BytesIO(file_bytes)
    return pd.read_csv(buf) if file_name.endswith(".csv") else pd.read_excel(buf)

@st.cache_data(show_spinner="Computing SHAP…", max_entries=3)
def compute_shap(_model, X_np, feature_names):
    try:
        import shap
        explainer = shap.TreeExplainer(_model)
        sv = explainer.shap_values(X_np)
        return sv, explainer.expected_value
    except Exception:
        return None, None

def build_patient_csv(patient_row, risk, patient_id=None):
    """Build a single-patient summary as CSV bytes including Outcome and Outcome_Reason."""
    try:
        outcome = patient_row["Outcome"] if "Outcome" in patient_row.index and pd.notna(patient_row["Outcome"]) else "N/A"
    except Exception:
        outcome = "N/A"
    try:
        reason = patient_row["Outcome_Reason"] if "Outcome_Reason" in patient_row.index and pd.notna(patient_row["Outcome_Reason"]) else "N/A"
    except Exception:
        reason = "N/A"

    export_fields = [
        ("Patient_ID",     str(patient_id) if patient_id else "N/A"),
        ("Predicted_Risk", risk),
        ("Outcome",        outcome),
        ("Outcome_Reason", reason),
    ]
    for key in ["Age","BMI","HbA1c_Baseline","HbA1c_Followup","HbA1c_Delta",
                "Adherence_Avg","Missed_Doses_Per_Month","Doctor_Visit_Frequency",
                "Fasting_Glucose_Baseline_mg_dL","Fasting_Glucose_Followup_mg_dL","Glucose_Delta"]:
        if key in patient_row.index and pd.notna(patient_row[key]):
            export_fields.append((key, patient_row[key]))
    df_export = pd.DataFrame(export_fields, columns=["Field", "Value"])
    return df_export.to_csv(index=False).encode("utf-8")

def _to_float(v):
    """Safely convert any value to float, return None if not possible."""
    if v is None: return None
    try:
        f = float(v)
        import math
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

def patient_insights(row, risk):
    """Generate actionable health insights for a patient."""
    insights = []

    adh    = _to_float(row.get("Adherence_Avg"))
    hba1c  = _to_float(row.get("HbA1c_Followup")) or _to_float(row.get("HbA1c_Baseline"))
    hba1c_b = _to_float(row.get("HbA1c_Baseline"))
    missed = _to_float(row.get("Missed_Doses_Per_Month"))
    bmi    = _to_float(row.get("BMI"))
    visits = _to_float(row.get("Doctor_Visit_Frequency"))
    delta  = _to_float(row.get("HbA1c_Delta"))

    # Adherence insights
    if adh is not None:
        if adh < 50:
            insights.append(("danger","Low Adherence Alert",
                f"Medication adherence is critically low at {adh:.0f}%. "
                "Recommend: daily pill organizer, SMS reminders, and family support check-in."))
        elif adh < 70:
            insights.append(("warning","Moderate Adherence",
                f"Adherence of {adh:.0f}% needs improvement. "
                "Consider smartphone reminder apps and bi-weekly check-in calls."))
        else:
            insights.append(("success","Good Adherence",
                f"Adherence of {adh:.0f}% is above target. Keep up the routine!"))

    # HbA1c insight
    if hba1c is not None:
        if hba1c >= 9.0:
            insights.append(("danger","Critical HbA1c Level",
                f"HbA1c of {hba1c:.1f}% is dangerously high (target <7%). "
                "Immediate physician review and possible medication adjustment required."))
        elif hba1c >= 7.5:
            insights.append(("warning","Elevated HbA1c",
                f"HbA1c of {hba1c:.1f}% exceeds the 7% target. "
                "Focus on diet, consistent medication, and increase doctor visits."))
        else:
            insights.append(("success","HbA1c Under Control",
                f"HbA1c of {hba1c:.1f}% is within or near target range. Maintain current plan."))

    # Improvement trend
    if delta is not None:
        if delta > 0.5:
            insights.append(("success","Positive Health Trend",
                f"HbA1c improved by {delta:.2f} points from baseline. "
                "Current treatment plan is working well — continue."))
        elif delta < 0:
            insights.append(("danger","Health Declining",
                f"HbA1c worsened by {abs(delta):.2f} points. "
                "Urgent: review medication adherence and dietary habits with physician."))

    # Missed doses
    if missed is not None:
        if missed >= 10:
            insights.append(("danger","High Missed Doses",
                f"{missed:.0f} missed doses/month detected. "
                "Recommend: blister packaging, caregiver involvement, and medication review."))
        elif missed >= 5:
            insights.append(("warning","Missed Doses Detected",
                f"{missed:.0f} doses missed per month. "
                "Set daily alarms and use a pill-tracking app to reduce this."))

    # BMI
    if bmi is not None:
        if bmi >= 30:
            insights.append(("warning","Obesity Concern",
                f"BMI of {bmi:.1f} increases insulin resistance. "
                "Recommend: registered dietitian referral and 150 min/week of physical activity."))
        elif bmi >= 25:
            insights.append(("info","Overweight",
                f"BMI of {bmi:.1f} is slightly above normal. "
                "Light dietary changes and regular walking can help significantly."))

    # Doctor visits
    if visits is not None and visits < 2:
        insights.append(("warning","Infrequent Doctor Visits",
            f"Only {visits:.0f} visits recorded. "
            "Quarterly check-ups are recommended for diabetes management."))

    # ── Outcome analysis: 3 categories ──────────────────────────────────────────
    if adh is not None and hba1c is not None:
        if adh >= 70 and hba1c < 8.0:
            insights.append(("success","Outcome: Well Controlled",
                f"Good adherence ({adh:.0f}%) and HbA1c under control ({hba1c:.1f}%). "
                "Current treatment plan is working — maintain routine."))
        elif adh >= 70 and hba1c >= 8.0:
            insights.append(("danger","Outcome: Treatment Ineffective",
                f"Patient has good adherence ({adh:.0f}%) but HbA1c remains high ({hba1c:.1f}%). "
                "The medication may be insufficient — physician should review drug dosage or type."))
        else:
            insights.append(("warning","Outcome: Poor Adherence",
                f"Low adherence ({adh:.0f}%) is driving elevated HbA1c ({hba1c:.1f}%). "
                "Focus on behaviour support, reminders, and caregiver involvement before changing medication."))

    # Risk-level summary
    if risk == "High":
        insights.append(("danger","Overall: High Risk",
            "This patient needs immediate clinical attention. "
            "Assign a case manager and schedule a review within 2 weeks."))
    elif risk == "Medium":
        insights.append(("warning","Overall: Medium Risk",
            "Monitor closely. Monthly adherence tracking and HbA1c testing every 3 months recommended."))
    else:
        insights.append(("success","Overall: Low Risk",
            "Patient is well-managed. Annual HbA1c review is sufficient. Encourage maintaining current habits."))

    return insights

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedAdherence AI")
    st.caption("Medication Adherence vs Outcome Correlation")
    st.divider()
    file = st.file_uploader("📂 Upload Patient Dataset", type=["csv","xlsx"])
    st.divider()
    st.markdown("**Model Status**")
    try:
        model, mapping, metrics, feat_imp = load_model()
        st.success(f"✅ Model loaded  |  Classes: {len(mapping)}")
        if metrics:
            st.metric("Test Accuracy", f"{metrics.get('accuracy',0):.1%}")
            if metrics.get("auc"):
                st.metric("ROC-AUC", f"{metrics['auc']:.3f}")
    except Exception:
        st.warning("⚠️ No model found. Run `python main.py` first.")
        model = mapping = metrics = feat_imp = None

    # ── FIX 1: Removed "Filter by Risk Tier" multiselect ──────────────────────
    st.divider()
    st.markdown("**More Filters**")

# ── LANDING ────────────────────────────────────────────────────────────────────
if not file:
    st.markdown("# 🏥 AI-Powered Medication Adherence System")
    st.markdown("Upload a patient dataset in the sidebar to begin.")
    st.stop()

if model is None:
    st.error("❌ No trained model found. Run `python main.py` first.")
    st.stop()

# ── LOAD FILE ──────────────────────────────────────────────────────────────────
raw_bytes = file.read()
try:
    df_raw = process_file(raw_bytes, file.name)
except Exception as e:
    st.error(f"❌ Could not read file: {e}")
    st.stop()

# ── VALIDATE — warn on missing optional cols, never block ─────────────────────
ok, msg, missing_optional = validate_schema(df_raw)
if not ok:
    st.error(f"❌ {msg}")
    st.stop()

if missing_optional:
    with st.expander(f"⚠️  {len(missing_optional)} optional column(s) missing — app will continue with available data", expanded=False):
        st.markdown("The following columns were not found in your file. Charts that depend on them will be skipped:")
        badges = " ".join(f'<span class="missing-badge">{c}</span>' for c in missing_optional)
        st.markdown(badges, unsafe_allow_html=True)
        st.caption("This does NOT stop the app — all available data will be used normally.")

# ── PREPROCESS & PREDICT ───────────────────────────────────────────────────────
try:
    df_clean = clean_data(df_raw.copy())
except Exception as e:
    st.error(f"❌ Preprocessing failed: {e}")
    st.stop()

try:
    X = df_clean.drop(columns=["Patient_Risk_Tier","Health_Improvement_Score"], errors="ignore")
    X = X.select_dtypes(include="number")
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    X = X[model.feature_names_in_]

    preds = model.predict(X)
    proba = model.predict_proba(X)
    df_clean["Predicted_Risk"] = [mapping[p] for p in preds]
    df_clean["Confidence"]     = proba.max(axis=1).round(3)
    if "Patient_ID" in df_raw.columns:
        df_clean["Patient_ID"] = df_raw["Patient_ID"].values

    # Rule-based override for small datasets (model trained on 100k can miss Medium)
    if len(df_clean) < 100 and "Adherence_Avg" in df_clean.columns:
        def rule_risk(row):
            adh   = row.get("Adherence_Avg", 60)
            miss  = row.get("Missed_Doses_Per_Month", 5)
            hba1c = row.get("HbA1c_Followup", row.get("HbA1c_Baseline", 8))
            if adh >= 75 and miss <= 4 and hba1c <= 7.5:
                return "Low"
            elif adh <= 45 or miss >= 12 or hba1c >= 9.0:
                return "High"
            return "Medium"
        df_clean["Predicted_Risk"] = df_clean.apply(rule_risk, axis=1)

except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# ── Assign Outcome label ──────────────────────────────────────────────────────
def assign_outcome(row):
    adh   = _to_float(row.get("Adherence_Avg", row.get("Adherence_Month1_Pct", 60)))
    hba1c = _to_float(row.get("HbA1c_Followup", row.get("HbA1c_Baseline", 8)))
    if adh is None or hba1c is None: return "Well Controlled"
    if adh >= 70 and hba1c < 8.0:   return "Well Controlled"
    if adh >= 70 and hba1c >= 8.0:  return "Treatment Ineffective"
    return "Poor Adherence"

def assign_outcome_reason(row):
    """Return a short human-readable explanation of why the outcome was assigned."""
    adh    = _to_float(row.get("Adherence_Avg", row.get("Adherence_Month1_Pct")))
    hba1c  = _to_float(row.get("HbA1c_Followup", row.get("HbA1c_Baseline")))
    missed = _to_float(row.get("Missed_Doses_Per_Month"))
    delta  = _to_float(row.get("HbA1c_Delta"))
    outcome = row.get("Outcome", "")

    if outcome == "Well Controlled":
        parts = []
        if adh  is not None: parts.append(f"Adherence {adh:.0f}% (>=70%)")
        if hba1c is not None: parts.append(f"HbA1c {hba1c:.1f}% (<8.0%)")
        if delta is not None and delta > 0: parts.append(f"HbA1c improved by {delta:.2f} pts")
        return "Good adherence and HbA1c within target. " + "; ".join(parts) if parts else "Meets adherence and HbA1c targets."

    elif outcome == "Treatment Ineffective":
        parts = []
        if adh   is not None: parts.append(f"Adherence {adh:.0f}% (>=70% — patient is compliant)")
        if hba1c is not None: parts.append(f"HbA1c {hba1c:.1f}% (>=8.0% despite compliance)")
        if delta is not None and delta < 0: parts.append(f"HbA1c worsened by {abs(delta):.2f} pts")
        return "Patient takes medication but HbA1c remains high — medication may need review. " + "; ".join(parts)

    else:  # Poor Adherence
        parts = []
        if adh   is not None: parts.append(f"Adherence {adh:.0f}% (<70%)")
        if missed is not None: parts.append(f"{missed:.0f} missed doses/month")
        if hba1c  is not None: parts.append(f"HbA1c {hba1c:.1f}%")
        return "Low medication adherence is driving poor glycaemic control. " + "; ".join(parts)

df_clean["Outcome"] = df_clean.apply(assign_outcome, axis=1)
df_clean["Outcome_Reason"] = df_clean.apply(assign_outcome_reason, axis=1)

# ── FIX 1: No risk_filter — df_view shows all patients ────────────────────────
df_view = df_clean.copy()

# Dynamic diabetes type filter (needs df loaded first)
diab_col = None
for c in df_raw.columns:
    if "diabetes" in c.lower() and "type" in c.lower():
        diab_col = c; break
if diab_col and diab_col in df_raw.columns:
    all_types = sorted(df_raw[diab_col].dropna().unique().tolist())
    with st.sidebar:
        diab_sel = st.multiselect("Diabetes Type", all_types, default=all_types, key="diab_actual")
    if diab_sel and "Patient_ID" in df_raw.columns:
        valid_ids = df_raw[df_raw[diab_col].isin(diab_sel)]["Patient_ID"].astype(str).tolist()
        if "Patient_ID" in df_view.columns:
            df_view = df_view[df_view["Patient_ID"].astype(str).isin(valid_ids)]

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📊 Dashboard","📈 Analytics","🔍 Patient Lookup","🧠 Explainability","🤖 AI Assistant","📉 Model Performance","📄 Data & Export"])

# ─── TAB 1: DASHBOARD ─────────────────────────────────────────────────────────
with tabs[0]:
    total  = len(df_view)
    high   = (df_view["Predicted_Risk"]=="High").sum()
    medium = (df_view["Predicted_Risk"]=="Medium").sum()
    low    = (df_view["Predicted_Risk"]=="Low").sum()
    avg_adh = df_view["Adherence_Avg"].mean() if "Adherence_Avg" in df_view.columns else None
    avg_conf = df_view["Confidence"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, val, sub in [
        (c1,"Total Patients", f"{total:,}", "in dataset"),
        (c2,"🔴 High Risk",   f"{high:,}",  f"{high/total:.0%} of total" if total else ""),
        (c3,"🟡 Medium Risk", f"{medium:,}", f"{medium/total:.0%} of total" if total else ""),
        (c4,"🟢 Low Risk",    f"{low:,}",   f"{low/total:.0%} of total" if total else ""),
        (c5,"Avg Adherence",  f"{avg_adh:.1f}%" if avg_adh else "N/A", f"Confidence {avg_conf:.0%}"),
    ]:
        col.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Outcome summary row
    if "Outcome" in df_view.columns:
        outcome_counts = df_view["Outcome"].value_counts()
        outcome_colors = {
            "Well Controlled": "#22c55e",
            "Treatment Ineffective": "#ef4444",
            "Poor Adherence": "#f97316",
        }
        out_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-bottom:20px">'
        for outcome, cnt in outcome_counts.items():
            color = outcome_colors.get(outcome, "#6366f1")
            pct = cnt / len(df_view) * 100 if len(df_view) else 0
            out_html += f"""<div style="border-left:4px solid {color};background:#fafafa;border-radius:0 10px 10px 0;padding:12px 14px">
                <div style="font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.05em">{outcome}</div>
                <div style="font-size:20px;font-weight:700;color:#0f172a">{cnt} <span style="font-size:13px;color:{color};font-weight:500">({pct:.0f}%)</span></div>
            </div>"""
        out_html += '</div>'
        st.markdown("**Outcome Analysis**", unsafe_allow_html=False)
        st.markdown(out_html, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        fig_pie = px.pie(df_view, names="Predicted_Risk",
            color="Predicted_Risk", hole=0.45,
            color_discrete_map=RISK_COLORS,
            title="Risk Distribution")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
            textfont_size=13, marker=dict(line=dict(color="white",width=2)))
        fig_pie.update_layout(**CHART_BASE, showlegend=True,
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        if "Adherence_Avg" in df_view.columns and "HbA1c_Followup" in df_view.columns:
            sample = df_view.sample(min(500, len(df_view)))
            fig_sc = px.scatter(
                sample, x="Adherence_Avg", y="HbA1c_Followup",
                color="Predicted_Risk",
                color_discrete_map=RISK_COLORS,
                opacity=0.75,
                title="Adherence vs HbA1c Outcome",
                labels={
                    "Adherence_Avg": "Adherence (%)",
                    "HbA1c_Followup": "HbA1c (%)",
                    "Predicted_Risk": "Risk",
                }
            )
            fig_sc.update_traces(marker=dict(size=9, line=dict(width=1, color="white")))
            sc_layout = {**CHART_BASE}
            sc_layout["xaxis"] = dict(title=dict(text="Adherence (%)", font=dict(size=12)),
                tickfont=dict(size=11), linecolor="#e5e7eb", showline=True, gridcolor="#f3f4f6")
            sc_layout["yaxis"] = dict(title=dict(text="HbA1c (%)", font=dict(size=12)),
                tickfont=dict(size=11), linecolor="#e5e7eb", showline=True, gridcolor="#f3f4f6")
            sc_layout["legend"] = dict(title=dict(text="Risk Tier", font=dict(size=11)),
                orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=11),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1)
            sc_layout["margin"] = dict(t=60, b=50, l=60, r=20)
            fig_sc.update_layout(**sc_layout)
            fig_sc.add_hline(
                y=7.0, line_dash="dot", line_color="#94a3b8", line_width=1.5,
                annotation_text="Target HbA1c 7%",
                annotation_position="top right",
                annotation_font=dict(size=10, color="#94a3b8"),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

# ─── TAB 2: ANALYTICS ─────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Analytics")

    RISK_ORDER = ["High", "Medium", "Low"]

    def styled_bar(x_vals, y_vals, title, yaxis_title, text_vals=None, yrange=None, xaxis_title="Risk Tier"):
        # Sort by High → Medium → Low
        paired = list(zip(x_vals, y_vals, text_vals if text_vals else [f"{v:.1f}" for v in y_vals]))
        paired.sort(key=lambda t: RISK_ORDER.index(t[0]) if t[0] in RISK_ORDER else 99)
        x_vals  = [p[0] for p in paired]
        y_vals  = [p[1] for p in paired]
        txt     = [p[2] for p in paired]
        colors  = [RISK_COLORS.get(str(x), "#6366f1") for x in x_vals]
        fig = go.Figure(go.Bar(
            x=x_vals, y=y_vals,
            marker_color=colors,
            marker_line=dict(color="white", width=1.5),
            text=txt,
            textposition="outside",
            textfont=dict(size=13, color="#374151"),
        ))
        layout = dict(
            title=dict(text=title, font=dict(size=14, color="#0f172a")),
            yaxis_title=yaxis_title,
            xaxis_title=xaxis_title,
            showlegend=False,
            **CHART_BASE,
        )
        if yrange:
            layout["yaxis_range"] = yrange
        fig.update_layout(**layout)
        return fig

    if "HbA1c_Baseline" in df_view.columns and "HbA1c_Followup" in df_view.columns:
        grp = df_view.groupby("Predicted_Risk")[["HbA1c_Baseline","HbA1c_Followup"]].mean().round(2).reset_index()
        grp = grp[grp["Predicted_Risk"].isin(RISK_COLORS)]
        # Order: High → Medium → Low
        risk_order_map = {"High": 0, "Medium": 1, "Low": 2}
        grp = grp.sort_values("Predicted_Risk", key=lambda s: s.map(risk_order_map))
        fig_hba = go.Figure()
        bar_colors_baseline = [RISK_COLORS[r] for r in grp["Predicted_Risk"]]
        bar_colors_followup = [RISK_COLORS[r] for r in grp["Predicted_Risk"]]
        for col_name, label, opacity in [("HbA1c_Baseline","Baseline",0.45),("HbA1c_Followup","Follow-up",1.0)]:
            fig_hba.add_trace(go.Bar(
                x=grp["Predicted_Risk"], y=grp[col_name], name=label,
                marker_color=[RISK_COLORS[r] for r in grp["Predicted_Risk"]],
                marker_opacity=opacity,
                marker_line=dict(color="white",width=1.5),
                text=grp[col_name].apply(lambda v: f"{v:.1f}"),
                textposition="outside",
            ))
        # Reference line: mean difference between baseline and follow-up (population level)
        ref_val = df_view[["HbA1c_Baseline","HbA1c_Followup"]].mean().mean()
        fig_hba.update_layout(
            barmode="group",
            title="HbA1c: Baseline vs Follow-up",
            yaxis_title="HbA1c (%)",
            xaxis_title="Risk Tier",
            showlegend=True,
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            **CHART_BASE,
        )
        fig_hba.add_hline(
            y=7.0,
            line_dash="dash",
            line_color="#6366f1",
            line_width=1.8,
            annotation_text="HbA1c Target (7%)",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#6366f1"),
        )
        st.plotly_chart(fig_hba, use_container_width=True)

    st.divider()
    # ── Combined: Avg Medication Adherence & Avg Missed Doses by Risk Tier ──────
    if "Adherence_Avg" in df_view.columns and "Missed_Doses_Per_Month" in df_view.columns:
        d_adh = df_view.groupby("Predicted_Risk")["Adherence_Avg"].mean().round(1).reset_index()
        d_adh = d_adh[d_adh["Predicted_Risk"].isin(RISK_COLORS)]
        d_mis = df_view.groupby("Predicted_Risk")["Missed_Doses_Per_Month"].mean().round(1).reset_index()
        d_mis = d_mis[d_mis["Predicted_Risk"].isin(RISK_COLORS)]
        # Merge and sort High → Medium → Low
        risk_order_map2 = {"High": 0, "Medium": 1, "Low": 2}
        d_adh = d_adh.sort_values("Predicted_Risk", key=lambda s: s.map(risk_order_map2))
        d_mis = d_mis.sort_values("Predicted_Risk", key=lambda s: s.map(risk_order_map2))

        # Build long-form dataframe for side-by-side grouped bars
        d_adh_renamed = d_adh.rename(columns={"Adherence_Avg": "Value"})
        d_adh_renamed["Metric"] = "Avg Medication Adherence (%)"
        d_mis_renamed = d_mis.rename(columns={"Missed_Doses_Per_Month": "Value"})
        d_mis_renamed["Metric"] = "Avg Missed Doses / Month"
        d_combo = pd.concat([d_adh_renamed, d_mis_renamed], ignore_index=True)

        METRIC_COLORS = {
            "Avg Medication Adherence (%)": "#6366f1",
            "Avg Missed Doses / Month": "#f59e0b",
        }

        fig_combo = go.Figure()
        for metric, color in METRIC_COLORS.items():
            sub = d_combo[d_combo["Metric"] == metric]
            fig_combo.add_trace(go.Bar(
                x=sub["Predicted_Risk"],
                y=sub["Value"],
                name=metric,
                marker_color=color,
                marker_opacity=0.88,
                marker_line=dict(color="white", width=1.5),
                text=[f"{v:.1f}" + ("%" if "Adherence" in metric else "") for v in sub["Value"]],
                textposition="outside",
                textfont=dict(size=12, color="#374151"),
            ))
        fig_combo.update_layout(
            barmode="group",
            title=dict(text="Avg Medication Adherence & Avg Missed Doses by Risk Tier", font=dict(size=14, color="#0f172a")),
            xaxis=dict(title="Risk Tier", linecolor="#e5e7eb", showline=True),
            yaxis=dict(title="Value", gridcolor="#f3f4f6", linecolor="#e5e7eb", showline=True, range=[0, 120]),
            showlegend=True,
            legend=dict(orientation="h", y=1.14, x=0.5, xanchor="center", font=dict(size=12)),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter, sans-serif", color="#374151", size=12),
            margin=dict(t=80, b=50, l=60, r=30),
        )
        st.plotly_chart(fig_combo, use_container_width=True)
    elif "Adherence_Avg" in df_view.columns:
        d = df_view.groupby("Predicted_Risk")["Adherence_Avg"].mean().round(1).reset_index()
        d = d[d["Predicted_Risk"].isin(RISK_COLORS)]
        st.plotly_chart(styled_bar(d["Predicted_Risk"], d["Adherence_Avg"],
            "Avg Medication Adherence by Risk Tier", "Adherence (%)",
            [f"{v:.1f}%" for v in d["Adherence_Avg"]], [0, 115],
            xaxis_title="Level of Risk Tier"), use_container_width=True)
    elif "Missed_Doses_Per_Month" in df_view.columns:
        d = df_view.groupby("Predicted_Risk")["Missed_Doses_Per_Month"].mean().round(1).reset_index()
        d = d[d["Predicted_Risk"].isin(RISK_COLORS)]
        st.plotly_chart(styled_bar(d["Predicted_Risk"], d["Missed_Doses_Per_Month"],
            "Avg Missed Doses per Month", "Missed Doses / Month",
            xaxis_title="Level of Risk Tier"), use_container_width=True)

    if "HbA1c_Delta" in df_view.columns:
        d = df_view.groupby("Predicted_Risk")["HbA1c_Delta"].mean().round(2).reset_index()
        d = d[d["Predicted_Risk"].isin(RISK_COLORS)]
        st.plotly_chart(styled_bar(d["Predicted_Risk"], d["HbA1c_Delta"],
            "Avg HbA1c Improvement","Change (+ = improved)",
            [f"{v:+.2f}" for v in d["HbA1c_Delta"]],
            xaxis_title="Risk Tier"), use_container_width=True)

    if len(df_view) <= 50 and "Adherence_Avg" in df_view.columns:
        st.divider()
        pf = df_view.copy()
        pf["Label"] = pf["Patient_ID"].astype(str) if "Patient_ID" in pf.columns else [f"P{i+1}" for i in range(len(pf))]
        pf = pf.sort_values("Adherence_Avg", ascending=False)
        fig_per = go.Figure()
        for risk, color in RISK_COLORS.items():
            sub = pf[pf["Predicted_Risk"]==risk]
            if sub.empty: continue
            fig_per.add_trace(go.Bar(
                x=sub["Label"], y=sub["Adherence_Avg"], name=risk,
                marker_color=color, marker_line=dict(color="white",width=1),
                text=sub["Adherence_Avg"].apply(lambda v: f"{v:.0f}%"), textposition="outside",
            ))
        fig_per.update_layout(title="Adherence % per Patient", yaxis_title="Adherence (%)",
            yaxis_range=[0,115], barmode="stack", showlegend=True,
            legend=dict(orientation="h",y=1.1,x=0.5,xanchor="center"), **CHART_BASE)
        st.plotly_chart(fig_per, use_container_width=True)

# ─── TAB 3: PATIENT LOOKUP ────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Patient Lookup")
    id_col = "Patient_ID" if "Patient_ID" in df_clean.columns else None

    search_id = st.text_input("🔍 Enter Patient ID", placeholder="e.g. 1, 2, P001…")

    if search_id and id_col:
        result = df_clean[df_clean[id_col].astype(str) == search_id.strip()]
        if result.empty:
            st.warning(f"Patient '{search_id}' not found in the dataset.")
        else:
            row  = result.iloc[0]
            risk = str(row["Predicted_Risk"]) if "Predicted_Risk" in row.index else "Unknown"
            conf_raw = row["Confidence"] if "Confidence" in row.index else 0
            conf = float(conf_raw) if pd.notna(conf_raw) else 0.0

            # ── Header card ──────────────────────────────────────────────────
            badge = f'<span class="risk-badge-{risk}">{risk} Risk</span>'
            icon  = "🔴" if risk=="High" else ("🟡" if risk=="Medium" else "🟢")
            st.markdown(f"""<div class="patient-card">
                <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
                    <div style="font-size:40px">{icon}</div>
                    <div>
                        <div style="font-size:22px;font-weight:700;color:#0f172a">Patient {search_id}</div>
                        <div style="margin-top:4px">{badge} &nbsp; <span style="color:#64748b;font-size:13px">Confidence: {conf:.0%}</span></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── Key metrics ──────────────────────────────────────────────────
            st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
            metric_fields = [
                ("Age","Age","yrs"), ("BMI","BMI",""), ("HbA1c Baseline","HbA1c_Baseline","%"),
                ("HbA1c Follow-up","HbA1c_Followup","%"), ("Adherence","Adherence_Avg","%"),
                ("Missed Doses/Mo","Missed_Doses_Per_Month",""), ("Doctor Visits","Doctor_Visit_Frequency",""),
                ("HbA1c Change","HbA1c_Delta",""), ("Glucose Change","Glucose_Delta","mg/dL"),
            ]
            # ── FIX 2: Only include fields where value exists AND is not NaN ──
            present = [
                (lbl, key, unit)
                for lbl, key, unit in metric_fields
                if key in row.index and pd.notna(row[key]) and str(row[key]).strip().lower() not in ("", "none", "nan")
            ]
            if present:
                cards_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin:12px 0 20px">'
                for lbl, key, unit in present:
                    val = row[key]
                    val_str = f"{val:.1f}{unit}" if isinstance(val, float) else f"{val}{unit}"
                    cards_html += f"""<div style="background:#f8faff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;text-align:center">
                        <div style="font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">{lbl}</div>
                        <div style="font-size:22px;font-weight:700;color:#0f172a">{val_str}</div>
                    </div>"""
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)

            # ── Health Insights ──────────────────────────────────────────────
            st.markdown('<div class="section-header">Health Insights & Recommendations</div>', unsafe_allow_html=True)
            row_dict = {}
            for k in row.index:
                try:
                    v = row[k]
                    if pd.isna(v): continue
                    row_dict[k] = float(v) if hasattr(v, '__float__') else v
                except Exception:
                    pass
            ins_list = patient_insights(row_dict, risk)
            icon_map = {"danger":"🔴","warning":"🟡","success":"🟢","info":"🔵"}
            for kind, title, msg in ins_list:
                st.markdown(f"""<div class="insight-card insight-{kind}">
                    <strong>{icon_map.get(kind,"")} {title}</strong><br>
                    <span style="font-size:13px;color:#374151">{msg}</span>
                </div>""", unsafe_allow_html=True)

            # ── Export: single-patient CSV (no external dependency) ──────────
            st.divider()
            st.markdown("**📤 Export This Patient**")
            pid = search_id.strip()
            csv_bytes = build_patient_csv(row, risk, patient_id=pid)
            st.download_button(
                label="⬇️ Download Patient Summary (CSV)",
                data=csv_bytes,
                file_name=f"patient_{pid}_summary.csv",
                mime="text/csv",
                key="patient_csv_download",
            )

    elif not id_col:
        st.info("No Patient_ID column found. Showing top 20 records.")
        show = ["Predicted_Risk","Confidence"] + [c for c in df_view.columns if c not in ("Predicted_Risk","Confidence")]
        st.dataframe(df_view[show].head(20), use_container_width=True)

# ─── TAB 4: EXPLAINABILITY ────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🧠 Explainability")
    st.caption("Understand why each patient received their risk prediction.")

    if feat_imp is not None:
        fig_imp = px.bar(feat_imp.head(15), x="Importance", y="Feature", orientation="h",
            title="Top 15 Predictive Features (Global)",
            color="Importance", color_continuous_scale="Blues")
        fig_imp.update_layout(**{**CHART_BASE, "yaxis": {"autorange":"reversed","gridcolor":"#f3f4f6","linecolor":"#e5e7eb","showline":True}, "coloraxis_showscale": False})
        st.plotly_chart(fig_imp, use_container_width=True)



# ─── TAB 5: AI ASSISTANT ──────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("🤖 AI Clinical Assistant")
    st.caption("Ask questions about your patient cohort — powered by your data, no API key needed.")

    if "chat_stats" not in st.session_state:
        st.session_state.chat_stats = build_stats(df_clean, feat_imp, metrics)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.chat_history:
        st.markdown("**Try one of these:**")
        prompts = [
            "How many high risk patients?",
            "What is the average adherence?",
            "Show HbA1c trends",
            "What are the top risk factors?",
            "Give me recommendations",
            "What is the model accuracy?",
        ]
        cols = st.columns(3)
        for i, p in enumerate(prompts):
            if cols[i%3].button(p, use_container_width=True, key=f"qp_{i}"):
                reply = respond(p, st.session_state.chat_stats)
                st.session_state.chat_history += [{"role":"user","content":p},{"role":"assistant","content":reply}]
                st.rerun()

    if user_input := st.chat_input("Ask about your patient cohort…"):
        reply = respond(user_input, st.session_state.chat_stats)
        st.session_state.chat_history += [{"role":"user","content":user_input},{"role":"assistant","content":reply}]
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()
# ─── TAB 6: MODEL PERFORMANCE ────────────────────────────────────────────────
with tabs[5]:
    st.subheader("📉 Model Performance")

    def compute_live_metrics(df_c, mdl, mp):
        try:
            from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
            from sklearn.model_selection import cross_val_score

            label_col = None
            for candidate in ["Patient_Risk_Tier", "Risk_Tier", "Risk", "Label"]:
                if candidate in df_c.columns:
                    label_col = candidate
                    break

            X_live = df_c.drop(columns=[
                "Patient_Risk_Tier","Health_Improvement_Score",
                "Predicted_Risk","Confidence","Outcome","Outcome_Reason"
            ], errors="ignore")

            X_live = X_live.select_dtypes(include="number")

            for col in mdl.feature_names_in_:
                if col not in X_live.columns:
                    X_live[col] = 0

            X_live = X_live[mdl.feature_names_in_]

            preds_live = mdl.predict(X_live)
            proba_live = mdl.predict_proba(X_live)

            if isinstance(mp, dict):
                class_names = [mp[i] for i in sorted(mp.keys())]
            else:
                class_names = list(mp)

            result = {
                "n_samples": len(X_live),
                "class_names": class_names,
                "preds": preds_live,
                "proba": proba_live,
            }

            if label_col:
                y_true = df_c[label_col].values

                if isinstance(mp, dict):
                    rev = {v: k for k, v in mp.items()}
                else:
                    rev = {str(v): i for i, v in enumerate(mp)}

                y_true_int = np.array([rev.get(str(y), -1) for y in y_true])
                valid = y_true_int >= 0

                if valid.sum() > 0:
                    acc = accuracy_score(y_true_int[valid], preds_live[valid])

                    cm = confusion_matrix(
                        y_true_int[valid],
                        preds_live[valid]
                    )

                    try:
                        auc = roc_auc_score(
                            y_true_int[valid],
                            proba_live[valid],
                            multi_class="ovr"
                        )
                    except:
                        auc = None

                    try:
                        cv_scores = cross_val_score(
                            mdl, X_live[valid], y_true_int[valid], cv=5
                        )
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    except:
                        cv_mean, cv_std = None, None

                    result.update({
                        "has_labels": True,
                        "accuracy": acc,
                        "confusion_matrix": cm.tolist(),
                        "auc": auc,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "label_col": label_col
                    })
                    return result

            result["has_labels"] = False
            return result

        except Exception as e:
            return {"error": str(e)}

    live = compute_live_metrics(df_clean, model, mapping)

    if "error" in live:
        st.warning(f"⚠️ Error computing live metrics: {live['error']}")

    has_live_labels = live.get("has_labels", False)

    if has_live_labels:
        st.success("✅ Live metrics computed from your uploaded dataset")
    else:
        st.info("ℹ️ Showing pre-trained model metrics (no labels found)")

    disp_accuracy = live.get("accuracy", metrics.get("accuracy")) if has_live_labels else metrics.get("accuracy")
    disp_auc      = live.get("auc", metrics.get("auc")) if has_live_labels else metrics.get("auc")
    disp_cv_mean  = live.get("cv_mean", metrics.get("cv_mean")) if has_live_labels else metrics.get("cv_mean")
    disp_cv_std   = live.get("cv_std", metrics.get("cv_std")) if has_live_labels else metrics.get("cv_std")
    disp_cm_raw   = live.get("confusion_matrix", metrics.get("confusion_matrix")) if has_live_labels else metrics.get("confusion_matrix")
    disp_classes  = live.get("class_names", metrics.get("classes"))
    n_samples     = live.get("n_samples", metrics.get("n_test"))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test Accuracy", f"{disp_accuracy:.2%}" if disp_accuracy else "N/A")
    c2.metric("CV Accuracy",
              f"{disp_cv_mean:.2%}" if disp_cv_mean else "N/A",
              f"± {disp_cv_std:.2%}" if disp_cv_std else "")
    c3.metric("ROC-AUC", f"{disp_auc:.3f}" if disp_auc else "N/A")
    c4.metric("Dataset Samples", f"{n_samples:,}" if n_samples else "N/A")

    st.markdown("---")

    # ✅ CONFUSION MATRIX WITH VALUES INSIDE
    if disp_cm_raw is not None and disp_classes is not None:
        cm_arr = np.array(disp_cm_raw)

        fig_cm = go.Figure(go.Heatmap(
            z=cm_arr,
            x=disp_classes,
            y=disp_classes,
            colorscale="Blues",
            text=cm_arr,                  # 👈 values inside
            texttemplate="%{text}",       # 👈 display format
            textfont={"size": 16}
        ))

        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual"
        )

        st.plotly_chart(fig_cm, use_container_width=True)
# ─── TAB 7: DATA & EXPORT ─────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Processed Data")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        tbl_risk = st.multiselect("Risk Tier", ["High","Medium","Low"],
            default=["High","Medium","Low"], key="tbl_risk")
    with fc2:
        tbl_outcome = st.multiselect("Outcome",
            ["Well Controlled","Treatment Ineffective","Poor Adherence"],
            default=["Well Controlled","Treatment Ineffective","Poor Adherence"], key="tbl_outcome")
    with fc3:
        if "Age" in df_view.columns:
            age_min, age_max = int(df_view["Age"].min()), int(df_view["Age"].max())
            tbl_age = st.slider("Age Range", age_min, age_max, (age_min, age_max), key="tbl_age")
        else:
            tbl_age = None

    tbl_df = df_view.copy()
    if tbl_risk:
        tbl_df = tbl_df[tbl_df["Predicted_Risk"].isin(tbl_risk)]
    if tbl_outcome and "Outcome" in tbl_df.columns:
        tbl_df = tbl_df[tbl_df["Outcome"].isin(tbl_outcome)]
    if tbl_age and "Age" in tbl_df.columns:
        tbl_df = tbl_df[(tbl_df["Age"] >= tbl_age[0]) & (tbl_df["Age"] <= tbl_age[1])]

    st.caption(f"Showing {len(tbl_df):,} of {len(df_view):,} patients")

    show_cols = (["Patient_ID","Predicted_Risk","Outcome","Outcome_Reason","Confidence"] if "Patient_ID" in tbl_df.columns
                 else ["Predicted_Risk","Outcome","Outcome_Reason","Confidence"])
    show_cols += [c for c in ["Age","BMI","Adherence_Avg","HbA1c_Baseline","HbA1c_Followup",
                               "Missed_Doses_Per_Month","HbA1c_Delta"] if c in tbl_df.columns]
    show_cols = [c for c in show_cols if c in tbl_df.columns]
    order = {"High":0,"Medium":1,"Low":2}
    sorted_df = tbl_df[show_cols].sort_values("Predicted_Risk", key=lambda x: x.map(order))
    st.dataframe(sorted_df, use_container_width=True, height=420)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📥 Download Full CSV", df_clean.to_csv(index=False),
            "processed_predictions.csv", "text/csv")
    with c2:
        st.download_button("📥 Download Filtered CSV", sorted_df.to_csv(index=False),
            "filtered_patients.csv", "text/csv")
    with c3:
        hr = df_clean[df_clean["Predicted_Risk"]=="High"]
        st.download_button("⚠️ High-Risk Patients CSV", hr.to_csv(index=False),
            "high_risk_patients.csv", "text/csv")
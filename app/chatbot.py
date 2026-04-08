"""
Local AI Assistant — no API key. Uses actual dataset stats to answer precisely.
"""
import re
import pandas as pd
import numpy as np

INTENTS = [
    (re.compile(r"how many|total patient|count patient|number of patient", re.I),  "count"),
    (re.compile(r"high.?risk|critical|danger|worst", re.I),                         "high_risk"),
    (re.compile(r"low.?risk|best|safe|well.?manag", re.I),                          "low_risk"),
    (re.compile(r"medium.?risk|moderate|middle", re.I),                             "medium_risk"),
    (re.compile(r"distribut|breakdown|split|how many.*risk|risk.*breakdown", re.I), "distribution"),
    (re.compile(r"adher", re.I),                                                    "adherence"),
    (re.compile(r"hba1c|hemoglobin|a1c|glycat", re.I),                             "hba1c"),
    (re.compile(r"miss(ed)?.?dose|skip|non.?adher|forgot", re.I),                  "missed_doses"),
    (re.compile(r"\bage\b|old|young|how old", re.I),                               "age"),
    (re.compile(r"bmi|weight|obese|overweight|body mass", re.I),                   "bmi"),
    (re.compile(r"glucose|fasting|blood.?sugar", re.I),                            "glucose"),
    (re.compile(r"doctor|visit|frequency|appointment|checkup", re.I),              "doctor_visits"),
    (re.compile(r"improve|better|trend|progress|getting", re.I),                   "improvement"),
    (re.compile(r"recommend|suggest|interven|action|what.?should|how to help|reduce risk", re.I), "recommendation"),
    (re.compile(r"feature|important|factor|driver|cause|what.*affect|which.*matter", re.I), "features"),
    (re.compile(r"confiden|certain|reliab|how sure|predict.*accur", re.I),         "confidence"),
    (re.compile(r"model|accuracy|auc|performance|f1|precision|recall|how good", re.I), "model_perf"),
    (re.compile(r"\bhi\b|\bhello\b|\bhey\b|help|what can|what do you", re.I),     "help"),
]

def _fmt(val, d=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{d}f}"

def _pct(n, total):
    return f"{n} ({n/total:.0%})" if total else str(n)

def build_stats(df, feat_imp=None, metrics=None):
    total = len(df)
    risk_counts = df["Predicted_Risk"].value_counts().to_dict() if "Predicted_Risk" in df.columns else {}

    def cs(col):
        if col not in df.columns: return None
        # Convert to numeric first — text values like "Semi-Annual" become NaN and are skipped
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty: return None
        by_risk = {}
        if "Predicted_Risk" in df.columns:
            try:
                num_col = pd.to_numeric(df[col], errors="coerce")
                by_risk = df.assign(**{col: num_col}).groupby("Predicted_Risk")[col].mean().dropna().round(2).to_dict()
            except Exception:
                by_risk = {}
        return {"mean": round(float(s.mean()),2), "median": round(float(s.median()),2),
                "min": round(float(s.min()),2), "max": round(float(s.max()),2), "by_risk": by_risk}

    top_features = list(feat_imp.head(5)["Feature"]) if feat_imp is not None else []

    return {
        "total": total, "risk_counts": risk_counts,
        "adherence": cs("Adherence_Avg"), "hba1c_baseline": cs("HbA1c_Baseline"),
        "hba1c_followup": cs("HbA1c_Followup"), "hba1c_delta": cs("HbA1c_Delta"),
        "missed_doses": cs("Missed_Doses_Per_Month"), "age": cs("Age"), "bmi": cs("BMI"),
        "glucose_b": cs("Fasting_Glucose_Baseline_mg_dL"), "glucose_f": cs("Fasting_Glucose_Followup_mg_dL"),
        "doctor_visits": cs("Doctor_Visit_Frequency"), "confidence": cs("Confidence"),
        "top_features": top_features, "metrics": metrics or {},
    }

def respond(query, stats):
    matched = [key for pat, key in INTENTS if pat.search(query)]
    total = stats["total"]
    rc    = stats["risk_counts"]

    if not matched or "help" in matched:
        return (
            "👋 I can answer questions about your patient dataset. Try:\n\n"
            "- **'How many high risk patients?'** → exact count\n"
            "- **'What is the average adherence?'** → overall + by risk tier\n"
            "- **'Show HbA1c trends'** → baseline vs follow-up\n"
            "- **'What are the top risk factors?'** → feature importance\n"
            "- **'What interventions reduce risk?'** → clinical recommendations\n"
            "- **'What is the model accuracy?'** → performance metrics\n"
            "- **'Give me the risk breakdown'** → all tiers at once"
        )

    # ── Distribution ──────────────────────────────────────────────────────────
    if "distribution" in matched or ("count" in matched and len(matched)==1):
        lines = [f"📊 **Risk distribution across {total} patients:**\n"]
        for tier in ["High","Medium","Low"]:
            n = rc.get(tier, 0)
            bar = "█" * max(1, int(n/max(total,1)*20))
            icon = {"High":"🔴","Medium":"🟡","Low":"🟢"}[tier]
            lines.append(f"{icon} **{tier}**: {_pct(n,total)}  {bar}")
        return "\n".join(lines)

    # ── Count patients ─────────────────────────────────────────────────────────
    if "count" in matched and not any(x in matched for x in ["high_risk","medium_risk","low_risk"]):
        return (f"📊 **Total patients: {total}**\n\n"
                f"🔴 High: {_pct(rc.get('High',0),total)}\n"
                f"🟡 Medium: {_pct(rc.get('Medium',0),total)}\n"
                f"🟢 Low: {_pct(rc.get('Low',0),total)}")

    # ── High risk ─────────────────────────────────────────────────────────────
    if "high_risk" in matched:
        n = rc.get("High",0)
        adh = stats["adherence"]
        miss = stats["missed_doses"]
        lines = [f"🔴 **High-risk patients: {_pct(n,total)}**\n"]
        if adh and adh["by_risk"].get("High"):
            lines.append(f"  • Avg adherence: **{_fmt(adh['by_risk']['High'])}%** (overall avg: {_fmt(adh['mean'])}%)")
        if miss and miss["by_risk"].get("High"):
            lines.append(f"  • Avg missed doses/month: **{_fmt(miss['by_risk']['High'])}**")
        lines.append("\n💡 High-risk patients need immediate intervention — reminders, follow-up calls, case manager.")
        return "\n".join(lines)

    # ── Medium risk ────────────────────────────────────────────────────────────
    if "medium_risk" in matched:
        n = rc.get("Medium",0)
        adh = stats["adherence"]
        lines = [f"🟡 **Medium-risk patients: {_pct(n,total)}**\n"]
        if adh and adh["by_risk"].get("Medium"):
            lines.append(f"  • Avg adherence: **{_fmt(adh['by_risk']['Medium'])}%**")
        lines.append("\n💡 Targeted coaching can prevent progression to high risk. Monitor HbA1c every 3 months.")
        return "\n".join(lines)

    # ── Low risk ──────────────────────────────────────────────────────────────
    if "low_risk" in matched:
        n = rc.get("Low",0)
        adh = stats["adherence"]
        lines = [f"🟢 **Low-risk patients: {_pct(n,total)}**\n"]
        if adh and adh["by_risk"].get("Low"):
            lines.append(f"  • Avg adherence: **{_fmt(adh['by_risk']['Low'])}%**")
        lines.append("\n✅ Maintain current regimen. Annual HbA1c check is sufficient.")
        return "\n".join(lines)

    # ── Adherence ─────────────────────────────────────────────────────────────
    if "adherence" in matched:
        s = stats["adherence"]
        if not s: return "Adherence data not found in this dataset."
        lines = [f"💊 **Medication adherence:**\n",
                 f"  • Overall average: **{_fmt(s['mean'])}%** (median {_fmt(s['median'])}%)"]
        if s["by_risk"]:
            lines.append("\n  **By risk tier:**")
            for tier, icon in [("High","🔴"),("Medium","🟡"),("Low","🟢")]:
                v = s["by_risk"].get(tier)
                if v is not None:
                    lines.append(f"    {icon} {tier}: **{_fmt(v)}%**")
        return "\n".join(lines)

    # ── HbA1c ─────────────────────────────────────────────────────────────────
    if "hba1c" in matched:
        b = stats["hba1c_baseline"]; f = stats["hba1c_followup"]; d = stats["hba1c_delta"]
        lines = ["🩸 **HbA1c Summary:**\n"]
        if b: lines.append(f"  • Baseline avg: **{_fmt(b['mean'])}%**")
        if f: lines.append(f"  • Follow-up avg: **{_fmt(f['mean'])}%**")
        if d:
            direction = "📉 improved" if d["mean"] > 0 else "📈 worsened"
            lines.append(f"  • Mean change: **{_fmt(d['mean'],2)}** ({direction})")
            if d["by_risk"]:
                lines.append("\n  **Improvement by tier:**")
                for tier in ["High","Medium","Low"]:
                    v = d["by_risk"].get(tier)
                    if v is not None: lines.append(f"    • {tier}: Δ {_fmt(v,2)}")
        lines.append("\n💡 Clinical target: HbA1c < 7% (ADA guidelines).")
        return "\n".join(lines)

    # ── Missed doses ──────────────────────────────────────────────────────────
    if "missed_doses" in matched:
        s = stats["missed_doses"]
        if not s: return "Missed doses data not available."
        lines = [f"💊 **Missed doses per month:**\n",
                 f"  • Overall avg: **{_fmt(s['mean'])} doses/month**"]
        if s["by_risk"]:
            for tier in ["High","Medium","Low"]:
                v = s["by_risk"].get(tier)
                if v is not None: lines.append(f"    • {tier}: {_fmt(v)} doses/month")
        lines.append("\n💡 Even 1–2 missed doses/month can raise HbA1c significantly.")
        return "\n".join(lines)

    # ── Age ───────────────────────────────────────────────────────────────────
    if "age" in matched:
        s = stats["age"]
        if not s: return "Age data not available."
        lines = [f"👤 **Patient ages:**\n",
                 f"  • Mean: **{_fmt(s['mean'])} yrs**, Median: {_fmt(s['median'])} yrs",
                 f"  • Range: {_fmt(s['min'])} – {_fmt(s['max'])} yrs"]
        if s["by_risk"]:
            lines.append("\n  **Mean age by tier:**")
            for t in ["High","Medium","Low"]:
                v = s["by_risk"].get(t)
                if v: lines.append(f"    • {t}: {_fmt(v)} yrs")
        return "\n".join(lines)

    # ── BMI ───────────────────────────────────────────────────────────────────
    if "bmi" in matched:
        s = stats["bmi"]
        if not s: return "BMI data not available."
        def cat(b): return "underweight" if b<18.5 else "normal" if b<25 else "overweight" if b<30 else "obese"
        lines = [f"⚖️ **BMI:**\n  • Mean: **{_fmt(s['mean'])}** ({cat(s['mean'])})"]
        if s["by_risk"]:
            for t in ["High","Medium","Low"]:
                v = s["by_risk"].get(t)
                if v: lines.append(f"    • {t}: {_fmt(v)} ({cat(v)})")
        return "\n".join(lines)

    # ── Glucose ───────────────────────────────────────────────────────────────
    if "glucose" in matched:
        b = stats["glucose_b"]; f = stats["glucose_f"]
        lines = ["🩺 **Fasting Glucose (mg/dL):**\n"]
        if b: lines.append(f"  • Baseline avg: **{_fmt(b['mean'])} mg/dL**")
        if f:
            lines.append(f"  • Follow-up avg: **{_fmt(f['mean'])} mg/dL**")
            if b:
                d = f["mean"]-b["mean"]
                lines.append(f"  • Change: {_fmt(d)} mg/dL ({'improved' if d<0 else 'increased'})")
        lines.append("\n💡 Normal fasting glucose < 100 mg/dL. Diabetic range ≥ 126 mg/dL.")
        return "\n".join(lines)

    # ── Doctor visits ─────────────────────────────────────────────────────────
    if "doctor_visits" in matched:
        s = stats["doctor_visits"]
        if not s: return "Doctor visit data not available."
        lines = [f"🏥 **Doctor visits:**\n  • Overall avg: **{_fmt(s['mean'])} visits/period**"]
        if s["by_risk"]:
            for t in ["High","Medium","Low"]:
                v = s["by_risk"].get(t)
                if v: lines.append(f"    • {t}: {_fmt(v)} visits")
        return "\n".join(lines)

    # ── Improvement ───────────────────────────────────────────────────────────
    if "improvement" in matched:
        d = stats["hba1c_delta"]; adh = stats["adherence"]
        lines = ["📈 **Health Improvement Trends:**\n"]
        if d:
            lines.append(f"  • Mean HbA1c change: **{_fmt(d['mean'],2)}** ({'↓ improved' if d['mean']>0 else '↑ worsened'})")
        if adh and adh["by_risk"]:
            lo = adh["by_risk"].get("Low"); hi = adh["by_risk"].get("High")
            if lo and hi:
                lines.append(f"\n  • Low-risk patients average **{_fmt(lo)}%** adherence vs **{_fmt(hi)}%** for high-risk — a {_fmt(lo-hi)} point gap.")
        lines.append("\n💡 Moving from <60% to >80% adherence typically produces the greatest HbA1c improvement.")
        return "\n".join(lines)

    # ── Recommendations ───────────────────────────────────────────────────────
    if "recommendation" in matched:
        n_high = rc.get("High",0)
        adh = stats["adherence"]
        adh_hr = adh["by_risk"].get("High") if adh else None
        lines = ["📋 **Clinical Recommendations:**\n"]
        if total and n_high/total > 0.3:
            lines += ["  🔴 **High-risk cohort is large (>30%):**",
                      "  1. Automated SMS/app reminders for daily doses",
                      "  2. Bi-monthly HbA1c testing", "  3. Case manager for top 10% risk patients"]
        else:
            lines += ["  🟡 **Moderate risk profile:**",
                      "  1. Monthly adherence check-ins", "  2. Peer support group enrollment"]
        if adh_hr and adh_hr < 70:
            lines.append(f"\n  💊 High-risk avg adherence is only **{_fmt(adh_hr)}%** — blister packs and pill organizers can help.")
        lines.append("\n  📚 Source: ADA Standards of Care, WHO adherence guidelines.")
        return "\n".join(lines)

    # ── Features ──────────────────────────────────────────────────────────────
    if "features" in matched:
        tf = stats["top_features"]
        if tf:
            lines = ["🧠 **Top predictive features (Random Forest):**\n"]
            for i,f in enumerate(tf,1): lines.append(f"  {i}. {f.replace('_',' ')}")
            lines.append("\n💡 Focus clinical interventions on the top 2–3 features for maximum impact.")
            return "\n".join(lines)
        return "Feature importance not available. Retrain with `python main.py`."

    # ── Confidence ────────────────────────────────────────────────────────────
    if "confidence" in matched:
        s = stats["confidence"]
        if not s: return "Confidence scores not available."
        return (f"🎯 **Prediction confidence:**\n"
                f"  • Mean: **{_fmt(s['mean']*100)}%**,  Median: {_fmt(s['median']*100)}%\n"
                f"  • Range: {_fmt(s['min']*100)}% – {_fmt(s['max']*100)}%\n\n"
                f"💡 Predictions below 60% confidence should be reviewed manually.")

    # ── Model performance ─────────────────────────────────────────────────────
    if "model_perf" in matched:
        m = stats["metrics"]
        if not m: return "Model metrics not found. Run `python main.py` to train."
        lines = [f"📊 **Model Performance:**\n",
                 f"  • Test Accuracy: **{m.get('accuracy',0):.1%}**",
                 f"  • CV Accuracy: {m.get('cv_mean',0):.1%} ± {m.get('cv_std',0):.1%}"]
        if m.get("auc"): lines.append(f"  • ROC-AUC: **{m['auc']:.3f}**")
        lines += [f"  • Trained on: {m.get('n_train','N/A'):,} samples",
                  f"  • Tested on:  {m.get('n_test','N/A'):,} samples"]
        return "\n".join(lines)

    # ── Fallback — try to be helpful not blank ────────────────────────────────
    return (f"🤔 I understood your question but couldn't find matching data. "
            f"The dataset has **{total} patients** with risk split: "
            f"High={rc.get('High',0)}, Medium={rc.get('Medium',0)}, Low={rc.get('Low',0)}.\n\n"
            f"Try asking about: **adherence, HbA1c, missed doses, age, BMI, glucose, "
            f"risk breakdown, recommendations, or model accuracy**.")
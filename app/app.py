
# SleepHealth-AI - Streamlit application
# All model / metric / figure paths are anchored to the repository root so the app
# can be launched from any working directory:
#     streamlit run app/app.py
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

# shared human-readable feature naming (used by the app AND the training figures)
sys.path.insert(0, str(ROOT / "src"))
from feature_labels import display_feature, display_features  # noqa: E402

st.set_page_config(page_title="SleepHealth-AI", page_icon="🌙", layout="wide")

DISCLAIMER = ("⚠️ **Disclaimer:** educational / research prototype. Outputs are statistical "
              "**risk estimates for screening**, not medical diagnoses. Do not use for "
              "clinical decisions.")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_models():
    """Load every trained pipeline. Missing files degrade gracefully."""
    wanted = {
        "heart_disease": "heart_disease_model.pkl",
        "insomnia_risk": "insomnia_risk_model.pkl",
        "osa_risk": "osa_risk_model.pkl",
        "large_sleep_risk": "large_sleep_risk_model.pkl",
        "sleep_disorder": "sleep_disorder_model.pkl",
        "diabetes": "diabetes_model.pkl",
        "hypertension": "hypertension_model.pkl",
    }
    models, missing = {}, []
    for key, fname in wanted.items():
        path = MODELS_DIR / fname
        if path.exists():
            try:
                models[key] = joblib.load(path)
            except Exception as exc:  # pragma: no cover
                missing.append(f"{fname} ({exc})")
        else:
            missing.append(fname)
    return models, missing


@st.cache_data
def load_metrics():
    out = {}
    for name in ("metrics.json", "large_model_metrics.json", "calibration.json"):
        path = METRICS_DIR / name
        if path.exists():
            try:
                out[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                out[name] = None
    return out


@st.cache_resource
def load_calibrators():
    """Cross-fitted isotonic maps that turn raw model scores into honest risks."""
    cal_dir = MODELS_DIR / "calibration"
    out = {}
    for task in ("heart_disease", "insomnia_risk", "osa_risk"):
        path = cal_dir / f"{task}_isotonic.pkl"
        if path.exists():
            try:
                out[task] = joblib.load(path)
            except Exception:
                pass
    return out


def calibrated_risk(task: str, raw: float) -> float:
    """Return the calibrated probability for a single raw model score."""
    cal = CALIBRATORS.get(task)
    if cal is None:
        return raw
    try:
        return float(cal.predict([raw])[0])
    except Exception:
        return raw


MODELS, MISSING_MODELS = load_models()
CALIBRATORS = load_calibrators()
ALL_METRICS = load_metrics()
NHANES_METRICS = ALL_METRICS.get("metrics.json") or {}
LARGE_METRICS = ALL_METRICS.get("large_model_metrics.json") or {}
TASK_META = NHANES_METRICS.get("tasks", {})
LEGACY_DEMO = NHANES_METRICS.get("legacy_demo", {})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def gauge(probability: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": title, "font": {"size": 17, "color": "gray"}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "black", "thickness": 0.2},
               "steps": [{"range": [0, 20], "color": "#00CC96"},
                         {"range": [20, 50], "color": "#FFA15A"},
                         {"range": [50, 100], "color": "#EF553B"}],
               "threshold": {"line": {"color": "black", "width": 3},
                             "thickness": 0.75, "value": probability}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=45, b=10))
    return fig


def risk_word(p: float) -> str:
    if p < 20:
        return "Low estimated risk band"
    if p < 50:
        return "Moderate estimated risk band"
    return "Elevated estimated risk band"


def shap_contributions(pipeline, row: pd.DataFrame, background: pd.DataFrame) -> np.ndarray:
    """Local SHAP contributions for a single row, tree or linear models."""
    feats = list(row.columns)
    transformed = np.asarray(pipeline[:-1].transform(row))
    clf = pipeline.named_steps["classifier"]
    kind = type(clf).__name__
    if kind in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(clf)
        vals = explainer.shap_values(transformed)
    else:  # LogisticRegression -> use training rows as the SHAP background
        bg = np.asarray(pipeline[:-1].transform(background[feats]))
        explainer = shap.LinearExplainer(clf, bg)
        vals = explainer.shap_values(transformed)
    if isinstance(vals, list):
        vals = vals[1]
    elif getattr(vals, "ndim", 0) == 3:
        vals = vals[:, :, 1]
    return np.asarray(vals)[0], feats


def shap_bar(values: np.ndarray, names: list, title: str) -> go.Figure:
    order = np.argsort(values)
    v, n = values[order], [names[i] for i in order]
    colors = ["#EF553B" if x > 0 else "#00CC96" for x in v]
    fig = go.Figure(go.Bar(x=v, y=n, orientation="h", marker_color=colors,
                           text=[f"{x:+.3f}" for x in v], textposition="auto"))
    fig.update_layout(title=title, xaxis_title="SHAP value (log-odds contribution)",
                      height=max(300, 34 * len(n)), margin=dict(l=20, r=20, t=45, b=20))
    return fig


def show_task_metrics(task_key: str):
    meta = TASK_META.get(task_key)
    if not meta:
        st.info("Metrics file not found for this task.")
        return
    best = meta.get("best_model", "?")
    m = meta.get("metrics_model_b", {})
    abl = meta.get("ablation", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants (n)", f"{meta.get('n', 0):,}")
    c2.metric("Positive class", f"{meta.get('prevalence', 0):.1%}")
    c3.metric("Best algorithm", best)
    c4.metric("ROC-AUC (95% CI)",
              f"{m.get('roc_auc', 0):.3f}",
              f"{m.get('roc_auc_ci95', [0, 0])[0]:.3f} – {m.get('roc_auc_ci95', [0, 1])[1]:.3f}")
    if abl:
        st.caption(
            f"Ablation: demographics ROC-AUC **{abl.get('model_a_demographics_roc_auc')}**  →  "
            f"+ sleep features **{abl.get('model_b_plus_sleep_roc_auc')}**  "
            f"(Δ = **{abl.get('delta'):+.4f}**; PR-AUC Δ = {abl.get('delta_pr_auc'):+.4f})  •  "
            f"PR-AUC {m.get('pr_auc')} • F1 {m.get('f1')} • Recall {m.get('recall')} • "
            f"Precision {m.get('precision')} • Brier {m.get('brier')}")
        st.caption(f"Label definition: {meta.get('definition', '')}")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## 🌙 SleepHealth-AI")
page = st.sidebar.radio("Navigation", [
    "📋 Executive Overview",
    "🫀 Sleep-Disorder & Heart Risk (NHANES)",
    "📈 Large-Scale Sleep Risk (100k)",
    "🧠 Sleep Pattern Demo (educational)",
    "🫀 Cardiometabolic Dashboard",
    "⚙️ Performance & Methodology",
])
st.sidebar.markdown("---")
st.sidebar.info("🎓 **MCA / Master's Thesis Project**\n\nSleep-pattern analysis and "
                "multi-disease risk estimation with explainable ML.")
st.sidebar.warning(DISCLAIMER)
if MISSING_MODELS:
    st.sidebar.error("Missing model files: " + ", ".join(MISSING_MODELS))


# --------------------------------------------------------------------------- #
# 1. Overview
# --------------------------------------------------------------------------- #
if page == "📋 Executive Overview":
    st.title("SleepHealth-AI — Sleep Pattern Analysis & Disease Risk Estimation")
    st.markdown("### Research prototype — machine learning with explainability")

    st.markdown("""
This system analyses **sleep patterns** and estimates the risk of **insomnia**, **obstructive
sleep apnea**, **heart disease**, **hypertension** and **diabetes** from demographic,
anthropometric and sleep-behavioural features. Every prediction is a *screening risk estimate*
and can be explained with SHAP.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📦 Datasets")
        st.markdown("""
| Dataset | Rows | Type | Year |
|---|---|---|---|
| CDC **NHANES 2017-2018** (Cycle J) | **5,088** adults | Real national survey | 2017–18 (rel. 2020) |
| Sleep Health & Daily Performance | **100,000** | Synthetic benchmark | 2026 |
| Kaggle sleep/lifestyle | 374 (109 unique) | Educational | — |
        """)
    with col2:
        st.markdown("#### 🎯 Prediction targets")
        st.markdown("""
- **Insomnia risk** — NHANES SLQ050 questionnaire label
- **Sleep-apnea risk** — symptom-based (snoring / witnessed pauses)
- **Heart disease** — CHD, CHF, angina, heart attack (MCQ_J)
- **Hypertension & Diabetes** — ablation study with sleep features
- **Sleep-disorder severity** (Healthy→Severe) on 100k records
        """)

    if NHANES_METRICS.get("data"):
        st.markdown("#### 🏆 Headline results (repeated stratified CV, 5×3)")
        rows = []
        for k, t in TASK_META.items():
            a = t.get("ablation", {})
            m = t.get("metrics_model_b", {})
            rows.append({
                "Task": t.get("label", k), "n": t.get("n"),
                "Model A (demographics)": a.get("model_a_demographics_roc_auc"),
                "Model B (+ sleep)": a.get("model_b_plus_sleep_roc_auc"),
                "Δ (value of sleep)": a.get("delta"),
                "ROC-AUC 95% CI": f"{m.get('roc_auc_ci95', ['', ''])[0]} – {m.get('roc_auc_ci95', ['', ''])[1]}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Δ = improvement in ROC-AUC obtained by appending sleep features "
                   "to the demographic baseline (the project's central ablation study).")

    if LARGE_METRICS:
        st.info(f"**Large-scale model:** {LARGE_METRICS.get('n_records', 0):,} records — "
                f"CV accuracy {LARGE_METRICS.get('cv_accuracy', {}).get('mean', 0):.3f} "
                f"(95% CI {LARGE_METRICS.get('cv_accuracy', {}).get('ci95', ['', ''])[0]}–"
                f"{LARGE_METRICS.get('cv_accuracy', {}).get('ci95', ['', ''])[1]}), "
                f"balanced accuracy {LARGE_METRICS.get('cv_balanced_accuracy', {}).get('mean', 0):.3f}, "
                f"macro-F1 {LARGE_METRICS.get('cv_macro_f1', {}).get('mean', 0):.3f}. "
                f"*Source data is synthetic benchmark data.*")
    st.warning(DISCLAIMER)


# --------------------------------------------------------------------------- #
# 2. NHANES sleep-disorder & heart risk
# --------------------------------------------------------------------------- #
elif page == "🫀 Sleep-Disorder & Heart Risk (NHANES)":
    st.title("Sleep-Disorder & Heart Disease Risk")
    st.markdown("Estimate **insomnia risk**, **sleep-apnea risk** and **heart-disease risk** "
                "from a demographic + sleep profile, using models trained on "
                "**CDC NHANES 2017-2018 (n≈5,000 adults)**.")

    required = ["heart_disease", "insomnia_risk", "osa_risk"]
    if not all(k in MODELS for k in required):
        st.error("Model files missing — run `python src/train_models.py` first.")
        st.stop()

    with st.form("nhanes_form"):
        st.markdown("#### 👤 Profile")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", 20, 85, 50)
        female = c2.selectbox("Sex", ["Male", "Female"]) == "Female"
        bmi = c3.number_input("BMI (kg/m²)", 12.0, 60.0, 28.0, 0.1)
        waist = c1.number_input("Waist circumference (cm)", 50.0, 190.0, 95.0, 0.5)
        sleep_h = c2.number_input("Weekday sleep (hours)", 2.0, 14.0, 7.0, 0.5)
        weekend_h = c3.number_input("Weekend sleep (hours)", 2.0, 14.0, 7.5, 0.5)

        st.markdown("#### 🌬️ Sleep symptoms (past 12 months / past month)")
        d1, d2, d3 = st.columns(3)
        snore = d1.select_slider("Snoring frequency",
                                 options=["Never", "Rarely (1-2/wk)", "Occasionally (3-4/wk)",
                                          "Frequently (5+/wk)"], value="Never")
        apnea = d2.select_slider("Witnessed breathing pauses / gasping",
                                 options=["Never", "Rarely (1-2/wk)", "Occasionally (3-4/wk)",
                                          "Frequently (5+/wk)"], value="Never")
        sleepy = d3.select_slider("Daytime sleepiness (past month)",
                                  options=["Never", "Rarely (1×/mo)", "Sometimes (2-4×/mo)",
                                           "Often (5-15×/mo)", "Almost always"], value="Never")
        trouble = d1.select_slider("Ever told a doctor about trouble sleeping?",
                                   options=["No", "Yes"], value="No")
        submitted = st.form_submit_button("🚀 Estimate risks", type="primary",
                                          width="stretch")

    freq_map = {"Never": 0, "Rarely (1-2/wk)": 1, "Occasionally (3-4/wk)": 2,
                "Frequently (5+/wk)": 3}
    sleepy_map = {"Never": 0, "Rarely (1×/mo)": 1, "Sometimes (2-4×/mo)": 2,
                  "Often (5-15×/mo)": 3, "Almost always": 4}

    if submitted:
        inputs = {
            "age": age, "gender_female": int(female), "bmi": bmi,
            "waist_circumference": waist,
            "sleep_duration_hrs": sleep_h, "weekend_sleep_hrs": weekend_h,
            "social_jetlag_hrs": round(weekend_h - sleep_h, 2),
            "snore_freq": freq_map[snore], "witnessed_apnea_freq": freq_map[apnea],
            "daytime_sleepiness": sleepy_map[sleepy],
            "trouble_sleeping": int(trouble == "Yes"),
        }
        # Remember the profile. Without this, any widget click outside the form
        # (e.g. the SHAP radio below) re-runs the script, `submitted` becomes
        # False again and every result disappears.
        st.session_state["nhanes_inputs"] = inputs

    inputs = st.session_state.get("nhanes_inputs")

    if inputs:
        df_in = pd.DataFrame([inputs])
        names = {"heart_disease": "Heart disease", "insomnia_risk": "Insomnia",
                 "osa_risk": "Sleep apnea"}

        st.markdown("#### 📊 Estimated screening risks")
        probs = {}
        cols = st.columns(3)
        for i, key in enumerate(required):
            meta = TASK_META.get(key, {})
            raw = float(MODELS[key].predict_proba(df_in[meta.get("features_model_b",
                                                               list(inputs))])[:, 1][0])
            p = calibrated_risk(key, raw) * 100          # honest, calibrated risk
            probs[key] = p
            with cols[i]:
                st.plotly_chart(gauge(p, f"{names[key]} risk"), width="stretch")
                st.caption(f"Model: {meta.get('best_model', '?')} • "
                           f"ROC-AUC {meta.get('metrics_model_b', {}).get('roc_auc', 0):.3f} • "
                           f"{risk_word(p)}")

        if probs.get("osa_risk", 0) >= 50:
            st.warning("🔴 Elevated **sleep-apnea risk** — consider discussing snoring, "
                       "witnessed breathing pauses and daytime sleepiness with a physician. "
                       "Confirmation requires a sleep study (polysomnography).")
        if probs.get("heart_disease", 0) >= 50:
            st.warning("🔴 Elevated **heart-disease risk** — this is a statistical estimate "
                       "from questionnaire data, not a cardiac evaluation.")
        if max(probs.values()) < 20:
            st.success("✅ Low estimated bands across all three screens.")

        # ---- local explanation ------------------------------------------------
        st.markdown("---")
        st.markdown("#### 🔍 Why? (local SHAP explanation)")
        which = st.radio("Explain the prediction for:", list(names.values()), horizontal=True)
        inv = {v: k for k, v in names.items()}
        key = inv[which]
        meta = TASK_META.get(key, {})
        feats = meta.get("features_model_b", list(inputs))
        row = df_in[feats]
        background = pd.read_csv(PROCESSED / "nhanes_sleep_health.csv")[feats]
        try:
            vals, fnames = shap_contributions(MODELS[key], row, background)
            # Raw column names (gender_female, snore_freq, ...) are code identifiers,
            # not readable labels. Show what the user actually chose, e.g. "Sex: Male".
            labels = display_features(fnames, row)
            st.plotly_chart(shap_bar(vals, labels,
                                     f"{which} — contribution of each feature"),
                            width="stretch")
            st.caption("Red pushes the estimate **up**, green pushes it **down**. "
                       "SHAP values are on the log-odds scale (a value of +0.69 roughly "
                       "doubles the estimated odds). The bracketed values are the inputs "
                       "**you selected** above.")
        except Exception as exc:
            st.info(f"SHAP unavailable: {exc}")

        st.markdown("#### 📈 Model performance for this task")
        show_task_metrics(key)

        cal = (ALL_METRICS.get("calibration.json") or {}).get(key)
        if cal:
            st.markdown("#### 🎛️ Probability calibration")
            c1, c2, c3 = st.columns(3)
            c1.metric("Brier score ↓", f"{cal['brier_before']:.4f}",
                      f"→ {cal['brier_after']:.4f}")
            c2.metric("ECE (equal-freq) ↓", f"{cal['ece_equalfreq_before']:.4f}",
                      f"→ {cal['ece_equalfreq_after']:.4f}")
            c3.metric("ROC-AUC", f"{cal['roc_auc_before']:.4f}",
                      f"→ {cal['roc_auc_after']:.4f}")
            st.caption(
                "Gauges above show **calibrated** risks. The raw model over-predicted "
                "severely (e.g. a raw 0.84 corresponded to a true rate near 0.34) because "
                "`class_weight='balanced'` shifts scores away from the ~9–32% base rates. "
                "A cross-fitted isotonic map fixes the level: Brier and ECE drop sharply "
                "while ROC-AUC stays essentially unchanged (calibration cannot change "
                "ranking). Reliability tables are in "
                "`results/metrics/calibration.json`.")

    else:
        st.info("Fill the profile and press **Estimate risks**.")
        for key in required:
            meta = TASK_META.get(key, {})
            if meta:
                st.markdown(f"**{meta.get('label', key)}** — {meta.get('definition', '')}")
        fig = FIGURES_DIR / "roc_heart_disease.png"
        if fig.exists():
            st.image(str(fig), caption="ROC curves — heart-disease model comparison")


# --------------------------------------------------------------------------- #
# 3. Large-scale sleep risk
# --------------------------------------------------------------------------- #
elif page == "📈 Large-Scale Sleep Risk (100k)":
    st.title("Large-Scale Sleep-Disorder Severity")
    st.markdown("XGBoost model trained on **100,000 records** (28 features) to estimate "
                "sleep-disorder severity: *Healthy → Mild → Moderate → Severe*.")

    if "large_sleep_risk" not in MODELS:
        st.error("Model file missing — run `python src/train_large_model.py` first.")
        st.stop()

    CLASSES = ["Healthy", "Mild", "Moderate", "Severe"]
    COLUMNS = ["age", "gender", "occupation", "bmi", "country", "sleep_duration_hrs",
               "sleep_quality_score", "rem_percentage", "deep_sleep_percentage",
               "sleep_latency_mins", "wake_episodes_per_night", "caffeine_mg_before_bed",
               "alcohol_units_before_bed", "screen_time_before_bed_mins", "exercise_day",
               "steps_that_day", "nap_duration_mins", "stress_score",
               "work_hours_that_day", "chronotype", "mental_health_condition",
               "heart_rate_resting_bpm", "sleep_aid_used", "shift_work",
               "room_temperature_celsius", "weekend_sleep_diff_hrs", "season", "day_type"]
    DEFAULTS = {"age": 33.0, "bmi": 26.3, "sleep_duration_hrs": 6.36,
                "sleep_quality_score": 4.9, "stress_score": 5.8, "rem_percentage": 20.3,
                "deep_sleep_percentage": 20.3, "sleep_latency_mins": 19.0,
                "wake_episodes_per_night": 3.0, "caffeine_mg_before_bed": 0.0,
                "alcohol_units_before_bed": 0.0, "screen_time_before_bed_mins": 51.0,
                "steps_that_day": 7442.0, "nap_duration_mins": 0.0,
                "work_hours_that_day": 7.4, "heart_rate_resting_bpm": 67.0,
                "room_temperature_celsius": 20.5, "weekend_sleep_diff_hrs": 1.2}
    GENDERS = ["Female", "Male", "Other"]
    OCCUPATIONS = ["Doctor", "Driver", "Freelancer", "Homemaker", "Lawyer", "Manager",
                   "Nurse", "Retired", "Sales", "Software Engineer", "Student", "Teacher"]
    COUNTRIES = ["Australia", "Brazil", "Canada", "France", "Germany", "India", "Italy",
                 "Japan", "Mexico", "Netherlands", "South Korea", "Spain", "Sweden",
                 "UK", "USA"]
    CHRONO = ["Evening", "Morning", "Neutral"]
    MENTAL = ["Anxiety", "Both", "Depression", "Healthy"]
    SEASONS = ["Autumn", "Spring", "Summer", "Winter"]
    DAYTYPE = ["Weekday", "Weekend"]

    with st.form("large_form"):
        st.markdown("#### 👤 Daily profile (unlisted features use dataset medians)")
        a1, a2, a3, a4 = st.columns(4)
        age = a1.number_input("Age", 18, 69, 33)
        gender = a2.selectbox("Gender", GENDERS)
        occupation = a3.selectbox("Occupation", OCCUPATIONS)
        country = a4.selectbox("Country", COUNTRIES)
        b1, b2, b3, b4 = st.columns(4)
        bmi = b1.number_input("BMI", 16.0, 45.0, 26.3, 0.1)
        dur = b2.number_input("Sleep duration (h)", 3.0, 10.5, 6.4, 0.1)
        qual = b3.slider("Sleep quality (1-10)", 1, 10, 5)
        stress = b4.slider("Stress score (1-10)", 1, 10, 6)
        e1, e2, e3, e4 = st.columns(4)
        latency = e1.number_input("Sleep latency (min)", 1, 58, 19)
        wakes = e2.number_input("Wake episodes / night", 0, 8, 3)
        chrono = e3.selectbox("Chronotype", CHRONO)
        mental = e4.selectbox("Mental-health condition", MENTAL)
        f1, f2, f3, f4 = st.columns(4)
        shift = f1.selectbox("Shift work", ["No", "Yes"]) == "Yes"
        aid = f2.selectbox("Sleep aid used", ["No", "Yes"]) == "Yes"
        screen = f3.number_input("Screen time before bed (min)", 2, 180, 51)
        hr = f4.number_input("Resting heart rate (bpm)", 45, 99, 67)
        go_btn = st.form_submit_button("🚀 Estimate severity", type="primary",
                                       width="stretch")

    if go_btn:
        row = {c: DEFAULTS.get(c, 0) for c in COLUMNS}
        row.update({"age": age, "gender": gender, "occupation": occupation,
                    "country": country, "bmi": bmi, "sleep_duration_hrs": dur,
                    "sleep_quality_score": float(qual), "stress_score": float(stress),
                    "sleep_latency_mins": float(latency),
                    "wake_episodes_per_night": float(wakes), "chronotype": chrono,
                    "mental_health_condition": mental, "shift_work": int(shift),
                    "sleep_aid_used": int(aid),
                    "screen_time_before_bed_mins": float(screen),
                    "heart_rate_resting_bpm": float(hr), "exercise_day": 0,
                    "season": "Spring", "day_type": "Weekday"})
        df_in = pd.DataFrame([row])[COLUMNS]
        proba = MODELS["large_sleep_risk"].predict_proba(df_in)[0] * 100
        pred = int(np.argmax(proba))

        st.markdown("#### 📊 Severity distribution")
        fig = go.Figure(go.Bar(x=[f"{p:.1f}%" for p in proba], y=CLASSES,
                               orientation="h",
                               marker_color=["#00CC96", "#FFA15A", "#FF7A5A", "#EF553B"],
                               text=[f"{p:.1f}%" for p in proba], textposition="auto"))
        fig.update_layout(xaxis=dict(range=[0, 100]),
                          xaxis_title="Estimated probability (%)", height=260,
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"### Most likely band: **{CLASSES[pred]}**  "
                    f"(confidence {proba[pred]:.1f}%)")
        if pred >= 2:
            st.warning("Elevated modelled sleep-disorder severity — review sleep duration, "
                       "latency, wake episodes, stress and caffeine. *This is a synthetic-data "
                       "benchmark model, not a clinical assessment.*")
        elif pred == 1:
            st.info("Mild modelled severity — modest disruption patterns detected.")

        if LARGE_METRICS:
            cv = LARGE_METRICS.get("cv_accuracy", {})
            ba = LARGE_METRICS.get("cv_balanced_accuracy", {})
            f1 = LARGE_METRICS.get("cv_macro_f1", {})
            st.markdown("#### 📈 Model performance (5×2 repeated CV, 100,000 records)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{cv.get('mean', 0):.3f}",
                      f"±{cv.get('std', 0):.3f}")
            m2.metric("Balanced accuracy", f"{ba.get('mean', 0):.3f}",
                      f"±{ba.get('std', 0):.3f}")
            m3.metric("Macro-F1", f"{f1.get('mean', 0):.3f}", f"±{f1.get('std', 0):.3f}")
            m4.metric("Hold-out rows", f"{LARGE_METRICS.get('n_records', 0) * 0.2:,.0f}")
            st.caption("Class imbalance (Healthy 54% / Mild 33% / Moderate 8% / Severe 4%) "
                       "is why balanced accuracy and macro-F1 are reported alongside accuracy.")
            cm = FIGURES_DIR / "large_confusion_matrix.png"
            if cm.exists():
                st.image(str(cm), caption="Hold-out confusion matrix (normalised)")
    else:
        st.info("Fill the daily profile and press **Estimate severity**.")
        cd = FIGURES_DIR / "large_class_distribution.png"
        if cd.exists():
            st.image(str(cd), caption="Class distribution of the 100,000-record dataset")
    st.warning(DISCLAIMER)


# --------------------------------------------------------------------------- #
# 4. Educational Kaggle demo
# --------------------------------------------------------------------------- #
elif page == "🧠 Sleep Pattern Demo (educational)":
    st.title("Sleep Pattern Classifier — Educational Demo")
    st.markdown("3-class pattern classifier (**None / Insomnia / Sleep Apnea**) trained on a "
                "small public Kaggle dataset (374 rows, only **109 unique feature patterns**). "
                "Use it to explore the interface — **not** for any health conclusion. "
                "The validated models live on the NHANES page.")

    if "sleep_disorder" not in MODELS:
        st.error("Model file missing — run `notebooks/02_sleep_disorder_model.ipynb` first.")
        st.stop()

    OCCUPATIONS = ["Accountant", "Doctor", "Engineer", "Lawyer", "Manager", "Nurse",
                   "Sales Representative", "Salesperson", "Scientist",
                   "Software Engineer", "Teacher"]
    with st.form("kaggle_form"):
        st.markdown("#### 👤 Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", 18, 70, 30)
            gender = st.selectbox("Sex", ["Male", "Female"])
            occ = st.selectbox("Occupation", OCCUPATIONS)
            bmi_cat = st.selectbox("BMI category", ["Normal", "Overweight", "Obese"])
        with col2:
            sleep_dur = st.number_input("Sleep duration (h)", 4.0, 10.0, 7.0, 0.1)
            qual = st.slider("Sleep quality (1-10)", 1, 10, 6)
            phys = st.number_input("Physical activity (min/day)", 0, 120, 60)
            stress = st.slider("Stress level (1-10)", 1, 10, 5)
        with col3:
            hr = st.number_input("Resting heart rate (bpm)", 50, 110, 70)
            steps = st.number_input("Daily steps", 1000, 25000, 6000, 100)
            sys_bp = st.number_input("Systolic BP (mmHg)", 90, 160, 120)
            dia_bp = st.number_input("Diastolic BP (mmHg)", 55, 110, 80)
        run = st.form_submit_button("🚀 Classify pattern", type="primary",
                                    width="stretch")

    if run:
        df_in = pd.DataFrame([[gender, occ, bmi_cat, age, sleep_dur, qual, phys, stress,
                               hr, steps, sys_bp, dia_bp]],
                             columns=["Gender", "Occupation", "BMI Category", "Age",
                                      "Sleep Duration", "Quality of Sleep",
                                      "Physical Activity Level", "Stress Level",
                                      "Heart Rate", "Daily Steps", "Systolic_BP",
                                      "Diastolic_BP"])
        probs = MODELS["sleep_disorder"].predict_proba(df_in)[0] * 100
        labels = ["No pattern detected", "Insomnia-like pattern", "Apnea-like pattern"]
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown("### Predicted pattern")
            st.markdown(f"### **{labels[int(np.argmax(probs))]}**")
            st.caption(f"Confidence {max(probs):.1f}%")
        with r2:
            fig = go.Figure(go.Bar(x=probs, y=labels, orientation="h",
                                   marker_color=["#00CC96", "#FFA15A", "#EF553B"],
                                   text=[f"{p:.1f}%" for p in probs],
                                   textposition="auto"))
            fig.update_layout(xaxis=dict(range=[0, 100]), height=230,
                              margin=dict(l=20, r=20, t=25, b=20))
            st.plotly_chart(fig, width="stretch")
        st.warning(f"**Educational dataset with synthetic, undocumented labels.** Honest "
                   f"group-aware cross-validation (identical patterns never split across "
                   f"train/test): accuracy "
                   f"**{LEGACY_DEMO.get('groupaware_cv_accuracy', {}).get('mean', 0):.3f}** "
                   f"(95% CI {LEGACY_DEMO.get('groupaware_cv_accuracy', {}).get('ci95', [0, 0])[0]:.3f}–"
                   f"{LEGACY_DEMO.get('groupaware_cv_accuracy', {}).get('ci95', [0, 1])[1]:.3f}) "
                   f"on {LEGACY_DEMO.get('n_unique_feature_patterns', 0)} unique patterns — "
                   f"versus **0.933** from the original single 80/20 split, which is "
                   f"optimistic. Use the NHANES page for the validated models.")


# --------------------------------------------------------------------------- #
# 5. Cardiometabolic dashboard (original ablation study)
# --------------------------------------------------------------------------- #
elif page == "🫀 Cardiometabolic Dashboard":
    st.title("Cardiometabolic Risk Dashboard")
    st.markdown("Diabetes & hypertension risk from demographics + sleep features "
                "(NHANES ablation study, Model B).")

    if "diabetes" not in MODELS or "hypertension" not in MODELS:
        st.error("Model files missing — run `notebooks/04_disease_models.ipynb` first.")
        st.stop()

    with st.form("cardio_form"):
        st.markdown("#### 👤 Profile")
        col1, col2, col3, col4 = st.columns(4)
        age = col1.number_input("Age (years)", 20, 100, 50)
        gender = col2.selectbox("Sex", ["Male", "Female"])
        bmi = col3.number_input("BMI (kg/m²)", 10.0, 80.0, 28.0, 0.1)
        waist = col4.number_input("Waist circumference (cm)", 50.0, 200.0, 95.0, 0.5)
        col5, col6, _, _ = st.columns(4)
        sleep_hrs = col5.number_input("Sleep (hours)", 2.0, 14.0, 7.0, 0.5)
        trouble = col6.selectbox("Trouble sleeping?", ["No", "Yes"])
        submitted = st.form_submit_button("🚀 Run predictive pipeline", type="primary")

    if submitted:
        df_in = pd.DataFrame({
            "age": [age], "gender_female": [1 if gender == "Female" else 0],
            "bmi": [bmi], "waist_circumference": [waist],
            "sleep_duration_hrs": [sleep_hrs],
            "trouble_sleeping": [1 if trouble == "Yes" else 0]})
        p_diab = float(MODELS["diabetes"].predict_proba(df_in)[0][1]) * 100
        p_htn = float(MODELS["hypertension"].predict_proba(df_in)[0][1]) * 100

        g1, g2 = st.columns(2)
        g1.plotly_chart(gauge(p_diab, "Diabetes risk"), width="stretch")
        g2.plotly_chart(gauge(p_htn, "Hypertension risk"), width="stretch")
        st.caption("These gauges show model-estimated probabilities. They are **not** "
                   "calibrated clinical risks — see *Performance & Methodology*.")

        st.markdown("---")
        st.markdown("#### 🔍 Local explanation (hypertension model)")
        feats = ["age", "gender_female", "bmi", "waist_circumference",
                 "sleep_duration_hrs", "trouble_sleeping"]
        try:
            vals, fnames = shap_contributions(MODELS["hypertension"], df_in[feats],
                                              df_in[feats])
            st.plotly_chart(shap_bar(vals, display_features(fnames, df_in),
                                     "Feature contribution (hypertension)"),
                            width="stretch")
            st.caption("Red raises the estimate, green lowers it (log-odds scale). "
                       "Bracketed values are the inputs you selected.")
        except Exception as exc:
            st.info(f"SHAP unavailable: {exc}")


# --------------------------------------------------------------------------- #
# 6. Performance & methodology (all numbers read from results/metrics)
# --------------------------------------------------------------------------- #
elif page == "⚙️ Performance & Methodology":
    st.title("Performance & Methodology")
    st.caption("Every number on this page is loaded from `results/metrics/*.json` — "
               "regenerated by `src/train_models.py` and `src/train_large_model.py`.")

    if not TASK_META:
        st.error("`results/metrics/metrics.json` not found — run `python src/train_models.py`.")
        st.stop()

    st.markdown("### NHANES models — repeated stratified CV (5 folds × 3 repeats)")
    rows = []
    for key, t in TASK_META.items():
        m = t.get("metrics_model_b", {})
        a = t.get("ablation", {})
        rows.append({
            "Task": t.get("label"),
            "n": t.get("n"),
            "Positive %": f"{t.get('prevalence', 0):.1%}",
            "Best model": t.get("best_model"),
            "ROC-AUC": m.get("roc_auc"),
            "95% CI": f"{m.get('roc_auc_ci95', ['', ''])[0]} – {m.get('roc_auc_ci95', ['', ''])[1]}",
            "PR-AUC": m.get("pr_auc"),
            "F1": m.get("f1"),
            "Recall": m.get("recall"),
            "Precision": m.get("precision"),
            "Brier ↓": m.get("brier"),
            "Δ from sleep": a.get("delta"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("### Algorithm comparison (Model B, ROC-AUC)")
    comp = []
    for key, t in TASK_META.items():
        row = {"Task": key}
        for alg, splits in t.get("cv", {}).items():
            row[alg] = splits.get("model_b", {}).get("roc_auc")
        comp.append(row)
    st.dataframe(pd.DataFrame(comp), width="stretch", hide_index=True)

    if LARGE_METRICS:
        st.markdown("### Large-scale model (100,000 records)")
        cv = LARGE_METRICS.get("cv_accuracy", {})
        st.markdown(
            f"- Accuracy **{cv.get('mean', 0):.4f}** (95% CI {cv.get('ci95', ['', ''])[0]} – "
            f"{cv.get('ci95', ['', ''])[1]}), balanced accuracy "
            f"**{LARGE_METRICS.get('cv_balanced_accuracy', {}).get('mean', 0):.4f}**, "
            f"macro-F1 **{LARGE_METRICS.get('cv_macro_f1', {}).get('mean', 0):.4f}**")
        st.caption(LARGE_METRICS.get("dataset_note", ""))

    st.markdown("### Figures")
    tabs = st.tabs(["ROC curves", "SHAP", "Large model"])
    with tabs[0]:
        for f in ["roc_heart_disease.png", "roc_insomnia_risk.png", "roc_osa_risk.png"]:
            p = FIGURES_DIR / f
            if p.exists():
                st.image(str(p), caption=f)
    with tabs[1]:
        for f in ["shap_heart_disease.png", "shap_insomnia_risk.png", "shap_osa_risk.png"]:
            p = FIGURES_DIR / f
            if p.exists():
                st.image(str(p), caption=f)
        st.caption("SHAP summary over the training cohort (log-odds scale).")
    with tabs[2]:
        for f in ["large_confusion_matrix.png", "large_class_distribution.png"]:
            p = FIGURES_DIR / f
            if p.exists():
                st.image(str(p), caption=f)

    st.markdown("### Methodology & honest limitations")
    st.markdown("""
- **Design:** repeated stratified cross-validation with 95% confidence intervals (no single
  lucky split); the ablation Δ compares identical folds for Model A vs Model B.
- **Leakage control:** label-defining questionnaire items are excluded from that label's
  features; sibling target columns and IDs are excluded from the 100k model.
- **Not diagnoses:** NHANES has no polysomnography (no AHI) and no clinical adjudication —
  insomnia/apnea outputs are **screening-risk estimates from questionnaire items**
  (SLQ050; SLQ030/SLQ040). Heart/diabetes/hypertension labels are self-reported.
- **Data vintage:** NHANES 2017-2018; the 100k benchmark is synthetic.
- **Remaining weaknesses:** cross-sectional data (association ≠ causation), unweighted
  NHANES sample, no external validation cohort, calibration not yet applied to gauges.
    """)
    st.warning(DISCLAIMER)

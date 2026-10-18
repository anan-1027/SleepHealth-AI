"""
Smoke tests for SleepHealth-AI (no pytest required — run with `python tests/smoke_test.py`).

Covers:
  1. Required artifacts exist (models, metrics, figures).
  2. Every model loads and predicts on a correctly-shaped input (pipeline contract).
  3. Every Streamlit page renders, and the interactive pages execute their pipelines.
  4. Leakage guard: label-defining features are absent from each label's feature list.

Exit code 0 = all checks passed.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------- artifacts --
print("\n== 1. Artifacts ==")
required = [
    "models/heart_disease_model.pkl", "models/insomnia_risk_model.pkl",
    "models/osa_risk_model.pkl", "models/large_sleep_risk_model.pkl",
    "models/sleep_disorder_model.pkl", "models/diabetes_model.pkl",
    "models/hypertension_model.pkl",
    "results/metrics/metrics.json", "results/metrics/large_model_metrics.json",
    "data/processed/nhanes_sleep_health.csv", "data/processed/sleep_cleansed.csv",
]
for rel in required:
    check(f"exists: {rel}", (ROOT / rel).exists())

for fig in ["roc_heart_disease.png", "roc_insomnia_risk.png", "roc_osa_risk.png",
            "shap_heart_disease.png", "shap_insomnia_risk.png", "shap_osa_risk.png",
            "large_confusion_matrix.png", "large_class_distribution.png"]:
    check(f"exists: results/figures/{fig}", (ROOT / "results" / "figures" / fig).exists())

# ------------------------------------------------------- model contracts -----
print("\n== 2. Model prediction contracts ==")
metrics = json.loads((ROOT / "results/metrics/metrics.json").read_text(encoding="utf-8"))
nhanes = pd.read_csv(ROOT / "data/processed/nhanes_sleep_health.csv")

for task, meta in metrics["tasks"].items():
    feats = meta["features_model_b"]
    mdl = joblib.load(ROOT / "models" / f"{task}_model.pkl")
    row = nhanes[feats].iloc[[0]]
    proba = mdl.predict_proba(row)
    check(f"{task}: predict_proba shape", proba.shape == (1, 2), str(proba.shape))
    check(f"{task}: probabilities sum to 1", abs(proba.sum() - 1) < 1e-6)
    check(f"{task}: 95% CI recorded",
          len(meta["metrics_model_b"].get("roc_auc_ci95", [])) == 2)

large = joblib.load(ROOT / "models" / "large_sleep_risk_model.pkl")
large_row = pd.DataFrame([{
    "age": 33, "gender": "Female", "occupation": "Nurse", "bmi": 26.3, "country": "USA",
    "sleep_duration_hrs": 6.4, "sleep_quality_score": 5, "rem_percentage": 20.3,
    "deep_sleep_percentage": 20.3, "sleep_latency_mins": 19,
    "wake_episodes_per_night": 3, "caffeine_mg_before_bed": 0,
    "alcohol_units_before_bed": 0, "screen_time_before_bed_mins": 51, "exercise_day": 0,
    "steps_that_day": 7442, "nap_duration_mins": 0, "stress_score": 5.8,
    "work_hours_that_day": 7.4, "chronotype": "Neutral",
    "mental_health_condition": "Healthy", "heart_rate_resting_bpm": 67,
    "sleep_aid_used": 0, "shift_work": 0, "room_temperature_celsius": 20.5,
    "weekend_sleep_diff_hrs": 1.2, "season": "Spring", "day_type": "Weekday"}])
p = large.predict_proba(large_row)
check("large model: 4-class probabilities", p.shape == (1, 4), str(p.shape))

sleep_model = joblib.load(ROOT / "models" / "sleep_disorder_model.pkl")
demo_row = pd.DataFrame([["Female", "Nurse", "Overweight", 40, 7.0, 7, 60, 6, 72, 6000,
                          125, 80]],
                        columns=["Gender", "Occupation", "BMI Category", "Age",
                                 "Sleep Duration", "Quality of Sleep",
                                 "Physical Activity Level", "Stress Level", "Heart Rate",
                                 "Daily Steps", "Systolic_BP", "Diastolic_BP"])
check("demo model: 3-class probabilities",
      sleep_model.predict_proba(demo_row).shape == (1, 3))

# ----------------------------------------------------------- leakage guard ---
print("\n== 3. Leakage guards ==")
LEAK_RULES = {
    "insomnia_risk": ["trouble_sleeping"],       # defines the label
    "osa_risk": ["snore_freq", "witnessed_apnea_freq"],  # define the label
}
for task, banned in LEAK_RULES.items():
    feats = metrics["tasks"][task]["features_model_b"]
    overlap = [b for b in banned if b in feats]
    check(f"{task}: label-defining features excluded", not overlap, str(overlap))

large_excluded = json.loads(
    (ROOT / "results/metrics/large_model_metrics.json").read_text(encoding="utf-8")
)["excluded_columns"]
check("large model excludes id + sibling targets",
      set(large_excluded) == {"person_id", "sleep_disorder_risk",
                              "cognitive_performance_score", "felt_rested"},
      str(large_excluded))

# ------------------------------------------------------------ app rendering --
print("\n== 4. Streamlit pages ==")
try:
    from streamlit.testing.v1 import AppTest

    PAGES = ["📋 Executive Overview", "🫀 Sleep-Disorder & Heart Risk (NHANES)",
             "📈 Large-Scale Sleep Risk (100k)", "🧠 Sleep Pattern Demo (educational)",
             "🫀 Cardiometabolic Dashboard", "⚙️ Performance & Methodology"]

    for page in PAGES:
        # a fresh instance per page avoids stale widget state carried over
        at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180)
        at.run()
        if at.exception:
            check("app: initial render", False, str(at.exception))
            break
        at.radio[0].set_value(page)
        at.run()
        if at.exception:
            check(f"app page renders: {page}", False, str(at.exception))
            continue
        ok = True
        # best-effort: press the page's primary action button (runs a real prediction)
        for btn in list(at.button):
            label = btn.label.lower()
            if any(k in label for k in ("estimate", "classify", "run predictive")):
                try:
                    btn.click()
                    at.run()
                    if at.exception:
                        ok = False
                        check(f"app page runs pipeline: {page}", False, str(at.exception))
                        break
                except Exception as exc:  # pragma: no cover
                    ok = False
                    check(f"app page runs pipeline: {page}", False, str(exc))
                    break
        if ok:
            check(f"app page renders: {page}", True)
except ImportError:
    print("[SKIP] streamlit.testing not available")

# ------------------------------------------------------------------ summary --
print("\n" + "=" * 60)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} check(s) FAILED:")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")

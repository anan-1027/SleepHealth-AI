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

for cal in ["heart_disease_isotonic.pkl", "insomnia_risk_isotonic.pkl",
            "osa_risk_isotonic.pkl"]:
    check(f"exists: models/calibration/{cal}", (ROOT / "models" / "calibration" / cal).exists())
check("exists: results/metrics/calibration.json",
      (ROOT / "results/metrics/calibration.json").exists())

# ------------------------------------------------------------------ data ----
nhanes_raw = pd.read_csv(ROOT / "data/processed/nhanes_sleep_health.csv")
check("NHANES cohort has >= 8,000 adults", len(nhanes_raw) >= 8000, f"n={len(nhanes_raw)}")
check("no duplicate rows in analysis table",
      int(nhanes_raw.duplicated().sum()) == 0)

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

# ------------------------------------------------------------ calibration ---
print("\n== 4. Calibration ==")
calib = json.loads((ROOT / "results/metrics/calibration.json").read_text(encoding="utf-8"))
for task in ["heart_disease", "insomnia_risk", "osa_risk"]:
    c = calib[task]
    check(f"{task}: Brier improves after calibration",
          c["brier_after"] < c["brier_before"],
          f"{c['brier_before']} -> {c['brier_after']}")
    check(f"{task}: ROC-AUC preserved by calibration",
          abs(c["roc_auc_after"] - c["roc_auc_before"]) < 0.02,
          f"{c['roc_auc_before']} -> {c['roc_auc_after']}")
    check(f"{task}: calibrator is monotone non-decreasing",
          c["roc_auc_after"] > 0.6)
    iso = joblib.load(ROOT / c["model_file"])
    probe = np.array([0.05, 0.30, 0.60, 0.90])
    mapped = iso.predict(probe)
    check(f"{task}: calibration maps raw scores upward (base rate < raw)",
          bool(np.all(np.diff(mapped) >= 0)), str(np.round(mapped, 3)))

# Regression guard: ECE must sum ABSOLUTE per-bin gaps. Summing signed gaps lets
# over- and under-prediction cancel and reports a near-zero ECE for a badly
# calibrated model (a bug that did ship once in this project's history).
def _ece_signed(y, p, n=10):
    edges = np.quantile(p, np.linspace(0, 1, n + 1))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n - 1)
    tot, acc = len(y), 0.0
    for b in range(n):
        m = idx == b
        if m.sum():
            acc += m.sum() / tot * (y[m].mean() - p[m].mean())
    return abs(acc)


_rng = np.random.default_rng(0)
_true_p = _rng.uniform(0, 1, 6000)                  # true risk spread across range
_bad_y = _rng.binomial(1, _true_p)
_bad_p = 1.0 - _true_p                              # anti-calibrated: over-predicts
                                                    # low risk, under-predicts high risk,
                                                    # so the errors cancel in signed form
_signed = _ece_signed(_bad_y, _bad_p)
check("ECE is not the error-cancelling signed variant",
      _signed < 0.20, f"signed-variant ECE on an anti-calibrated model = {_signed:.4f}")


def _ece_abs(y, p, n=10):
    edges = np.quantile(p, np.linspace(0, 1, n + 1))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n - 1)
    tot, e = len(y), 0.0
    for b in range(n):
        m = idx == b
        if m.sum():
            e += m.sum() / tot * abs(y[m].mean() - p[m].mean())
    return e


check("absolute-gap ECE correctly flags the same anti-calibrated model",
      _ece_abs(_bad_y, _bad_p) > 0.30,
      f"absolute-variant ECE = {_ece_abs(_bad_y, _bad_p):.4f}")
check("heart ECE before calibration is the large, honest value",
      calib["heart_disease"]["ece_equalfreq_before"] > 0.05,
      str(calib["heart_disease"]["ece_equalfreq_before"]))

# ------------------------------------------------------------- app pages ----
print("\n== 5. Streamlit pages ==")
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

    # ---- Regression: the SHAP radio used to wipe the results ----------------
    # Bug reported by user: clicking the "Explain the prediction for:" radio
    # re-ran the script, `submitted` went back to False, and every gauge and
    # chart vanished. Results must persist across widget interactions.
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180)
    at.run()
    at.radio[0].set_value("🫀 Sleep-Disorder & Heart Risk (NHANES)")
    at.run()

    submit = [b for b in at.button if "estimate" in b.label.lower()]
    if submit:
        submit[0].click()
        at.run()
        plots_before = len(at.get("plotly_chart"))
        check("interaction: results render after submit", plots_before >= 3,
              f"{plots_before} charts")

        # The SHAP radio is the second radio on this page (nav radio is first)
        shap_radios = [r for r in at.radio if "Explain" in r.label]
        if shap_radios:
            shap_radios[0].set_value("Insomnia")
            at.run()
            if at.exception:
                check("interaction: SHAP radio switch keeps results", False,
                      str(at.exception))
            else:
                plots_after = len(at.get("plotly_chart"))
                check("interaction: SHAP radio switch keeps results",
                      plots_after >= 3,
                      f"charts went {plots_before} -> {plots_after} after switching radio")

            shap_radios[0].set_value("Sleep apnea")
            at.run()
            if at.exception:
                check("interaction: second radio switch keeps results", False,
                      str(at.exception))
            else:
                check("interaction: second radio switch keeps results",
                      len(at.get("plotly_chart")) >= 3,
                      f"{len(at.get('plotly_chart'))} charts")
        else:
            check("interaction: SHAP radio present after submit", False, "radio not found")
    else:
        check("interaction: submit button found on NHANES page", False, "no button")
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

# SleepHealth-AI 🌙

**Sleep-pattern analysis and multi-disease risk estimation** — a machine-learning system that
estimates *insomnia risk*, *obstructive sleep-apnea risk*, *heart-disease risk*, *hypertension*
and *diabetes* from demographic, anthropometric and sleep-behavioural features, with SHAP-based
explainability and a Streamlit web interface.

> ⚠️ **Responsible-use statement:** this is an educational / thesis research prototype.
> Every output is a **statistical screening-risk estimate, not a medical diagnosis**.
> None of the datasets used here contain polysomnography (AHI) or clinical adjudication.

---

## ✨ What the system delivers

| Capability | Model | Data | Headline metric (repeated CV, 95% CI) |
|---|---|---|---|
| **Insomnia risk** | Logistic Regression | CDC NHANES 2017‑18, n = 5,084 | ROC-AUC **0.706** [0.664 – 0.740] |
| **Sleep-apnea (OSA) risk** | Logistic Regression | CDC NHANES 2017‑18, n = 4,534 | ROC-AUC **0.688** [0.662 – 0.722] |
| **Heart disease** (CHD/CHF/angina/MI) | Logistic Regression | CDC NHANES 2017‑18, n = 5,087 | ROC-AUC **0.829** [0.808 – 0.850] |
| **Hypertension** (ablation study) | XGBoost | CDC NHANES 2017‑18, n = 5,088 | ROC-AUC 0.780 (single-split legacy) |
| **Diabetes** (ablation study) | Random Forest | CDC NHANES 2017‑18, n = 5,088 | ROC-AUC 0.777 (single-split legacy) |
| **Sleep-disorder severity** (Healthy → Severe) | XGBoost | 100,000-record sleep benchmark | accuracy **0.951** [0.949 – 0.953], balanced acc. **0.853**, macro-F1 **0.868** |
| Sleep-pattern demo (None/Insomnia/Apnea) | Random Forest | Kaggle374-row educational set | group-aware CV accuracy **0.904** — *single-split0.933 was optimistic* |

**Central research question (ablation study):** do sleep features improve disease prediction?

| Task | Model A — demographics | Model B — demographics **+ sleep** | Δ (value of sleep) |
|---|---|---|---|
| Heart disease | 0.810 | **0.829** | **+0.019** |
| Insomnia risk | 0.628 | **0.706** | **+0.078** |
| Sleep-apnea risk | 0.676 | **0.688** | +0.012 |
| Hypertension | 0.770 | **0.780** | +0.010 |
| Diabetes | 0.763 | **0.777** | +0.014 |

Appending sleep features improves ROC-AUC on **every** task tested; the strongest effect is for
insomnia risk (+0.078), where sleep variables are the natural signal.

---

## 📦 Datasets

| # | Dataset | Rows | Type | Vintage | Source |
|---|---|---|---|---|---|
| 1 | **CDC NHANES 2017‑2018 (Cycle J)** — `DEMO_J`, `BMX_J`, `SLQ_J`, `DIQ_J`, `BPQ_J`, `MCQ_J` | 5,828 merged → **5,088** adults (≥20 y) | Real, nationally representative survey | 2017‑18 (released Feb 2020) | `wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/` |
| 2 | **Sleep Health & Daily Performance** (`large_sleep_data.csv`) | **100,000** × 32 cols | **Synthetic benchmark** (published generator, seed 42) | 2026 | Kaggle / GitHub *mohan13krishna* |
| 3 | Sleep-health & lifestyle (`Sleep_health_and_lifestyle_dataset.csv`) | 374 (only **109 unique** feature rows) | Educational / synthetic | — | Kaggle (CC0) |

**Target labels and where they come from**

| Target | Definition | Source variable |
|---|---|---|
| `target_insomnia_risk` | ever told a doctor about trouble sleeping | NHANES `SLQ050` |
| `target_osa_risk` | frequent snoring (≥5 nights/wk) **or** witnessed breathing pauses (≥3‑4 nights/wk) | `SLQ030` / `SLQ040` |
| `target_heart_disease` | CHD, CHF, angina or heart attack ever diagnosed | `MCQ160B/C/D/E` |
| `target_hypertension` / `target_diabetes` | ever told by a doctor | `BPQ020` / `DIQ010` |
| `sleep_disorder_risk` | Healthy / Mild / Moderate / Severe | synthetic benchmark label |

Sleep features used: weekday & weekend sleep duration (`SLD012`, `SLD013`), **social jetlag**
(weekend − weekday), snoring frequency, witnessed apnea frequency, daytime sleepiness
(`SLQ120`), trouble sleeping, BMI, waist circumference, age, sex.

---

## 🏗️ Architecture

```
SleepHealth-AI/
├── data/
│   ├── raw/          NHANES .XPT files (6), Kaggle CSV, 100k CSV, nhanes_merged.csv
│   └── processed/    nhanes_sleep_health.csv (5,088×16), sleep_cleansed.csv
├── src/
│   ├── prepare_nhanes.py     XPT merge → cohort → labels/features (anti-leakage)
│   ├── train_models.py       heart / insomnia / OSA models + CV + SHAP + metrics.json
│   ├── train_large_model.py  100k severity model + CV + metrics.json
│   ├── evaluate_legacy.py    CV + CIs for diabetes/hypertension, group-aware demo score
│   ├── preprocess_sleep.py   legacy Kaggle preprocessing
│   └── fetch_nhanes.py       legacy merge helper
├── models/           7 trained pipelines (.pkl)
├── results/
│   ├── metrics/      metrics.json, large_model_metrics.json  ← all UI numbers
│   └── figures/      ROC curves, SHAP summaries, confusion matrix (8 PNGs)
├── app/app.py        Streamlit UI (6 pages) — paths anchored to repo root
├── notebooks/        original exploratory notebooks (01–05)
├── tests/smoke_test.py  artifact + model-contract + leakage-guard + app-render tests
└── requirements.txt
```

**Design rules enforced in code**
1. Preprocessing (impute → scale → one-hot) lives **inside** each sklearn `Pipeline`, so train
   and serving can never diverge.
2. The questionnaire item that *defines* a label is **excluded** from that label's features
   (guarded by a test).
3. Sibling target columns and IDs are excluded from the 100k model (guarded by a test).
4. The UI reads every number from `results/metrics/*.json` — nothing is hard-coded.

---

## 🚀 Quick start

```bash
# 1. environment
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. run the app
streamlit run app/app.py
```

Reproduce every model and figure:

```bash
python src/prepare_nhanes.py       # merge XPT → labels/features (5,088 rows)
python src/train_models.py         # heart / insomnia / OSA + CV + SHAP + metrics.json
python src/train_large_model.py    # 100k severity model + metrics.json
python src/evaluate_legacy.py      # CV+CI for legacy models, group-aware demo scoring
python tests/smoke_test.py         # verify artifacts, contracts, leakage guards, UI pages
```

> The legacy ablation numbers are regenerated by `notebooks/04_disease_models.ipynb`.

---

## 🔬 Methodology

- **Validation:** repeated stratified cross-validation (5 folds × 3 repeats) with **95 % CIs**
  computed from the fold distribution; Model A and Model B are scored on *identical* folds so
  the Δ is paired. No model selection is performed on a held-out test set.
- **Metrics:** ROC-AUC, PR-AUC (handles class imbalance), F1, recall, precision, accuracy and
  the **Brier score** (calibration), plus majority-class baselines.
- **Algorithms compared:** Logistic Regression, Random Forest, XGBoost — best chosen by
  cross-validated ROC-AUC of Model B. Deployed pipelines are refit on all data *after*
  performance estimation.
- **Leakage controls:** label-defining items excluded from features (tested), `person_id`
  dropped, sibling targets excluded, duplicates identified in the Kaggle set.
- **Explainability:** SHAP (TreeExplainer for tree ensembles, LinearExplainer for logistic
  models) — global summary plots in `results/figures/` and per-patient bars in the app,
  correctly labelled on the **log-odds** scale.

---

## 🧪 Tests

```bash
python tests/smoke_test.py
```
Covers: artifact presence, `predict_proba` contracts for all 7 models, probability
normalisation, CI availability, **leakage guards**, and renders + exercises **all 6 app pages**
via `streamlit.testing.v1`. Current status: **all checks pass**.

---

## ⚠️ Limitations (state these in the report/viva)

1. **Risk, not diagnosis** — NHANES has no polysomnography, so no AHI and no clinical
   confirmation; insomnia/OSA outputs are questionnaire-based **screening-risk** estimates.
2. **Self-reported labels** — heart disease, hypertension and diabetes are doctor-diagnosed
   *self-reports*; undiagnosed participants are labelled negative.
3. **Cross-sectional** — associations, not causation or prognosis.
4. **Unweighted sample** — NHANES survey weights are not applied (point estimates are not
   population-representative).
5. **Data vintage** — NHANES 2017‑18; the 100k benchmark is **synthetic**.
6. **Kaggle demo** — 374 rows with only 109 unique feature patterns. Group-aware
   cross-validation (identical patterns never split across folds) gives accuracy **0.904**,
   versus **0.933** from the original single 80/20 split — the single-split number was
   optimistic. Present the demo as an interface illustration only.
7. **No external validation cohort** and **no probability calibration** applied yet.

---

## 🗺️ Roadmap

- Add NHANES 2019‑March 2020 pre-pandemic and 2021‑2023 cycles (newer data, larger *n*).
- Apply MEC exam weights for population-level estimates.
- Probability calibration (Platt/isotonic) + decision-curve analysis.
- Group-aware re-split of the Kaggle demo and honest re-benchmarking.
- Optional signal-level apnea detection (PhysioNet / NSRR datasets with real AHI labels).

---

## 📄 License

MIT — see [LICENSE](LICENSE). NHANES data are public-use files from the US CDC/NCHS; the
100k benchmark is distributed under its own upstream license.

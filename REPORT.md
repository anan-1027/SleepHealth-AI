# SleepHealth-AI — Project Report

**Title:** Sleep-Pattern Analysis and Multi-Disease Risk Estimation Using Explainable Machine Learning

**Type:** MCA / Master's dissertation project

**Deliverable:** End-to-end ML system — data engineering, model training, rigorous validation,
probability calibration, explainability, tested web application, full documentation.

---

## Abstract

Sleep disorders are strongly associated with cardiovascular and metabolic disease, yet they are
under-detected because confirmation requires an overnight sleep study. This project asks a
precise, testable question: **how much does sleep information actually improve the prediction of
disease risk beyond ordinary demographics?**

We built a supervised-learning system on **CDC NHANES 2017–March 2020 pre-pandemic** data
(8,282 adults) that estimates five binary risks — **insomnia**, **obstructive sleep apnea
(OSA)**, **heart disease**, **hypertension** and **diabetes** — plus a **100,000-record** model
that grades sleep-disorder severity from *Healthy* to *Severe*. Every model is validated with
repeated stratified cross-validation and 95% confidence intervals, guarded against data
leakage, probability-calibrated, and explained with SHAP.

**Central finding:** appending sleep features to a demographic baseline improved ROC-AUC on
**every one of five tasks**, with the largest gain for insomnia risk (**+0.078**, 0.629 → 0.707).
Sleep information is measurably useful, but the effect size is task-dependent: large where sleep
is causally close to the outcome (insomnia), modest where it is one risk factor among many
(heart disease, diabetes).

All outputs are **screening-risk estimates for education and research, not clinical diagnoses.**

---

## 1. Problem Statement

1. Analyse sleep patterns (duration, continuity, snoring, daytime sleepiness, social jetlag).
2. Predict **insomnia** and **sleep apnea** risk.
3. Predict **heart-related disease** risk indicated through sleep-pattern analysis.
4. Demonstrate the approach at scale (100,000 records) and with explainability.
5. Present it as a working, defensible system — not a notebook demo.

### The scientific question behind it

A system that reports "your sleep pattern predicts heart disease" is only interesting if sleep
features add information *beyond* age, sex, BMI and waist circumference. Most student projects
skip this test. We make it the **central ablation study** of the project.

---

## 2. Data

### 2.1 Primary dataset — CDC NHANES 2017–March 2020 pre-pandemic

**NHANES** (National Health and Nutrition Examination Survey) is run by the U.S. National Center
for Health Statistics. Unlike a normal questionnaire, it combines a household interview with a
**physical examination and laboratory testing** in a mobile examination centre, and it is
sampled to be nationally representative. It is free, public, and has a published codebook for
every variable — which is what makes the labels in this project verifiable.

We merged six public-use modules by respondent ID (`SEQN`):

| File | Module | Variables used |
|---|---|---|
| `P_DEMO` | Demographics | `RIDAGEYR` (age), `RIAGENDR` (sex) |
| `P_BMX` | Body measures | `BMXBMI`, `BMXWAIST` |
| `P_SLQ` | Sleep | `SLD012`, `SLD013`, `SLQ030`, `SLQ040`, `SLQ050`, `SLQ120` |
| `P_DIQ` | Diabetes | `DIQ010` |
| `P_BPQ` | Hypertension | `BPQ020` |
| `P_MCQ` | Medical conditions | `MCQ160B/C/D/E` |

**Cohort definition:** age ≥ 20, with valid diabetes and hypertension answers → **8,282 adults**.

**Why this release, and not a newer one.** This is a point worth stating precisely, because it
is the first question a reviewer asks:

- The NHANES **2019–2020 cycle was suspended** in March 2020 after data collection in only 18 of
  30 planned survey locations. Those partial data are *not nationally representative on their
  own* and are **not published as a standalone public-use file**.
- NCHS therefore merged 2019–Mar 2020 into the 2017–2018 cycle and released a combined
  **2017–March 2020 pre-pandemic sample** (May 2021) with special survey weights. This
  **supersedes** the 2017–2018 files.
- We therefore use the pre-pandemic release, which raises our cohort from 5,088 to **8,282
  adults (+63%)** while prevalences stay stable (heart disease 9.5% → 9.3%, insomnia 28.1% →
  28.7%). That stability is itself a useful validity signal: adding two more years of data did
  not change what the model sees.
- The next complete cycle (**Aug 2021 – Aug 2023**) straddles the pandemic and uses a **new
  sample design**. Folding it in would require careful re-weighting and could confound the
  sleep–disease associations we measure, so it is listed as future work rather than forced in.

### 2.2 Secondary dataset — 100,000-record sleep benchmark

`large_sleep_data.csv`: 100,000 rows × 32 columns, published with a reproducible generator
(seed 42). It contains sleep, lifestyle, work and physiological features with an ordinal
**`sleep_disorder_risk`** label (Healthy / Mild / Moderate / Severe).

**This dataset is synthetic and is labelled as such everywhere it appears** — in the README, in
the app, and in `results/metrics/large_model_metrics.json`. Its value is scale and class
imbalance (54% / 33% / 8% / 4%), which lets us demonstrate large-n training and honest
imbalanced-class evaluation.

### 2.3 Educational dataset — Kaggle sleep/lifestyle (374 rows)

A small public dataset used only for the interface demo. It has 374 rows but only **109 unique
feature patterns**, and its labels are undocumented synthetic values. It is presented as an
educational demo, never as evidence — see §5.3.

---

## 3. Labels — and why each one is defensible

| Target | Definition | Source | Type |
|---|---|---|---|
| `target_insomnia_risk` | ever told a doctor about trouble sleeping | `SLQ050` = 1 | self-report |
| `target_osa_risk` | frequent snoring (≥5 nights/wk) **or** witnessed breathing pauses (≥3–4 nights/wk) | `SLQ030` = 3 **or** `SLQ040` ≥ 2 | symptom-based |
| `target_heart_disease` | CHD, CHF, angina or heart attack ever diagnosed | `MCQ160B/C/D/E` = 1 | self-report |
| `target_hypertension` | told by a doctor one has high blood pressure | `BPQ020` = 1 | self-report |
| `target_diabetes` | told by a doctor one has diabetes | `DIQ010` = 1 | self-report |
| `sleep_disorder_risk` | Healthy < Mild < Moderate < Severe | synthetic benchmark | synthetic |

**Why these definitions.** NHANES contains **no polysomnography and no apnea–hypopnea index
(AHI)**, so true clinical OSA labels do not exist in this data. Rather than invent a clinical
claim, we define a **symptom-based screening label** from two published questionnaire items and
say exactly that in the documentation. The insomnia label uses the one item that directly asks
whether a physician identified a sleep problem.

**Sleep features extracted:** weekday sleep duration (`SLD012`), weekend sleep duration
(`SLD013`), **social jetlag** (weekend − weekday, a behavioural rhythm measure), snoring
frequency, witnessed-apnea frequency, daytime sleepiness (`SLQ120`), trouble sleeping, plus
BMI, waist circumference, age and sex.

### 3.1 A data-engineering trap worth knowing

`pandas.read_sas` decodes the SAS transport integer `0` as the denormalised float
`5.397605346934028e-79`. In the raw files this value appears in 1,583 rows of `SLQ030` and 4,234
rows of `SLQ040`. Left alone it silently corrupts every frequency feature; converted carelessly
it destroys genuine `NaN` missing values (BMI appeared to have zero missingness). The fix in
`prepare_nhanes.py` normalises only tiny non-null values and preserves true missingness, and is
validated against the CDC codebook counts.

---

## 4. Methodology

### 4.1 Leakage control (the most important section)

Data leakage — where information about the answer accidentally reaches the model — is the
single most common way a student project produces impressive but meaningless numbers. Four
specific controls were implemented, each enforced by an automated test:

1. **Label-defining items are excluded from that label's features.** Predicting insomnia means
   `trouble_sleeping` must not be a feature; predicting OSA means `snore_freq` and
   `witnessed_apnea_freq` must not be features. Otherwise the model answers by copying the
   question.
2. **Identifiers and sibling targets excluded.** `person_id` and the two other target columns
   (`cognitive_performance_score`, `felt_rested`) are dropped from the 100k model.
3. **No model selection on a held-out test set.** The original notebook picked the best model
   by test accuracy and reported that same accuracy — optimistic by construction. Here,
   algorithm selection happens inside cross-validation and every reported number is out-of-fold.
4. **Duplicate-aware scoring for the Kaggle demo.** See §5.3.

### 4.2 Validation design

- **Repeated stratified k-fold CV, 5 folds × 3 repeats** (15 out-of-fold evaluations per task).
- **95% confidence intervals** from the fold distribution (2.5th / 97.5th percentiles).
- **Paired ablation:** Model A and Model B are scored on *identical* folds, so the Δ is a
  paired comparison, not two unrelated numbers.
- **Metrics:** ROC-AUC, **PR-AUC** (honest under class imbalance), F1, recall, precision,
  accuracy, **Brier score** (calibration), plus the majority-class baseline for context.
- **Algorithms compared:** Logistic Regression, Random Forest, XGBoost. The winner is chosen by
  cross-validated ROC-AUC of Model B; deployed pipelines are refit on all data *after*
  performance estimation.

### 4.3 Feature sets (the ablation)

- **Model A (baseline):** age, sex, BMI, waist circumference.
- **Model B (ours):** Model A **+** weekday/weekend sleep duration, social jetlag, snoring
  frequency, witnessed-apnea frequency, daytime sleepiness, trouble sleeping.

The difference between A and B is the project's scientific answer.

### 4.4 Preprocessing inside the pipeline

Imputation (median) → scaling (standard) → one-hot encoding all live **inside the sklearn
`Pipeline`**, so the training code and the serving code cannot diverge. There is no separate
preprocessing step to forget.

---

## 5. Results

### 5.1 Headline performance (repeated CV, 95% CI)

| Task | n | Positive rate | Best model | ROC-AUC | 95% CI | PR-AUC |
|---|---|---|---|---|---|---|
| Heart disease | 8,281 | 9.3% | Logistic Regression | **0.821** | [0.799, 0.839] | 0.305 |
| Hypertension | 8,282 | 38.2% | XGBoost | **0.785** | [0.766, 0.805] | 0.662 |
| Diabetes | 8,282 | 15.9% | RandomForest | **0.770** | [0.740, 0.796] | 0.354 |
| Insomnia risk | 8,275 | 28.7% | Logistic Regression | **0.707** | [0.689, 0.727] | 0.495 |
| OSA risk | 7,444 | 32.1% | Logistic Regression | **0.692** | [0.663, 0.709] | 0.501 |

### 5.2 Central ablation — does sleep help?

| Task | Model A (demographics) | Model B (+ sleep) | **Δ** |
|---|---|---|---|
| Insomnia risk | 0.629 | **0.707** | **+0.078** |
| Heart disease | 0.803 | **0.821** | **+0.018** |
| OSA risk | 0.679 | **0.692** | +0.013 |
| Hypertension | 0.774 | **0.785** | +0.011 |
| Diabetes | 0.763 | **0.770** | +0.008 |

**Interpretation.** Sleep features improve all five tasks, and the ordering is exactly what a
clinician would predict: the gain is largest for **insomnia** (+0.078), where sleep variables
are essentially the defining signal, and modest for **heart disease** (+0.018) and **diabetes**
(+0.008), where sleep is one risk factor among many (age and adiposity dominate). This pattern
is a *credibility* signal — a project that reported uniformly large gains across all tasks
would be the one to distrust.

### 5.3 The Kaggle demo scored honestly

The original notebook reported 93.3% accuracy from a single 80/20 split. Because the dataset has
only 109 unique patterns across 374 rows, that split is not trustworthy. We re-scored it both
ways:

| Scoring | Accuracy |
|---|---|
| Single 80/20 split (original) | 0.933 |
| Row-wise repeated CV | 0.906 |
| **Group-aware CV** (`StratifiedGroupKFold`, identical patterns never straddle folds) | **0.904** |

The ~3-point drop quantifies the optimism of the single split. The group-aware number is the one
the app displays.

### 5.4 The 100,000-record severity model

| Metric | Value |
|---|---|
| Accuracy | **0.951** [0.949, 0.953] |
| **Balanced accuracy** | **0.853** |
| **Macro-F1** | **0.868** |

Per-class hold-out recall: Healthy 0.997, Mild 0.962, Moderate 0.683, Severe 0.754.

Because the classes are 54 / 33 / 8 / 4%, plain accuracy is close to meaningless — a model that
always said "Healthy" would score 54%. **Balanced accuracy and macro-F1 are therefore reported
alongside it**, and the confusion matrix shows the minority classes are genuinely being learned
rather than ignored.

### 5.5 Probability calibration

Raw model scores were **badly over-confident**: a raw score of 0.84 corresponded to an observed
event rate near 0.34. The cause is `class_weight="balanced"`, which is the right choice for
discriminating a rare class but shifts scores away from the true base rate.

We fitted a **cross-fitted isotonic** map (the calibrator is fitted on 4/5 of the out-of-fold
predictions and scored on the held-out 1/5, rotating, so it is never evaluated on data it saw):

| Task | Brier ↓ | ECE (equal-freq) ↓ | ROC-AUC |
|---|---|---|---|
| Heart disease | 0.180 → **0.074** | 0.276 → **0.005** | 0.820 → 0.813 |
| Insomnia risk | 0.217 → **0.182** | 0.185 → **0.013** | 0.707 → 0.704 |
| OSA risk | 0.222 → **0.197** | 0.158 → **0.013** | 0.691 → 0.686 |

ROC-AUC is essentially unchanged, as theory requires — calibration can change the *level* of a
prediction but never its *ranking*. **The app gauges display calibrated risks.**

> **A note on intellectual honesty in this section.** An early version of the ECE calculation
> took the absolute value *after* summing signed per-bin gaps, which let over- and
> under-prediction cancel and reported an implausible ECE of 0.0001. The bug was caught by
> comparing against an independent re-implementation, corrected (ECE must be Σ w·|gap|), and a
> **regression test** was added that feeds an anti-calibrated model to both variants and asserts
> the buggy one stays near zero while the correct one exceeds 0.30.

### 5.6 Explainability

SHAP provides both global and local explanations: TreeExplainer for the tree ensembles and
LinearExplainer for the logistic models, with the axes correctly labelled on the **log-odds**
scale (an earlier version mislabelled log-odds as probability). Global summary plots are saved
to `results/figures/`; the app renders per-patient contribution bars.

---

## 6. Software engineering

| Aspect | Implementation |
|---|---|
| **Architecture** | `src/` (reproducible scripts) → `models/` → `results/` → `app/`. Notebooks are exploratory only; the production path is the scripts. |
| **Reproducibility** | Fixed seeds, `fetch_data.py` downloads and **validates the XPORT header**, `prepare_nhanes.py` regenerates the cohort from raw files. |
| **Path handling** | The app anchors all paths to the repository root, so it runs from any working directory. |
| **Dynamic metrics** | The UI reads every number from `results/metrics/*.json`. **Nothing is hard-coded in the UI**, so a retrained model cannot be contradicted by a stale caption. |
| **Testing** | `tests/smoke_test.py` — artifact presence, `predict_proba` contracts for all 7 models, probability normalisation, CI availability, **leakage guards**, **calibration sanity + regression guard**, cohort size, duplicate rows, and all 6 app pages rendered and executed via `streamlit.testing.v1`. |
| **Dependencies** | `requirements.txt` complete (a missing `plotly` was previously a hard deploy blocker). |

### Known defects found and fixed

1. All five "raw CDC files" in the original repository were actually **20,905-byte HTML error
   pages**, not data — the real data only ever existed inside a CSV. Replaced with genuine
   downloads plus an XPORT validation guard so this cannot recur silently.
2. `requirements.txt` missing `plotly` — the app crashed on any fresh install.
3. UI wording "Primary Diagnosis" — inappropriate for statistical outputs; now "screening risk
   estimate, not a diagnosis", with a persistent disclaimer.
4. Dropdown values that never appeared in training (`Other`, `Underweight`) and a missing valid
   value (`Sales Representative`) — silently produced wrong predictions.
5. SHAP axis labelled "probability" while showing log-odds.
6. Metrics hard-coded in the UI, capable of contradicting the models.
7. CWD-relative paths broke the app when launched from elsewhere.
8. `read_sas` zero-artifact corrupting sleep-frequency features (§3.1).
9. Model selection performed on the test set (optimistic by construction).

---

## 7. Application

Six pages, built with Streamlit:

1. **Executive Overview** — datasets, targets, headline results.
2. **Sleep-Disorder & Heart Risk (NHANES)** — the core deliverable: one profile form →
   calibrated heart / insomnia / OSA gauges, SHAP local explanation, per-task metrics.
3. **Large-Scale Sleep Risk (100k)** — severity distribution, confusion matrix, imbalanced-class
   metrics.
4. **Sleep Pattern Demo (educational)** — the Kaggle demo, with its honest group-aware score.
5. **Cardiometabolic Dashboard** — diabetes and hypertension with explanations.
6. **Performance & Methodology** — every metric table, figures, and the limitations list.

Run it: `streamlit run app/app.py`

---

## 8. Limitations — state these proactively

1. **Risk, not diagnosis.** No polysomnography, no AHI, no clinical adjudication. Insomnia and
   OSA outputs are questionnaire-based screening-risk estimates.
2. **Self-reported labels.** Heart disease, hypertension and diabetes are self-reports of a
   physician diagnosis; undiagnosed participants are labelled negative, which biases toward the
   null and caps achievable performance.
3. **Cross-sectional.** NHANES measures everything at one time point, so results are
   **associations, not causation or prognosis**. We cannot claim that changing sleep changes
   disease risk.
4. **Unweighted sample.** NHANES survey weights are not applied, so point estimates are not
   population-representative.
5. **Synthetic secondary data.** The 100k benchmark is generated, not observed.
6. **No external validation.** Every metric is cross-validated within the same NHANES sample;
   the models have never been tested on a different population.
7. **Modest effect sizes for OSA.** AUC ~0.69 is useful for ranking, not for individual
   decisions — consistent with symptom-based labels rather than a true clinical endpoint.

---

## 9. Conclusion

This project delivers a working, tested and honestly-evaluated system that analyses sleep
patterns and estimates the risk of insomnia, sleep apnea, heart disease, hypertension and
diabetes. Its scientific contribution is a **quantified answer to a question most projects
never ask**: sleep features improve disease-risk prediction on all five tasks studied, most
strongly for insomnia (+0.078 AUC) and more modestly for heart disease (+0.018), where
demographics already carry most of the signal. The methodology — repeated cross-validation with
confidence intervals, paired ablations, explicit leakage guards, probability calibration and
SHAP explainability — is designed to survive scrutiny rather than to maximise a headline
number.

---

## 10. Future work

- Add the **NHANES Aug 2021–Aug 2023** cycle with proper re-weighting to extend recency.
- Apply **MEC exam weights** for population-level estimates.
- **Decision-curve analysis** to convert calibrated risks into clinically meaningful net benefit.
- **External validation** on an independent cohort.
- Signal-level apnea detection using PhysioNet / NSRR datasets that carry **real AHI labels**,
  which would upgrade the OSA task from symptom-based screening to true clinical prediction.

---

## Appendix A — Reproduce everything

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

python src/fetch_data.py           # download NHANES (validates XPORT header)
python src/prepare_nhanes.py       # merge + labels + features  -> 8,282 rows
python src/train_models.py         # heart/insomnia/OSA + CV + SHAP
python src/train_large_model.py    # 100k severity model
python src/calibrate_models.py     # cross-fitted isotonic calibration
python src/evaluate_legacy.py      # diabetes/HTN CV + group-aware demo score
python tests/smoke_test.py         # full verification

streamlit run app/app.py
```

## Appendix B — Artifact map

| Path | Contents |
|---|---|
| `data/processed/nhanes_sleep_health.csv` | 8,282 × 16 analysis table |
| `models/*.pkl` | 7 trained pipelines |
| `models/calibration/*.pkl` | 3 isotonic calibration maps |
| `results/metrics/metrics.json` | all NHANES task metrics + CIs + ablations |
| `results/metrics/large_model_metrics.json` | 100k model metrics |
| `results/metrics/calibration.json` | Brier/ECE before–after + reliability tables |
| `results/figures/*.png` | 8 figures: ROC curves, SHAP summaries, confusion matrix |

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| **ROC-AUC** | Probability that a randomly chosen positive case scores higher than a randomly chosen negative one. 0.5 = chance, 1.0 = perfect ranking. |
| **PR-AUC** | Average precision–recall; more informative than ROC-AUC when the positive class is rare. |
| **Brier score** | Mean squared error of the probability — measures *calibration*, lower is better. |
| **ECE** | Expected Calibration Error: the average gap between stated confidence and observed frequency. |
| **Cross-validation** | Rotating train/test splits so every observation is scored by a model that never trained on it. |
| **Ablation** | Removing a feature group to measure exactly what it contributed. |
| **Data leakage** | Information about the answer reaching the model; inflates metrics without real skill. |
| **SHAP** | Shapley Additive Explanations — attributes a prediction to each feature. |
| **Social jetlag** | Difference between weekend and weekday sleep; a proxy for circadian misalignment. |
| **AHI** | Apnea–Hypopnea Index — the clinical measure of OSA severity. **Not present in NHANES**, which is why our OSA output is screening-only. |
| **NHANES** | National Health and Nutrition Examination Survey (CDC/NCHS). |

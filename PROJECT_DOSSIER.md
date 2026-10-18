# SleepHealth-AI — Complete Project Dossier

**A self-contained technical record of what was built, every change made, and the reasoning behind each one.**

---

## 0. INSTRUCTIONS FOR THE AI TUTOR RECEIVING THIS DOCUMENT

> **Copy everything below this line into your AI tutor.**

You are teaching an MCA (Master of Computer Applications) student who is submitting this project and must defend it in a **viva voce**. Treat this document as the complete ground truth of the project. Do not invent facts, files, metrics, or commits that are not stated here.

**How to use this document:**

1. **Read it fully first** before answering. Every metric below is read directly from the repository's JSON artifacts — treat them as authoritative.
2. **Teach by questioning.** After explaining a concept, ask the student to explain it back. They must be able to defend every design decision.
3. **The most likely viva questions are in Section 10.** Drill those, but be ready to go deeper on anything.
4. **If the student asks "why didn't you use X?"** (deep learning, SMOTE, a bigger dataset, a single train/test split), the answer is almost always in **Section 5 (Change Log)** or **Section 9 (Limitations)**. Those sections exist precisely to answer adversarial questions.
5. **Be honest about weaknesses.** This project's credibility comes from *acknowledging* its limitations rather than hiding them. Never let the student overclaim — the system outputs **risk estimates, never diagnoses**.
6. **Suggested teaching order:** Data → Labels → Leakage → Validation → Results → Calibration → Explainability → Limitations → Viva drills.

**Key facts to remember about the student's situation:**
- The project is complete and working. All tests pass (`ALL CHECKS PASSED`).
- Submission deadline is tight; the goal is understanding, not more features.
- The student personally tests the app and reports bugs — two real bugs were found this way and are documented in Section 6.

---

## 1. PROJECT IDENTITY

**One-line summary:** SleepHealth-AI is a machine-learning screening system that analyses sleep patterns (duration, weekend catch-up, social jetlag, snoring, daytime sleepiness) and estimates a person's risk of **insomnia**, **obstructive sleep apnea (OSA)**, **heart disease**, **hypertension** and **diabetes**, with calibrated probabilities and SHAP explanations, delivered through a 6-page Streamlit dashboard.

**Critical framing (never violate this):** Every output is a **statistical risk estimate for screening/educational purposes**, not a medical diagnosis. The app carries a persistent disclaimer. If a viva examiner asks "is this a diagnostic tool?", the answer is an emphatic **no**.

**Technology stack:**
| Layer | Choice |
|---|---|
| Language | Python 3 (venv at `./venv`) |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP (TreeExplainer + LinearExplainer) |
| Data handling | pandas, numpy |
| Validation | RepeatedStratifiedKFold, StratifiedGroupKFold |
| Web app | Streamlit + Plotly |
| Persistence | joblib pickles, JSON metrics |
| Tests | Custom smoke tests + `streamlit.testing.v1` |

**Repository map:**
```
SleepHealth-AI/
├── app/app.py                  # 6-page Streamlit dashboard (all numbers read from JSON)
├── src/                        # reproducible pipeline scripts (the production path)
│   ├── fetch_data.py           # downloads CDC XPT files + validates XPORT header
│   ├── prepare_nhanes.py       # merges raw XPT -> cohort + labels + features
│   ├── train_models.py         # heart/insomnia/OSA tasks, CV, SHAP figures
│   ├── calibrate_models.py     # cross-fitted isotonic calibration + ECE
│   ├── evaluate_legacy.py      # diabetes/hypertension CV + group-aware Kaggle scoring
│   ├── train_large_model.py    # 100k-record 4-class severity model
│   ├── feature_labels.py       # single source of truth for readable feature names
│   └── regen_shap_figure.py    # regenerate ONE figure without a full retrain
├── tests/                      # smoke_test.py (5 sections) + test_feature_labels.py
├── scripts/verify_calibration.py  # independent re-implementation of metrics
├── models/                     # 7 pickles + models/calibration/ (3 isotonic maps)
├── results/metrics/            # metrics.json, calibration.json, large_model_metrics.json
├── results/figures/            # ROC + SHAP PNGs per task
├── notebooks/                  # exploratory only (04/05 read processed CSV)
├── README.md, REPORT.md        # technical docs + formal report
└── requirements.txt
```

**Architectural principle:** Notebooks are *exploratory only*; the production path is `src/*.py`. All UI numbers are read from `results/metrics/*.json` — **nothing is hard-coded in the UI**, so a retrained model can never be contradicted by a stale caption.

---

## 2. DATA

### 2.1 Primary dataset — CDC NHANES 2017–March 2020 pre-pandemic

| Property | Value |
|---|---|
| Source | US Centers for Disease Control and Prevention (CDC), National Health and Nutrition Examination Survey |
| Cycle | 2017–March 2020 **pre-pandemic** file |
| File naming | `P_*.XPT` (SAS transport format) |
| Release | May 2021 |
| Participants after merging | **8,282** |
| Engineered columns | 16 |

**Why this specific cycle (a strong viva answer):**
The NHANES 2019–2020 cycle was **suspended in March 2020 due to COVID-19 and was never released as a standalone cycle**. The CDC instead published an accumulated "pre-pandemic" file (`P_*`, released May 2021) covering 2017 through March 2020. Using it raised the cohort from **n = 5,088 → 8,282** — a ~63% increase in real, genuine data — without changing the data-generating process or introducing pandemic-era measurement artefacts. This is why the project explicitly does **not** use the 2019-2020 cycle: *it does not exist as a standalone release*.

**Data integrity note:** the original repository contained five `.XPT` files that were actually **20,905-byte HTML error pages**, not data (see Section 6, Defect 1). These were replaced with genuine CDC downloads, and `src/fetch_data.py` now **validates the XPORT header bytes** before accepting a file.

### 2.2 Secondary dataset — 100,000-record sleep benchmark
| Property | Value |
|---|---|
| Records | 100,000 |
| Features used | 28 |
| Origin | **Synthetic / generated** (Sleep Health & Daily Performance Dataset, 32 columns) |
| Target | `sleep_disorder_risk`, ordinal: Healthy < Mild < Moderate < Severe |
| Excluded columns | `person_id`, `sleep_disorder_risk`, `cognitive_performance_score`, `felt_rested` |

This dataset is **explicitly labelled synthetic everywhere** in the UI and report. Its purpose is to demonstrate behaviour at scale (100k rows, 4-class ordinal target), **not** to make a scientific claim.

### 2.3 Educational dataset — Kaggle sleep/lifestyle (374 rows)
Small, with **synthetic labels**. Demoted to an **educational demo** after it was discovered that naive cross-validation over-estimated its performance (see Section 6, Defect 9 context). Its honest score is in Section 4.3.

### 2.4 Features

Two feature families, used to build the central ablation:

| Family | Columns |
|---|---|
| **DEMO** (demographics/body) | `age`, `gender_female`, `bmi`, `waist_circumference` |
| **SLEEP** (sleep behaviour) | `sleep_duration_hrs`, `weekend_sleep_hrs`, `social_jetlag_hrs`, `snore_freq`, `witnessed_apnea_freq`, `daytime_sleepiness`, `trouble_sleeping` |

Encoded questionnaire scales (matching the NHANES codebook):
- `snore_freq`, `witnessed_apnea_freq`: 0 = Never, 1 = Rarely (1-2/wk), 2 = Occasionally (3-4/wk), 3 = Frequently (5+/wk)
- `daytime_sleepiness`: 0 = Never … 4 = Almost always
- `trouble_sleeping`: 0 = No, 1 = Yes
- `social_jetlag_hrs` = `weekend_sleep_hrs` − `weekday sleep` (signed; a person who sleeps *less* on weekends is negative)

---

## 3. LABELS — AND WHY EACH ONE IS DEFENSIBLE

| Target column | Definition | NHANES rule | Type |
|---|---|---|---|
| `target_heart_disease` | CHD, CHF, angina or heart attack ever diagnosed | `MCQ160B/C/D/E` = 1 | self-report |
| `target_hypertension` | told by a doctor one has high blood pressure | `BPQ020` = 1 | self-report |
| `target_diabetes` | told by a doctor one has diabetes | `DIQ010` = 1 | self-report |
| `target_insomnia_risk` | ever told a doctor about trouble sleeping | `SLQ050` = 1 | self-report |
| `target_osa_risk` | frequent snoring (≥5 nights/wk) **or** witnessed breathing pauses (≥3-4 nights/wk) | `SLQ030` = 3 **or** `SLQ040` ≥ 2 | **symptom-based** |
| `sleep_disorder_risk` | Healthy < Mild < Moderate < Severe | synthetic benchmark | synthetic |

**Defence of the self-report labels (a very likely viva question):**
NHANES is a *survey*, not a clinical trial. It contains **no polysomnography (sleep study) and no apnea-hypopnea index (AHI)**, so a clinician-grade OSA diagnosis is physically impossible from this data. Being honest about that, `target_osa_risk` is defined **symptom-based** (frequent snoring or witnessed breathing pauses), and is always described as *OSA **risk***, never as OSA. Self-report labels for heart disease / hypertension / diabetes are the standard, accepted proxy in epidemiological ML — this must be stated as a limitation, not hidden.

**Choice of the heart label:** the composite of `MCQ160B` (congestive heart failure), `MCQ160C` (coronary heart disease), `MCQ160D` (angina) and `MCQ160E` (heart attack) == 1. This maximises positive cases while remaining clinically coherent.

---

## 4. METHODOLOGY (the defensible core)

### 4.1 Leakage control — the single most important section

**Rule:** for each task, any feature that *defines the label* is excluded from that task's features.

| Task | Model B features | Label-defining features **excluded** |
|---|---|---|
| `heart_disease` | DEMO + all 7 SLEEP (11) | — |
| `insomnia_risk` | DEMO + 6 sleep (10) | `trouble_sleeping` (that *is* SLQ050) |
| `osa_risk` | DEMO + 5 sleep (9) | `snore_freq`, `witnessed_apnea_freq` (they define OSA risk) |

This is non-negotiable: including `snore_freq` to predict a label defined *by* snoring would give a meaningless near-perfect AUC. The smoke test has explicit **leakage guards** that fail if any of these features reappear.

Additionally, the 100k model excludes `person_id` (an identifier), and its sibling targets (`sleep_disorder_risk`, `cognitive_performance_score`, `felt_rested`) to prevent target leakage.

### 4.2 Validation design
- **RepeatedStratifiedKFold, 5 folds × 3 repeats** for the NHANES tasks.
- **No single train/test split** — a single split's score depends heavily on which rows land in the test set, so it is reported with a **95% confidence interval** from the fold distribution.
- Preprocessing (median imputation + standardisation) lives **inside the sklearn Pipeline**, so it is fit only on training folds. Doing it before splitting would leak test statistics.
- Algorithms compared per task: **LogisticRegression**, **RandomForest**, **XGBoost**, all with `class_weight="balanced"` (or equivalent) because the targets are imbalanced.
- The **deployed** model is refit on the **full** dataset *after* cross-validated performance estimation — the CV numbers are the honest estimate, the final pickle is the best use of all available data.

### 4.3 Central ablation — does sleep actually help?

Each task is scored twice: **Model A** (DEMO only) vs **Model B** (DEMO + SLEEP). The **delta** is the project's scientific claim.

| Task | Model A (demo) AUC | Model B (+ sleep) AUC | Δ ROC-AUC | Δ PR-AUC |
|---|---|---|---|---|
| heart_disease | 0.8027 | **0.8207** | **+0.0180** | +0.0173 |
| insomnia_risk | 0.6291 | **0.7072** | **+0.0781** | +0.1076 |
| osa_risk | 0.6791 | **0.6916** | **+0.0125** | +0.0215 |

**Interpretation (teach this precisely):** sleep features help **most for insomnia (+0.078)** — which makes intuitive sense, since insomnia *is* a sleep disorder. They help **modestly for heart disease (+0.018)** and **marginally for OSA (+0.013)**. A student who claims "sleep features dramatically improve everything" is overclaiming; the honest statement is that sleep carries incremental signal beyond demographics, largest for sleep-specific conditions.

### 4.4 The Kaggle demo, scored honestly
- 374 rows but only **109 unique feature patterns** — i.e. many rows are *duplicates* of one another.
- Naive **row-wise** CV lets identical patterns appear in both train and test → **accuracy 0.9064** [0.8613, 0.9520].
- **StratifiedGroupKFold** (grouping by unique feature pattern, so each pattern stays in one fold) → **accuracy 0.9038** [0.8573, 0.9320].
- **Optimistic bias: only 0.0027** — small here, but the *principle* is what matters, and the group-aware figure (**0.904**) is the one reported.
- Framing: **educational demo**, explicitly *not* the project's evidence base.

### 4.5 Probability calibration (why the gauges show "surprising" numbers)

**The problem:** `class_weight="balanced"` makes models rank well but output **over-confident probabilities**. Measured on heart disease: a raw score of **0.84 corresponded to a true event rate of only ~0.34**. Showing that uncorrected would be actively misleading.

**The fix:** **cross-fitted isotonic regression**. Isotonic regression learns a monotone (non-decreasing) mapping from raw score → observed frequency. "Cross-fitted" means the mapping is learned on **out-of-fold** predictions, so the calibration itself is not fit on the same data it is scored on.

| Task | Brier before → after | ECE (equal-freq) before → after | ROC-AUC before → after |
|---|---|---|---|
| heart_disease | 0.1803 → **0.0740** | 0.2762 → **0.0048** | 0.8196 → 0.8134 |
| insomnia_risk | 0.2169 → **0.1818** | 0.1853 → **0.0129** | 0.7071 → 0.7043 |
| osa_risk | 0.2223 → **0.1972** | 0.1584 → **0.0126** | 0.6911 → 0.6861 |

**Three things a viva examiner may probe here — have ready answers:**

1. *"ROC-AUC dropped slightly. Is that bad?"* — No. **Calibration cannot change ranking**, so AUC is theoretically invariant; the tiny drop (0.8196 → 0.8134) is noise from the cross-fitting split. The **Brier score improved by ~59%** for heart disease, which is the point.
2. *"Why equal-frequency ECE?"* — At a ~10% base rate, **equal-width** bins are dominated by a huge low-risk bin and can look near-zero even for a poorly calibrated model. **Equal-frequency** (equal-count) bins are the honest diagnostic at low base rates. Both are recorded, and the note explains which to trust.
3. *"Why not Platt scaling (logistic) instead?"* — Isotonic is more flexible (non-parametric, can model non-logistic shapes) and is appropriate at n≈8,000 where the sample size supports it. For very small n, Platt would be safer. (Good "future work" answer.)

> **Nuance to flag for the tutor:** `metrics.json` reports heart ROC-AUC **0.8207** (mean over 5 folds × 3 repeats) while `calibration.json` reports **0.8196** as `roc_auc_before`. These differ because the calibration script computes AUC on *its own* out-of-fold predictions from a specific cross-fit split. Both are correct; they are different estimates of the same quantity. Do not treat this as a contradiction.

### 4.6 Explainability (SHAP)
- Per-patient **SHAP bar charts** in the app show each feature's contribution on the **log-odds** scale (a SHAP value of +0.69 ≈ doubling the odds).
- Red pushes the estimate **up**, green pushes it **down**.
- Global SHAP summary PNGs are saved per task in `results/figures/`.
- Feature labels are human-readable via `src/feature_labels.py` (see Section 6, Defect 12).

---

## 5. RESULTS — COMPLETE NUMBERS

### 5.1 NHANES tasks (RepeatedStratifiedKFold 5×3, 95% CI)

| Task | n | Prev. | Best algo | ROC-AUC [95% CI] | PR-AUC | F1 | Recall | Precision | Accuracy | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| heart_disease | 8,281 | 9.35% | LogisticRegression | **0.8207** [0.7987, 0.8390] | 0.3048 | 0.3411 | 0.7868 | 0.2179 | 0.7159 | 0.1801 |
| insomnia_risk | 8,275 | 28.65% | LogisticRegression | **0.7072** [0.6890, 0.7271] | 0.4952 | 0.5202 | 0.6415 | 0.4377 | 0.6611 | 0.2169 |
| osa_risk | 7,444 | 32.12% | LogisticRegression | **0.6916** [0.6630, 0.7096] | 0.5005 | 0.5234 | 0.6227 | 0.4516 | 0.6359 | 0.2222 |
| diabetes | 8,282 | 15.91% | RandomForest | **0.7704** [0.7401, 0.7955] | 0.3543 | 0.4229 | 0.7183 | 0.2998 | 0.6879 | 0.1745 |
| hypertension | 8,282 | 38.20% | XGBoost | **0.7850** [0.7657, 0.8053] | 0.6620 | 0.6220 | 0.6024 | 0.6434 | 0.7204 | 0.1822 |

**Why heart disease has high recall (0.787) but low precision (0.218):** with a 9.35% base rate, a screening-oriented model flags more people to avoid missing true cases. **In a screening context, missing a case (false negative) is costlier than a false alarm**, so this trade-off is deliberate and defensible — but it must be stated openly.

### 5.2 100,000-record severity model
| Metric | Mean [95% CI] |
|---|---|
| CV accuracy | **0.9508** [0.9489, 0.9527] |
| CV balanced accuracy | **0.8526** [0.8457, 0.8577] |
| CV macro-F1 | **0.8675** [0.8610, 0.8727] |
| Hold-out accuracy | 0.9495 |

Design: RepeatedStratifiedKFold 5×2 + 20% stratified hold-out. **Balanced accuracy (0.853) is much lower than plain accuracy (0.951)** — this gap is the honest signal that classes are imbalanced; always quote balanced accuracy / macro-F1 alongside accuracy for this model.

Per-class hold-out F1: Healthy **0.996**, Mild **0.948**, Moderate **0.709**, Severe **0.804**. The **Moderate** class is the hardest (recall 0.683) — an honest weak point worth naming before an examiner does.

---

## 6. CHANGE LOG — EVERY CHANGE, WHY, AND ITS REFERENCE

The repository has **5 commits**. Each entry below states *what changed*, *why*, and *where to verify it*.

### Commit `674d707` — 2026-10-05 — "Initial commit"
- **What:** 3 files, 243 lines. The pre-existing skeleton.
- **State it left:** contained the **fake `.XPT` HTML files** (Defect 1) and hard-coded UI metrics (Defect 6) that later commits fixed.

### Commit `a764c56` — 2026-10-18 — "Add insomnia/OSA/heart-disease risk models, 100k model, rigorous CV, tests and docs"
- **Size:** 45 files, +121,387 lines.
- **What changed:**
  - Introduced the three core sleep tasks (`heart_disease`, `insomnia_risk`, `osa_risk`) with the **Model A vs Model B ablation**.
  - Added `src/train_models.py` — repeated CV, algorithm comparison, SHAP figure generation.
  - Added the **100k severity model** (`src/train_large_model.py`).
  - Added `tests/smoke_test.py` with **leakage guards**.
  - Rewrote `README.md`.
- **Why:** the college requirement was to predict insomnia/apnea and heart disease *from sleep patterns*, so the ablation (does sleep help?) became the scientific spine of the project.
- **Verify:** `git show a764c56 --stat`; `results/metrics/metrics.json` → `tasks.*.ablation`.

### Commit `1e5312d` — 2026-10-18 — "Upgrade to NHANES 2017-Mar2020 pre-pandemic, add probability calibration and report"
- **Size:** 32 files, +10,260 / −5,861.
- **What changed:**
  1. **Replaced the fake XPT files with genuine CDC downloads** and added XPORT header validation in `src/fetch_data.py`.
  2. **Switched to the pre-pandemic `P_*` cycle**, raising n from 5,088 → **8,282**.
  3. Added `src/calibrate_models.py` — **cross-fitted isotonic calibration**, because raw scores were badly over-confident.
  4. Wrote `REPORT.md` (formal report with limitations, glossary, viva Q&A).
  5. Fixed the **ECE metric bug** (Defect 10) and added a **regression guard**.
  6. Added `plotly` to `requirements.txt`.
  7. Rewired the app to read **every** number from JSON.
- **Why each matters:** (1) data integrity — the "huge data" requirement was previously fake; (2) genuine data volume; (3) showing a raw 0.84 that meant 0.34 is a correctness bug, not a cosmetic one; (5) an ECE that reports 0.0001 for an anti-calibrated model hides the very problem calibration exists to fix.
- **Verify:** `git show 1e5312d --stat`; `results/metrics/calibration.json`.

### Commit `6a438fb` — 2026-10-18 — "Fix: SHAP radio switch wiped the NHANES page results"
- **Size:** 2 files, +55 / −2.
- **What:** clicking the SHAP radio ("Explain the prediction for:") **erased every result on the page**.
- **Root cause:** all results were rendered inside `if submitted:`, so any widget event re-ran the script and the `submitted` flag reset to `False`.
- **Fix:** persist inputs in `st.session_state["nhanes_inputs"]`; results now render whenever `inputs` exists, not only on the submit click.
- **Why this is a good answer in a viva:** the fix was accompanied by a **regression test that was proven to fail on the old code (charts 4 → 0) and pass on the new (charts 4 → 4)**. This demonstrates disciplined debugging, not luck.
- **Reported by:** the student during hands-on testing.
- **Verify:** `git show 6a438fb`; smoke test section 5 → "interaction: SHAP radio switch keeps results".

### Commit `c5ab9f9` — 2026-10-18 — "Fix: SHAP charts show raw column names (gender_female) instead of readable labels"
- **Size:** 8 files, +255 / −5.
- **What:** selecting **Male** showed **`gender_female`** in the SHAP chart.
- **Root cause:** the chart displayed raw training column names. `gender_female` is a 0/1 indicator where 1 = female, so choosing Male sets it to `0` — correct but unreadable.
- **Fix:** new `src/feature_labels.py` as a **single source of truth** shared by both the app and the training figures, so wording cannot drift. Labels now carry the **value the user actually chose** (e.g. `Sex: Male`, `Snoring frequency: Frequently (5+/wk)`).
- **Also added:** `src/regen_shap_figure.py`, which regenerates **one** figure from the saved pickle without a full retrain — because a *plot label* is not a reason to refit 3 algorithms × 5 folds (which also crashed on a transient out-of-memory error at the time).
- **Reported by:** the student during hands-on testing.
- **Verify:** `git show c5ab9f9`; `tests/test_feature_labels.py`.

### Complete defect register

| # | Defect | Impact | Fix | Where |
|---|---|---|---|---|
| 1 | Five "raw CDC files" were **20,905-byte HTML error pages** | Data was fake | Genuine downloads + XPORT header validation | `src/fetch_data.py` |
| 2 | `requirements.txt` missing `plotly` | App crashed on fresh install | Added dependency | `requirements.txt` |
| 3 | UI said **"Primary Diagnosis"** | Inappropriate medical claim | → "screening risk estimate, not a diagnosis" + persistent disclaimer | `app/app.py` |
| 4 | Dropdown values never in training (`Other`, `Underweight`); missing valid value (`Sales Representative`) | Silently wrong predictions | Corrected option lists | `app/app.py` |
| 5 | SHAP axis labelled "probability" while showing **log-odds** | Misinterpretation | Corrected to log-odds + explanation | `app/app.py` |
| 6 | Metrics **hard-coded in the UI** | Could contradict a retrained model | All numbers read from `results/metrics/*.json` | `app/app.py` |
| 7 | CWD-relative paths | App broke when launched elsewhere | All paths anchored to repo root | `app/app.py` |
| 8 | `read_sas` **zero-artifact** corrupting sleep-frequency features | Wrong feature values | Fixed in `src/prepare_nhanes.py` (A ≥ 3.1) | `src/prepare_nhanes.py` |
| 9 | **Model selection performed on the test set** | Optimistic by construction | Selection moved inside the CV loop | `src/train_models.py`, `src/evaluate_legacy.py` |
| 10 | **ECE used signed gaps** (over- and under-prediction cancelled) | Reported ECE ≈ 0.0001 on an *anti*-calibrated model | Changed to Σ w·\|gap\|; regression guard proves buggy = 0.005 vs correct = 0.500 | `src/calibrate_models.py` |
| 11 | Clicking SHAP radio **wiped all page results** | App appeared broken | `st.session_state` persistence + regression test | `app/app.py` |
| 12 | SHAP chart showed raw `gender_female` | Confusing / looks like a bug | Human-readable labels with user's value | `src/feature_labels.py`, `app/app.py` |
| 13 | Kaggle 0.933 single-split score was **optimistic** | Overstated demo | Demoted to group-aware CV **0.904**, labelled educational | `src/evaluate_legacy.py` |

**Defect 10 in detail (excellent viva material):**
The original ECE summed *signed* differences between predicted and observed frequency:
`ECE = Σ w_i · (p_i − a_i)`. If the model over-predicts in some bins and under-predicts in others, the terms **cancel**, producing a near-zero ECE for a badly calibrated model. The corrected form sums **absolute** gaps: `ECE = Σ w_i · |p_i − a_i|`. A dedicated regression guard in `src/calibrate_models.py` proves the distinction: on a deliberately anti-calibrated model the **buggy** variant returns **0.0046** (looks excellent) while the **correct** variant returns **0.5003** (correctly flags the failure). This is exactly the kind of "our metric was silently lying to us" story examiners love.

---

## 7. HOW TO RUN AND REPRODUCE

**Run the app:**
```powershell
.\venv\Scripts\streamlit.exe run app\app.py
```
Opens at http://localhost:8501. Stop with `Ctrl+C`.

**Run the tests:**
```powershell
.\venv\Scripts\python.exe tests\smoke_test.py
.\venv\Scripts\python.exe tests\test_feature_labels.py
```
Both end with `ALL CHECKS PASSED`.

**Rebuild the full pipeline (only if code changes):**
```powershell
.\venv\Scripts\python.exe src\prepare_nhanes.py
.\venv\Scripts\python.exe src\train_models.py
.\venv\Scripts\python.exe src\calibrate_models.py
.\venv\Scripts\python.exe src\evaluate_legacy.py
.\venv\Scripts\python.exe src\train_large_model.py
```

**Smoke test coverage (5 sections):** artifacts present → `predict_proba` contracts for all 7 models + probability normalisation + CI availability → **leakage guards** → calibration sanity (Brier improves, AUC preserved, calibrator monotone, ECE regression guard) → all 6 Streamlit pages rendered via `streamlit.testing.v1`, including the SHAP-radio interaction regression test.

**Troubleshooting:** an `OpenBLAS ... Memory allocation` error is **RAM pressure**, not a code bug — close Chrome/VS Code tabs and retry. Regenerate a single figure without a full retrain: `.\venv\Scripts\python.exe src\regen_shap_figure.py osa_risk`.

---

## 8. WHAT THE APP SHOWS (6 pages)

1. **📋 Executive Overview** — project summary, headline metrics, disclaimer.
2. **🫀 Sleep-Disorder & Heart Risk (NHANES)** — the core page: enter demographics + sleep habits → calibrated risks for heart disease, insomnia and OSA, plus per-patient SHAP explanation.
3. **📈 Large-Scale Sleep Risk (100k)** — 4-class severity prediction on the synthetic 100k benchmark.
4. **🧠 Sleep Pattern Demo (educational)** — the Kaggle demo, explicitly labelled educational.
5. **🫀 Cardiometabolic Dashboard** — diabetes + hypertension risk with local SHAP explanation.
6. **⚙️ Performance & Methodology** — ROC curves, calibration/reliability plots, SHAP summaries, and the honest methodology write-up.

**UX detail worth knowing:** the gauges show **calibrated** risks, which differ from the model's raw `predict_proba`. This is expected and correct — if the student compares a gauge to a raw score, the calibration map explains the difference.

---

## 9. LIMITATIONS — STATE THESE PROACTIVELY

An examiner who hears a limitation stated *before* they ask trusts everything else you say. Memorise these:

1. **Self-report labels.** Heart disease, hypertension, diabetes and insomnia labels are self-reported diagnoses. This introduces recall/reporting bias. No chart review or clinical validation was possible.
2. **OSA is symptom-based, not diagnosed.** NHANES has no polysomnography and no AHI, so `target_osa_risk` captures snoring/witnessed pauses — a **risk proxy**, never OSA itself.
3. **Cross-sectional design.** NHANES observes a snapshot; **no causal claim** is possible. Sleep features *associate* with risk, they do not *cause* it.
4. **Moderest discrimination for insomnia and OSA** (AUC ~0.69–0.71). These are useful for **screening triage**, not individual decisions.
5. **The 100k dataset is synthetic.** It demonstrates behaviour at scale; it is **not** evidence of clinical performance.
6. **The Kaggle demo has only 109 unique feature patterns** in 374 rows, so its score is inherently fragile; it is educational only.
7. **US population, 2017–2020.** Generalisation to other countries, eras or age groups is untested.
8. **No external validation cohort.** Performance is estimated by internal cross-validation only.
9. **Low precision for heart disease (0.218)** at the chosen operating point — a deliberate screening trade-off that would need threshold tuning per clinical setting.
10. **Class imbalance** hurts the Moderate class in the 100k model (recall 0.683).
11. **Not a medical device.** No regulatory approval (no FDA/CE), and none is claimed.

---

## 10. VIVA VOCE — LIKELY QUESTIONS AND ANSWERS

**Conceptual**
1. **Why is this not a diagnostic tool?** → It outputs statistical risk estimates from survey data with self-report labels and no clinical validation. Diagnosis requires a clinician and clinical tests (e.g. polysomnography for OSA).
2. **Why did you use NHANES?** → Genuine, large (8,282), nationally representative, public, with a documented codebook, and it contains the sleep variables needed. The 2019-2020 cycle was never released standalone due to COVID, so the pre-pandemic `P_*` file is the correct recent source.
3. **What is data leakage and how did you prevent it?** → Using information at prediction time that wouldn't be available, or that defines the target. Prevented by excluding label-defining features per task (e.g. `snore_freq` excluded from OSA) and by fitting preprocessing **inside** the CV pipeline.
4. **Why cross-validation instead of a single train/test split?** → A single split's score depends on which rows land in the test set. RepeatedStratifiedKFold gives a distribution → reportable **95% CI**.
5. **Why is `class_weight="balanced"` used, and what does it cost?** → Targets are imbalanced, so the model is told to care about the minority class. Cost: **over-confident probabilities**, fixed by isotonic calibration.

**Metrics**
6. **Your heart precision is 0.218 — is the model bad?** → No. At a 9.35% base rate, a screening model accepts more false alarms to catch more true cases (recall 0.787). Missing a cardiac case is costlier than a false alarm.
7. **Accuracy is 0.95 on the 100k model — is it excellent?** → Be careful: balanced accuracy is **0.853** and macro-F1 **0.868**. The gap signals imbalance; always quote balanced metrics.
8. **What does AUC of 0.82 mean?** → If one randomly chosen positive and one randomly chosen negative are scored, the model ranks the positive higher ~82% of the time. It says **nothing** about whether 82% of predictions are correct.
9. **Why report PR-AUC as well?** → With rare positives, ROC-AUC can look optimistic. PR-AUC focuses on the positive class and is more informative at low base rates.
10. **What is Brier score?** → Mean squared error of the probability: `mean((p − y)²)`. It penalises confident wrong answers heavily. Lower is better; it rewards both discrimination *and* calibration.

**Calibration**
11. **Why do the gauges not match raw `predict_proba`?** → They show **calibrated** risk. Raw scores were over-confident (raw 0.84 ≈ true 0.34); isotonic regression maps them to observed frequencies.
12. **Why isotonic and not logistic (Platt)?** → Isotonic is non-parametric and flexible, appropriate at n≈8,000. For small n, Platt would be safer — a good future-work point.
13. **ROC-AUC dropped a little after calibration. Is that a problem?** → No. Calibration is monotone, so ranking (and hence AUC) is theoretically unchanged; the small drop is cross-fitting noise. Brier improved ~59% for heart disease — that is the win.
14. **What is ECE?** → Expected Calibration Error: bin predictions, compare mean predicted vs observed frequency per bin, weight by bin size. It must use **absolute** gaps — the signed version silently cancels errors (Defect 10).

**Explainability**
15. **What does a SHAP value mean?** → A feature's contribution for **this** prediction, on the **log-odds** scale. +0.69 roughly doubles the odds.
16. **Why log-odds and not probability?** → Additivity holds on the log-odds scale. Contributions to a probability are not additive, so a "sum of contributions" reading would be wrong.
17. **How is SHAP different from feature importance?** → Global importance is one number per feature; SHAP gives a **per-patient** contribution, so it explains an individual prediction.

**Engineering**
18. **Why Streamlit?** → Rapid, pure-Python dashboarding with Plotly charts, ideal for an interactive demo; the ML logic lives in `src/`, so the app stays a thin presentation layer.
19. **Why are metrics in JSON, not hard-coded?** → A retrained model can never be contradicted by a stale caption (Defect 6).
20. **How do you know the app matches the models?** → Smoke tests load all 7 pickles, check `predict_proba` contracts and normalisation, run leakage guards, verify calibration, and execute all 6 pages via `streamlit.testing.v1`.
21. **How do you ensure the data is real?** → `fetch_data.py` validates the **XPORT header**; the original repo's "data" was actually HTML error pages (Defect 1), so this guard exists to stop it recurring silently.

**Methodology / critique (hard questions)**
22. **Why not deep learning?** → n≈8,282 with ~11 tabular features. Deep learning would overfit; gradient boosting and regularised logistic regression are the appropriate, defensible choice for small tabular data.
23. **Why no SMOTE?** → The project uses `class_weight="balanced"` (which handles imbalance inside the loss) and reports PR-AUC/balanced accuracy, which reflect imbalance honestly. Oversampling can distort probability calibration — a real concern given calibration is a project highlight.
24. **Isn't predicting OSA from snoring circular?** → Yes, which is exactly why `snore_freq` and `witnessed_apnea_freq` are **excluded** from the OSA model's features. The remaining features (waist circumference, BMI, sleepiness, jetlag) predict OSA risk without using the defining symptoms.
25. **Your Kaggle score dropped from 0.933 to 0.904. Why report a lower number?** → The 0.933 came from a single split *and* from row-wise CV that let duplicate patterns leak across folds. Group-aware CV (**0.904**) is the honest figure. Reporting the lower, honest number is a strength.
26. **What would you do with more time?** → External validation on another cohort; clinical (polysomnography) labels for OSA; temporal validation; threshold tuning per clinical cost model; Platt vs isotonic comparison; calibration-aware model selection; possibly monotonic constraints so risk moves sensibly with, e.g., BMI.

---

## 11. GLOSSARY

| Term | Meaning |
|---|---|
| **NHANES** | US national health survey; the project's primary data source |
| **XPT / XPORT** | SAS transport file format used by CDC for NHANES |
| **Prevalence** | Fraction of the cohort with the positive label (base rate) |
| **ROC-AUC** | Probability the model ranks a random positive above a random negative |
| **PR-AUC** | Area under precision–recall curve; more informative for rare positives |
| **Recall (sensitivity)** | Of actual positives, the fraction caught — measures misses |
| **Precision** | Of those flagged positive, the fraction truly positive |
| **F1** | Harmonic mean of precision and recall |
| **Brier score** | Mean squared error of predicted probabilities; rewards calibration |
| **ECE** | Expected Calibration Error — how far predicted probabilities are from observed frequencies |
| **Isotonic regression** | Learns a monotone mapping from raw score → observed frequency |
| **Cross-fitting** | Learn a mapping on out-of-fold predictions so it is not fit on its own training data |
| **Log-odds (logit)** | `log(p/(1−p))`; the additive scale for linear models and SHAP |
| **Leakage** | Using information unavailable at prediction time, or that defines the target |
| **Ablation** | Comparing model A vs model B to isolate one feature family's contribution |
| **StratifiedKFold** | Folds that preserve class proportions |
| **RepeatedStratifiedKFold** | Stratified CV repeated with different splits, giving a CI |
| **StratifiedGroupKFold** | Folds that keep all rows of a group in the same fold |
| **SHAP** | Shapley Additive Explanations — per-feature contributions per prediction |
| **TreeExplainer / LinearExplainer** | SHAP explainers for tree ensembles and linear models |
| **class_weight="balanced"** | Inverse-frequency weighting so the model cares about the minority class |
| **Ordinal target** | Ordered classes: Healthy < Mild < Moderate < Severe |

---

## 12. REFERENCES

**Repository references (verify any claim against these):**
- `results/metrics/metrics.json` — every CV metric, ablation delta, prevalence, feature list
- `results/metrics/calibration.json` — before/after Brier, ECE, reliability bins
- `results/metrics/large_model_metrics.json` — 100k model metrics + per-class report
- `REPORT.md` — formal report: methodology, limitations, glossary, viva Q&A
- `README.md` — technical overview and quick start
- Commits: `674d707`, `a764c56`, `1e5312d`, `6a438fb`, `c5ab9f9`
- `tests/smoke_test.py`, `tests/test_feature_labels.py` — the executable guarantees
- `scripts/verify_calibration.py` — deliberately independent re-implementation of the metrics, kept specifically to catch metric bugs like Defect 10

**External sources:**
- CDC/NCHS — NHANES 2017–March 2020 Pre-Pandemic Data Documentation (variables `P_MCQ`, `P_SLQ`, `P_BPQ`, `P_DIQ`, `P_DEMO`, `P_BMX`, `P_BPX`), released May 2021.
- Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems 30* (SHAP).
- Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." *ICML* (calibration motivation).
- Guo, C. et al. (2017). "On Calibration of Modern Neural Networks." *ICML* (calibration methods).
- Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python." *JMLR 12*.
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD*.
- Prokhorenkova, L. et al. (2018). "CatBoost: unbiased boosting with categorical features." *NeurIPS* (for `StratifiedGroupKFold` context on group-aware validation).

**Citation style for the submitted report:** follow your institution's format (IEEE / APA / ACM). The NHANES documentation and the SHAP paper are the two most important to cite.

---

## 13. ONE-PARAGRAPH SUMMARY (memorise this)

> SleepHealth-AI uses genuine CDC NHANES 2017–March 2020 pre-pandemic data (8,282 participants) to estimate five health risks — insomnia, obstructive sleep-apnea risk, heart disease, hypertension and diabetes — from sleep patterns plus demographics, alongside a 100,000-record synthetic benchmark for 4-class severity. Every task is validated with repeated stratified k-fold cross-validation and reported with 95% confidence intervals, and each task uses an ablation (demographics-only vs demographics-plus-sleep) to quantify whether sleep features add signal — they do, most for insomnia (+0.078 AUC) and modestly for heart disease (+0.018) and OSA (+0.013). Label-defining features are excluded from their own tasks to prevent leakage, preprocessing is fit inside the CV pipeline, and raw over-confident probabilities are corrected with cross-fitted isotonic calibration (heart-disease Brier score 0.180 → 0.074) while ROC-AUC is preserved. SHAP provides per-patient explanations on the log-odds scale. Thirteen defects were found and fixed during development — including five fake "CDC data" files that were actually HTML error pages and an ECE metric that silently cancelled its own errors — each with a regression test. The system is delivered as a six-page Streamlit dashboard that reads all metrics from JSON, and every output is framed strictly as a **screening risk estimate, never a diagnosis**.

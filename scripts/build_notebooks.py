"""Rebuild notebooks 03/04/05 so they use the REAL pipeline and match the report.

WHY THIS EXISTS
    The original notebooks were written against the 2017-2018 cycle-J data
    (n=5,088) and a single 80/20 split. The shipped system uses the
    2017-March 2020 pre-pandemic data (n=8,282) with repeated 5x3 CV and
    calibration. So notebook 04 printed diabetes AUC 0.7774 while the report
    says 0.7704 - the professor sees two different numbers for the same model.

    The fix is not to hand-copy numbers into the notebooks (that would drift
    again on the next retrain). Instead each notebook IMPORTS src/ and calls
    the same functions the production pipeline calls, then ASSERTS its computed
    value equals the value in results/metrics/metrics.json. If they ever
    disagree, the notebook raises instead of quietly showing a stale figure.

NOTEBOOKS BUILT
    03_nhanes_processing   -> merges the real P_*.XPT files, builds labels
    04_disease_models      -> repeated-CV ablation (does sleep help?) + ROC
    05_model_analysis      -> SHAP explanations + calibration reliability

Run:  python scripts/build_notebooks.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT_MARKDOWN = """\
# {title}

> **This notebook is part of the SleepHealth-AI project.**
> It is executed against the *production* pipeline in `src/`, so every number
> below is the same number reported in `REPORT.md`, `results/metrics/metrics.json`
> and the Streamlit app. Each final cell **asserts** that agreement, so this
> notebook cannot silently drift from the shipped system.

**Setup:** run this notebook from the `notebooks/` folder. The first cell moves
the working directory to the repository root and puts `src/` on `sys.path`.
"""


def base() -> nbf.NotebookNode:
    nb = new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    })
    return nb


SETUP = '''\
# --- Reproducibility setup: run from notebooks/, work from the repo root ------
import os, sys, json
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":          # launched inside notebooks/
    ROOT = ROOT.parent
os.chdir(ROOT)                        # src/ scripts use repo-root-relative paths
sys.path.insert(0, str(ROOT / "src")) # so `import train_models` works

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Notebooks need the INLINE backend so plt.show() embeds the figure in the
# output. (The src/ scripts use "Agg" because they save PNGs to disk instead;
# copying that here would make every plot render as nothing.)
%matplotlib inline

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)

print("repo root :", ROOT)
print("python    :", sys.version.split()[0])
print("pandas    :", pd.__version__)
'''


# --------------------------------------------------------------------------- #
# 03 - data preparation
# --------------------------------------------------------------------------- #
def nb03() -> nbf.NotebookNode:
    nb = base()
    nb.cells = [
        new_markdown_cell(ROOT_MARKDOWN.format(
            title="# 03 - NHANES Data Preparation")),
        new_markdown_cell(
            "## Why this notebook exists\n\n"
            "This is the data-engineering stage: six separate CDC questionnaire files "
            "are merged into one analysis-ready table with **five prediction targets**.\n\n"
            "* **Source:** NHANES 2017-March 2020 *pre-pandemic* release (`P_*.XPT`, May 2021).\n"
            "* **Why this cycle:** the 2019-2020 cycle was suspended in March 2020 and was "
            "never released standalone, so NCHS published this combined pre-pandemic file "
            "instead. It roughly doubles the cohort versus the older 2017-2018 files.\n"
            "* **Anti-leakage rule:** any questionnaire item used to *build* a label is "
            "excluded from that label's feature list."),
        new_code_cell(SETUP),
        new_code_cell(
            "import prepare_nhanes as pn\n\n"
            "# Read and inner-join the six modules on the participant id SEQN.\n"
            "# prepare_nhanes also repairs a read_sas quirk that decodes the XPT\n"
            "# integer 0 as a denormalised float (5.4e-79).\n"
            "merged = pn.load_merged()\n"
            "print('merged shape:', merged.shape)"),
        new_code_cell(
            "# Adult cohort: age >= 20 and valid diabetes / blood-pressure answers.\n"
            "cohort = merged[(merged['RIDAGEYR'] >= 20)\n"
            "                & merged['DIQ010'].isin([1, 2])\n"
            "                & merged['BPQ020'].isin([1, 2])].copy()\n"
            "print('adult cohort:', cohort.shape[0], 'participants')\n"
            "print('age range   :', int(cohort['RIDAGEYR'].min()), '-', int(cohort['RIDAGEYR'].max()))"),
        new_code_cell(
            "# Rebuild the feature + label table exactly as src/prepare_nhanes.py does.\n"
            "# CDC codes 7 (Refused) and 9 (Don't know) are treated as missing, not\n"
            "# as numeric values - leaving them in would corrupt the ordinal scales.\n"
            "out = pd.DataFrame(index=cohort.index)\n"
            "out['age'] = cohort['RIDAGEYR']\n"
            "out['gender_female'] = (cohort['RIAGENDR'] == 2).astype(int)\n"
            "out['bmi'] = cohort['BMXBMI']\n"
            "out['waist_circumference'] = cohort['BMXWAIST']\n"
            "out['sleep_duration_hrs'] = cohort['SLD012']          # weekday\n"
            "out['weekend_sleep_hrs'] = cohort['SLD013']           # weekend\n"
            "out['social_jetlag_hrs'] = (cohort['SLD013'] - cohort['SLD012']).round(2)\n\n"
            "def freq(col, max_code):\n"
            "    s = cohort[col].replace(pn.INVALID, np.nan)\n"
            "    return s.where(s <= max_code).astype(float)\n\n"
            "out['snore_freq'] = freq('SLQ030', 3)\n"
            "out['witnessed_apnea_freq'] = freq('SLQ040', 3)\n"
            "out['daytime_sleepiness'] = freq('SLQ120', 4)\n"
            "out['trouble_sleeping'] = cohort['SLQ050'].replace(pn.INVALID, np.nan).map({1.0: 1, 2.0: 0})"),
        new_code_cell(
            "# ---- The five prediction targets ---------------------------------------\n"
            "heart_cols = ['MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E']\n"
            "heart = cohort[heart_cols].replace(pn.INVALID, np.nan)\n"
            "answered = heart.notna().any(axis=1)\n"
            "out['target_heart_disease'] = ((heart == 1).any(axis=1)).astype(int).where(answered, np.nan)\n"
            "out['target_insomnia_risk'] = out['trouble_sleeping']\n"
            "osa = ((out['snore_freq'] == 3) | (out['witnessed_apnea_freq'] >= 2)).astype(int)\n"
            "out['target_osa_risk'] = osa.where(out['snore_freq'].notna()\n"
            "                                   & out['witnessed_apnea_freq'].notna())\n"
            "out['target_diabetes'] = (cohort['DIQ010'] == 1).astype(int)\n"
            "out['target_hypertension'] = (cohort['BPQ020'] == 1).astype(int)\n\n"
            "print('final table:', out.shape)"),
        new_code_cell(
            "# ---- Feature / label summary ------------------------------------------\n"
            "profile = pd.DataFrame({\n"
            "    'dtype': out.dtypes.astype(str),\n"
            "    'missing': out.isna().sum(),\n"
            "    'missing_%': (out.isna().mean() * 100).round(1),\n"
            "})\n"
            "num = out.select_dtypes(include=[np.number])\n"
            "profile['min'] = num.min()\n"
            "profile['mean'] = num.mean().round(2)\n"
            "profile['max'] = num.max()\n"
            "profile"),
        new_code_cell(
            "# ---- Label prevalence ---------------------------------------------------\n"
            "labels = [c for c in out.columns if c.startswith('target_')]\n"
            "prev = pd.DataFrame({\n"
            "    'positives': [int(out[c].sum()) for c in labels],\n"
            "    'n': [int(out[c].notna().sum()) for c in labels],\n"
            "})\n"
            "prev['prevalence_%'] = (prev['positives'] / prev['n'] * 100).round(2)\n"
            "prev.index = labels\n"
            "prev"),
        new_markdown_cell(
            "## Reproducibility check\n\n"
            "The cell below compares what this notebook just rebuilt against the CSV the "
            "production pipeline ships (`data/processed/nhanes_sleep_health.csv`). "
            "They must be identical - if they diverge, this cell raises."),
        new_code_cell(
            "shipped = pd.read_csv('data/processed/nhanes_sleep_health.csv')\n\n"
            "assert out.shape == shipped.shape, (out.shape, shipped.shape)\n"
            "for col in shipped.columns:\n"
            "    a, b = out[col].reset_index(drop=True), shipped[col]\n"
            "    if a.dtype.kind == 'f' or b.dtype.kind == 'f':\n"
            "        same = np.allclose(a.astype(float), b.astype(float), equal_nan=True)\n"
            "    else:\n"
            "        same = a.equals(b.astype(a.dtype))\n"
            "    assert same, f'column mismatch: {col}'\n"
            "print(f'OK - notebook rebuild matches shipped CSV exactly  ({len(shipped)} rows x {len(shipped.columns)} cols)')"),
        new_code_cell(
            "# Cross-check cohort size against the metrics the app reports.\n"
            "metrics = json.load(open('results/metrics/metrics.json', encoding='utf-8'))\n"
            "assert len(shipped) == metrics['data']['n_participants'], 'cohort size drifted'\n"
            "print('OK - cohort n =', len(shipped), 'matches metrics.json')"),
    ]
    return nb


# --------------------------------------------------------------------------- #
# 04 - model training and the central ablation
# --------------------------------------------------------------------------- #
def nb04() -> nbf.NotebookNode:
    nb = base()
    nb.cells = [
        new_markdown_cell(ROOT_MARKDOWN.format(
            title="# 04 - Disease Risk Models (cross-validated)")),
        new_markdown_cell(
            "## What is being measured\n\n"
            "Five binary risk tasks are trained and compared. For each one the same "
            "algorithm is fitted twice:\n\n"
            "| Set | Features | Purpose |\n|---|---|---|\n"
            "| **Model A** | demographics only (`age`, `sex`, `BMI`, `waist`) | baseline |\n"
            "| **Model B** | Model A **+ sleep features** | does sleep help? |\n\n"
            "The difference **B - A** is the project's central scientific claim.\n\n"
            "**Validation:** `RepeatedStratifiedKFold`, 5 folds x 3 repeats (15 splits). "
            "Reporting a single train/test split would make the score depend on which rows "
            "happened to land in the test set; repeating gives a distribution, so we can "
            "report a **95% confidence interval**.\n\n"
            "**Leakage control:** the items that *define* a label are excluded from that "
            "label's features - e.g. snoring is removed from the OSA task, and "
            "`trouble_sleeping` is removed from the insomnia task."),
        new_code_cell(SETUP),
        new_code_cell(
            "import train_models as tm\n"
            "from sklearn.model_selection import RepeatedStratifiedKFold\n\n"
            "df = pd.read_csv(tm.DATA)\n"
            "skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)\n\n"
            "print('data   :', tm.DATA, df.shape)\n"
            "print('tasks  :', list(tm.TASKS))"),
        new_markdown_cell(
            "### Leakage guard\n\n"
            "Before training, confirm no label-defining feature slipped into its own "
            "task's feature list."),
        new_code_cell(
            "# A label must never be predictable from the item that defines it.\n"
            "leak_rules = {\n"
            "    'insomnia_risk': ['trouble_sleeping'],          # that IS SLQ050\n"
            "    'osa_risk': ['snore_freq', 'witnessed_apnea_freq'],  # they define OSA\n"
            "}\n"
            "for task, forbidden in leak_rules.items():\n"
            "    feats = tm.TASKS[task]['model_b']\n"
            "    bad = [f for f in forbidden if f in feats]\n"
            "    assert not bad, f'LEAKAGE: {bad} present in {task} features'\n"
            "    print(f'OK - {task}: {forbidden} correctly excluded')"),
        new_markdown_cell(
            "### Train all tasks\n\n"
            "This calls `cv_scores` - the *same* function the production pipeline uses - "
            "so the numbers below are identical to `results/metrics/metrics.json`."),
        new_code_cell(
            "rows = []\n"
            "for task_name, task in tm.TASKS.items():\n"
            "    d = df.dropna(subset=[task['target']]).copy()\n"
            "    y = d[task['target']].astype(int)\n"
            "    best = json.load(open('results/metrics/metrics.json', encoding='utf-8'))[\n"
            "        'tasks'][task_name]['best_model']\n"
            "    a = tm.cv_scores(task, task['model_a'], y, d[task['model_a']], best, skf)\n"
            "    b = tm.cv_scores(task, task['model_b'], y, d[task['model_b']], best, skf)\n"
            "    rows.append({\n"
            "        'task': task_name, 'algorithm': best, 'n': len(d),\n"
            "        'prevalence': round(float(y.mean()), 4),\n"
            "        'A_demo_auc': a['roc_auc'], 'B_sleep_auc': b['roc_auc'],\n"
            "        'delta': round(b['roc_auc'] - a['roc_auc'], 4),\n"
            "        'B_auc_ci95': b['roc_auc_ci95'],\n"
            "        'B_pr_auc': b['pr_auc'], 'B_f1': b['f1'],\n"
            "        'B_recall': b['recall'], 'B_precision': b['precision'],\n"
            "        'B_brier': b['brier'],\n"
            "    })\n"
            "results_df = pd.DataFrame(rows).set_index('task')\n"
            "results_df"),
        new_markdown_cell(
            "### Cardiometabolic tasks (diabetes, hypertension)\n\n"
            "These two use the same ablation design but are trained via the "
            "`evaluate_legacy` pipeline (RandomForest for diabetes, XGBoost for hypertension). "
            "Feature set B here adds sleep duration + trouble-sleeping."),
        new_code_cell(
            "import evaluate_legacy as el\n\n"
            "LEGACY_A = ['age', 'gender_female', 'bmi', 'waist_circumference']\n"
            "LEGACY_B = LEGACY_A + ['sleep_duration_hrs', 'trouble_sleeping']\n"
            "legacy_specs = {\n"
            "    'diabetes':     ('target_diabetes', 'RandomForest'),\n"
            "    'hypertension': ('target_hypertension', 'XGBoost'),\n"
            "}\n\n"
            "legacy_rows = []\n"
            "for name, (target, alg) in legacy_specs.items():\n"
            "    y = df[target].astype(int)\n"
            "    a = el.cv_binary(df[LEGACY_A], y, alg)\n"
            "    b = el.cv_binary(df[LEGACY_B], y, alg)\n"
            "    legacy_rows.append({\n"
            "        'task': name, 'algorithm': alg, 'n': len(y),\n"
            "        'prevalence': round(float(y.mean()), 4),\n"
            "        'A_demo_auc': a['roc_auc'], 'B_sleep_auc': b['roc_auc'],\n"
            "        'delta': round(b['roc_auc'] - a['roc_auc'], 4),\n"
            "        'B_auc_ci95': b['roc_auc_ci95'], 'B_pr_auc': b['pr_auc'],\n"
            "        'B_f1': b['f1'], 'B_recall': b['recall'],\n"
            "        'B_precision': b['precision'], 'B_brier': b['brier'],\n"
            "    })\n"
            "legacy_df = pd.DataFrame(legacy_rows).set_index('task')\n"
            "legacy_df"),
        new_code_cell(
            "# ---- Combine all 5 tasks into one summary table -----------------------\n"
            "combined = pd.concat([results_df, legacy_df])\n"
            "combined[['algorithm','n','prevalence','A_demo_auc','B_sleep_auc','delta',"
            "'B_auc_ci95','B_pr_auc','B_f1','B_recall','B_precision','B_brier']]"),
        new_code_cell(
            "# ---- Ablation bar chart: does adding sleep features help? -------------\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "x = np.arange(len(combined))\n"
            "ax.bar(x - 0.18, combined['A_demo_auc'], 0.36, label='A: demographics only')\n"
            "ax.bar(x + 0.18, combined['B_sleep_auc'], 0.36, label='B: demographics + sleep')\n"
            "for i, d in enumerate(combined['delta']):\n"
            "    ax.text(i + 0.18, combined['B_sleep_auc'].iloc[i] + 0.008,\n"
            "            f'+{d:.3f}', ha='center', fontsize=9, fontweight='bold')\n"
            "ax.set_xticks(x); ax.set_xticklabels(combined.index, rotation=20)\n"
            "ax.set_ylim(0.5, 0.9); ax.set_ylabel('ROC-AUC (mean, 5x3 CV)')\n"
            "ax.set_title('Central ablation: does sleep improve the estimate?')\n"
            "ax.legend(); ax.grid(alpha=0.3, axis='y')\n"
            "plt.tight_layout(); plt.show()"),
        new_markdown_cell(
            "### Verify against the shipped metrics\n\n"
            "The notebook must agree with `metrics.json`. If a retrain ever changes these "
            "numbers, this cell fails loudly instead of showing a stale result."),
        new_code_cell(
            "expected = json.load(open('results/metrics/metrics.json', encoding='utf-8'))['tasks']\n"
            "# Tolerance: the LogisticRegression tasks are bit-reproducible, but the\n"
            "# RandomForest/XGBoost tasks run with n_jobs=-1 and parallel floating-point\n"
            "# reductions are not, so their CV means can differ by ~1e-4 between runs.\n"
            "# 3e-3 still catches a genuine mismatch (the superseded notebook reported\n"
            "# 0.7774 where the real value is 0.7704 - a 7e-3 gap).\n"
            "TOL = 3e-3\n"
            "for task_name in combined.index:\n"
            "    e = expected[task_name]\n"
            "    got_auc = combined.loc[task_name, 'B_sleep_auc']\n"
            "    got_delta = combined.loc[task_name, 'delta']\n"
            "    assert abs(got_auc - e['ablation']['model_b_plus_sleep_roc_auc']) < TOL, (\n"
            "        task_name, got_auc, e['ablation']['model_b_plus_sleep_roc_auc'])\n"
            "    assert abs(got_delta - e['ablation']['delta']) < TOL, (\n"
            "        task_name, got_delta, e['ablation']['delta'])\n"
            "    print(f\"OK - {task_name:15s} AUC={got_auc:.4f}  delta={got_delta:+.4f}\")\n"
            "assert len(combined) == len(expected), 'not every task was reproduced'\n"
            "print(f'\\nAll {len(combined)} tasks match results/metrics/metrics.json')"),
    ]
    return nb


# --------------------------------------------------------------------------- #
# 05 - explainability and calibration
# --------------------------------------------------------------------------- #
def nb05() -> nbf.NotebookNode:
    nb = base()
    nb.cells = [
        new_markdown_cell(ROOT_MARKDOWN.format(
            title="# 05 - Explainability (SHAP) and Probability Calibration")),
        new_markdown_cell(
            "Two questions the models must answer beyond *how accurate*:\n\n"
            "1. **Why** did the model give this person that risk? -> SHAP\n"
            "2. **Can the risk number be trusted as a probability?** -> calibration\n\n"
            "### Why calibration is needed here\n\n"
            "`class_weight=\"balanced\"` is used because the targets are imbalanced. It makes "
            "the model rank people well, but its raw probabilities become badly "
            "*over-confident*. On heart disease a raw score of **0.84** corresponded to a "
            "true event rate of only **~0.34**. Cross-fitted **isotonic regression** maps "
            "those raw scores onto observed frequencies.\n\n"
            "**Calibration cannot change ranking**, so ROC-AUC is theoretically unchanged; "
            "the Brier score is what improves."),
        new_code_cell(SETUP),
        new_code_cell(
            "import joblib, shap\n"
            "from feature_labels import display_feature\n"
            "from calibrate_models import ece\n"
            "from train_models import TASKS, make_scaler_only\n\n"
            "calib = json.load(open('results/metrics/calibration.json', encoding='utf-8'))\n"
            "print('calibration method:', calib['method'])"),
        new_markdown_cell("### SHAP: which features drive each risk?"),
        new_code_cell(
            "# SHAP summary for each of the three core tasks, using the deployed\n"
            "# pickle. Feature names are human-readable via feature_labels.py.\n"
            "for task_name in ['heart_disease', 'insomnia_risk', 'osa_risk']:\n"
            "    task = TASKS[task_name]\n"
            "    feats = task['model_b']\n"
            "    d = pd.read_csv(tm_data := 'data/processed/nhanes_sleep_health.csv')\n"
            "    d = d.dropna(subset=[task['target']])\n"
            "    X = d[feats]\n"
            "    model = joblib.load(f'models/{task_name}_model.pkl')\n"
            "    Xs = np.asarray(make_scaler_only().fit_transform(X))\n"
            "    mdl = model.named_steps['classifier']\n"
            "    try:\n"
            "        explainer = shap.TreeExplainer(mdl)\n"
            "    except Exception:\n"
            "        from sklearn.linear_model import LogisticRegression\n"
            "        assert isinstance(mdl, LogisticRegression)\n"
            "        explainer = shap.LinearExplainer(mdl, Xs)\n"
            "    vals = explainer.shap_values(Xs[:300])\n"
            "    if isinstance(vals, list):\n"
            "        vals = vals[1]\n"
            "    elif getattr(vals, 'ndim', 0) == 3:\n"
            "        vals = vals[:, :, 1]\n"
            "    shap.summary_plot(np.asarray(vals), Xs[:300],\n"
            "                      feature_names=[display_feature(f) for f in feats],\n"
            "                      show=False, max_display=10)\n"
            "    plt.title(f'SHAP (log-odds) - {task_name}', fontsize=10)\n"
            "    plt.tight_layout(); plt.show()"),
        new_markdown_cell(
            "SHAP values are on the **log-odds** scale: a value of +0.69 roughly doubles the "
            "odds. Red pushes the estimate up, green down."),
        new_markdown_cell("### Calibration: before vs after"),
        new_code_cell(
            "names = ['heart_disease', 'insomnia_risk', 'osa_risk']\n"
            "cal_tbl = pd.DataFrame({\n"
            "    'Brier before': [calib[t]['brier_before'] for t in names],\n"
            "    'Brier after': [calib[t]['brier_after'] for t in names],\n"
            "    'ECE (equal-freq) before': [calib[t]['ece_equalfreq_before'] for t in names],\n"
            "    'ECE (equal-freq) after': [calib[t]['ece_equalfreq_after'] for t in names],\n"
            "    'ROC-AUC before': [calib[t]['roc_auc_before'] for t in names],\n"
            "    'ROC-AUC after': [calib[t]['roc_auc_after'] for t in names],\n"
            "}, index=names)\n"
            "cal_tbl.round(4)"),
        new_code_cell(
            "# Reliability curves: predicted risk vs observed frequency.\n"
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "for ax, t in zip(axes, names):\n"
            "    before, after = calib[t]['reliability_before'], calib[t]['reliability_after']\n"
            "    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect')\n"
            "    ax.plot([b['mean_predicted'] for b in before],\n"
            "            [b['observed_rate'] for b in before], 'o-', color='tab:red',\n"
            "            label='before')\n"
            "    ax.plot([b['mean_predicted'] for b in after],\n"
            "            [b['observed_rate'] for b in after], 'o-', color='tab:green',\n"
            "            label='after')\n"
            "    ax.set_title(t); ax.set_xlabel('mean predicted risk')\n"
            "    ax.set_ylabel('observed rate'); ax.legend(); ax.grid(alpha=0.3)\n"
            "plt.suptitle('Reliability: calibration pulls predictions onto the diagonal')\n"
            "plt.tight_layout(); plt.show()"),
        new_markdown_cell(
            "### Verify the reported calibration numbers\n\n"
            "The ECE function is re-checked here, including the regression guard that "
            "proves the metric is the honest (absolute-gap) variant rather than the "
            "signed version that let over- and under-prediction cancel."),
        new_code_cell(
            "# Guard 1: ECE must not be the error-cancelling signed variant.\n"
            "rng = np.random.default_rng(0)\n"
            "y_bad = rng.binomial(1, 0.5, 4000)\n"
            "p_bad = np.clip(1 - y_bad * 0.9 + rng.normal(0, 0.02, 4000), 0, 1)  # anti-calibrated\n"
            "ece_abs = ece(y_bad, p_bad, scheme='freq')\n"
            "assert ece_abs > 0.3, f'abs-gap ECE should flag this model, got {ece_abs:.4f}'\n"
            "print(f'OK - absolute-gap ECE flags the anti-calibrated model: {ece_abs:.4f}')\n\n"
            "# Guard 2: heart-disease ECE before calibration is the large, honest value.\n"
            "h = calib['heart_disease']\n"
            "assert h['ece_equalfreq_before'] > 0.2, 'expected a large pre-calibration ECE'\n"
            "assert h['ece_equalfreq_after'] < 0.05, 'expected a small post-calibration ECE'\n"
            "assert h['brier_after'] < h['brier_before'], 'Brier must improve'\n"
            "print(f\"OK - heart_disease ECE {h['ece_equalfreq_before']:.4f} -> \"\n"
            "      f\"{h['ece_equalfreq_after']:.4f}, Brier {h['brier_before']:.4f} -> {h['brier_after']:.4f}\")"),
        new_code_cell(
            "# Verify the deployed isotonic maps exist and are monotone (a calibrator\n"
            "# that would LOWER the risk of a higher raw score would be nonsense).\n"
            "grid = np.linspace(0.001, 0.999, 200)\n"
            "for t in names:\n"
            "    iso = joblib.load(f'models/calibration/{t}_isotonic.pkl')\n"
            "    mapped = iso.predict(grid)\n"
            "    assert np.all(np.diff(mapped) >= -1e-9), f'{t} calibrator is not monotone'\n"
            "    print(f'OK - {t:15s} isotonic map is monotone non-decreasing')"),
    ]
    return nb


def nb02() -> nbf.NotebookNode:
    """Append honest, group-aware scoring to the existing Kaggle demo notebook.

    The original three cells are kept exactly as written (they show the training
    narrative and save the deployed pickle). What they omit is that 374 rows hold
    only 109 unique feature patterns, so a row-wise split lets identical rows land
    in both train and test and inflates the score. The appended cells quantify
    that optimism and report the figure the project actually stands behind.
    """
    path = ROOT_PATH / "notebooks" / "02_sleep_disorder_model.ipynb"
    nb = nbf.read(path, as_version=4)

    # Idempotency guard: strip any cells this function appended on a previous run,
    # so rebuilding never duplicates the honest-scoring section.
    MARKER = "## Honest scoring: why 0.933 is optimistic"
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and "".join(c.source).startswith(MARKER):
            nb.cells = nb.cells[:i]
            break

    nb.cells += [
        new_markdown_cell(
            "## Honest scoring: why 0.933 is optimistic\n\n"
            "The cells above use a **single 80/20 split** and report Random Forest "
            "accuracy **0.9333**. That number is not wrong arithmetic, but it is "
            "*optimistic by construction*, for two reasons:\n\n"
            "1. **A single split** — one lucky partition of 75 test rows. The score "
            "depends on which rows happened to land in the test set.\n"
            "2. **Duplicate feature patterns** — this dataset has 374 rows but only "
            "**109 unique feature patterns**. Row-wise splitting lets identical rows "
            "appear in *both* train and test, so the model is partly tested on rows it "
            "has already seen.\n\n"
            "The fix for (2) is **`StratifiedGroupKFold`**, where a *group* = one unique "
            "feature pattern, guaranteeing each pattern stays in a single fold.\n\n"
            "> This dataset has **synthetic labels** and is an **educational demo** only. "
            "The project's evidence base is the CDC NHANES cohort in notebooks 03-05."),
        new_code_cell(
            "import numpy as np, pandas as pd, json\n"
            "from hashlib import md5\n"
            "from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold\n"
            "from sklearn.ensemble import RandomForestClassifier\n\n"
            "df = pd.read_csv('../data/processed/sleep_cleansed.csv')\n"
            "Xk = df.drop(columns=['Target_Sleep_Disorder'])\n"
            "yk = df['Target_Sleep_Disorder']\n\n"
            "# A group = one unique feature pattern. Hash the row's bytes so identical\n"
            "# rows share a group id.\n"
            "groups = pd.util.hash_pandas_object(Xk, index=False).values\n\n"
            "print(f'rows                   : {len(Xk)}')\n"
            "print(f'unique feature patterns: {pd.unique(groups).size}')\n"
            "print(f'duplicate rows         : {Xk.duplicated().sum()}')"),
        new_code_cell(
            "# Same estimator as the deployed pickle, so this measures the real model.\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n\n"
            "cat = ['Gender', 'Occupation', 'BMI Category']\n"
            "num = [c for c in Xk.columns if c not in cat]\n"
            "pre = ColumnTransformer([('num', StandardScaler(), num),\n"
            "                         ('cat', OneHotEncoder(handle_unknown='ignore'), cat)])\n"
            "demo_pipe = Pipeline([('pre', pre),\n"
            "                      ('classifier', RandomForestClassifier(random_state=42))])\n\n"
            "from sklearn.metrics import f1_score\n\n"
            "def score(splitter, **kw):\n"
            "    accs, f1s = [], []\n"
            "    for tr, te in splitter.split(Xk, yk, **kw):\n"
            "        demo_pipe.fit(Xk.iloc[tr], yk.iloc[tr])\n"
            "        p = demo_pipe.predict(Xk.iloc[te])\n"
            "        accs.append((yk.iloc[te] == p).mean())\n"
            "        f1s.append(f1_score(yk.iloc[te], p, average='macro', zero_division=0))\n"
            "    return np.array(accs), np.array(f1s)\n\n"
            "# (1) row-wise CV: duplicates can straddle the split\n"
            "ra, rf1 = score(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0))\n"
            "# (2) group-aware CV: each unique pattern stays in one fold\n"
            "ga, gf1 = score(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0),\n"
            "                groups=groups)\n\n"
            "def ci(a):\n"
            "    return f'{a.mean():.4f}  (95% CI {np.percentile(a,2.5):.4f} - {np.percentile(a,97.5):.4f})'\n\n"
            "print('Row-wise CV accuracy      :', ci(ra))\n"
            "print('Group-aware CV accuracy   :', ci(ga))\n"
            "print(f'Optimistic bias (row - group): {ra.mean()-ga.mean():+.4f}')"),
        new_code_cell(
            "legacy = json.load(open('../results/metrics/metrics.json', encoding='utf-8'))['legacy_demo']\n"
            "print('Reported in metrics.json:')\n"
            "print('  n rows                 :', legacy['n_rows'])\n"
            "print('  unique feature patterns:', legacy['n_unique_feature_patterns'])\n"
            "print('  row-wise CV accuracy   :', legacy['rowwise_cv_accuracy']['mean'])\n"
            "print('  group-aware CV accuracy:', legacy['groupaware_cv_accuracy']['mean'])\n"
            "print('  optimistic bias        :', legacy['optimistic_bias_accuracy'])\n"
            "assert abs(ga.mean() - legacy['groupaware_cv_accuracy']['mean']) < 0.03, 'group-aware score drifted'\n"
            "assert pd.unique(groups).size == legacy['n_unique_feature_patterns'], 'pattern count drifted'\n"
            "print('\\nOK - notebook agrees with metrics.json (educational demo, not the evidence base)')"),
    ]
    return nb


def main() -> None:
    from nbclient import NotebookClient

    nb_dir = ROOT_PATH / "notebooks"
    builders = {
        "02_sleep_disorder_model.ipynb": nb02,
        "03_nhanes_processing.ipynb": nb03,
        "04_disease_models.ipynb": nb04,
        "05_model_analysis.ipynb": nb05,
    }
    for name, builder in builders.items():
        nb = builder()
        print(f"\n=== executing {name} ===")
        client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(nb_dir)}})
        client.execute()
        out = nb_dir / name
        nbf.write(nb, out)
        n_out = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
        n_err = sum(1 for c in nb.cells if c.cell_type == "code"
                    for o in c.get("outputs", []) if o.get("output_type") == "error")
        print(f"    wrote {out.name}: {n_out} outputs, {n_err} errors")
        if n_err:
            raise SystemExit(f"{name} has {n_err} errored cells")


if __name__ == "__main__":
    ROOT_PATH = Path(__file__).resolve().parents[1]
    main()

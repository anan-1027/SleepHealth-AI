"""
Trains and cross-validates the NHANES sleep-disorder / cardiovascular models.

Tasks
-----
  heart_disease   -> target_heart_disease   (CHD / CHF / angina / MI, self-report)
  insomnia_risk   -> target_insomnia_risk   (SLQ050 trouble sleeping)
  osa_risk        -> target_osa_risk        (frequent snoring OR witnessed apnea)

Methodology
-----------
  * RepeatedStratifiedKFold (5 folds x 3 repeats) -> mean, std and 95% CI
  * Three algorithms compared per task: LogisticRegression, RandomForest, XGBoost
  * Metrics: ROC-AUC, PR-AUC, F1, recall, precision, accuracy, Brier (calibration)
  * Ablation (Model A: demographics  vs  Model B: + sleep) for every task
  * Winning algorithm is refit on the FULL dataset and deployed to models/
  * All numbers written to results/metrics/metrics.json, figures to results/figures/

Anti-leakage: the questionnaire items that define a label never appear in
that label's feature list.
"""
import json
import os
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from feature_labels import display_feature

warnings.filterwarnings("ignore")

DATA = os.path.join("data", "processed", "nhanes_sleep_health.csv")
MODELS = "models"
FIGURES = os.path.join("results", "figures")
METRICS = os.path.join("results", "metrics", "metrics.json")

DEMO = ["age", "gender_female", "bmi", "waist_circumference"]
SLEEP = ["sleep_duration_hrs", "weekend_sleep_hrs", "social_jetlag_hrs",
         "snore_freq", "witnessed_apnea_freq", "daytime_sleepiness", "trouble_sleeping"]

TASKS = {
    "heart_disease": {
        "target": "target_heart_disease",
        "label": "Heart disease (self-reported CHD / CHF / angina / heart attack)",
        "definition": ("Composite of MCQ160B (congestive heart failure), MCQ160C (coronary "
                       "heart disease), MCQ160D (angina) and MCQ160E (heart attack) == 1. "
                       "Source: CDC NHANES 2017-March 2020 pre-pandemic P_MCQ."),
        "model_a": DEMO,
        "model_b": DEMO + SLEEP,
    },
    "insomnia_risk": {
        "target": "target_insomnia_risk",
        "label": "Insomnia risk (ever told a doctor about trouble sleeping)",
        "definition": ("SLQ050 == 1 (Yes). Source: CDC NHANES 2017-March 2020 pre-pandemic P_SLQ. "
                       "Label item is excluded from the features."),
        "model_a": DEMO,
        "model_b": DEMO + ["sleep_duration_hrs", "weekend_sleep_hrs", "social_jetlag_hrs",
                            "snore_freq", "witnessed_apnea_freq", "daytime_sleepiness"],
    },
    "osa_risk": {
        "target": "target_osa_risk",
        "label": "Obstructive sleep-apnea risk (symptom based)",
        "definition": ("SLQ030 == 3 (snoring >= 5 nights/week) OR SLQ040 >= 2 "
                       "(witnessed breathing pauses >= 3-4 nights/week). Source: CDC NHANES "
                       "2017-March 2020 pre-pandemic P_SLQ. Defining items are excluded from the features."),
        "model_a": DEMO,
        "model_b": DEMO + ["sleep_duration_hrs", "weekend_sleep_hrs", "social_jetlag_hrs",
                            "daytime_sleepiness", "trouble_sleeping"],
    },
}


def make_model(name: str) -> Pipeline:
    if name == "LogisticRegression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    elif name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    else:  # XGBoost
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                            random_state=42, n_jobs=-1)
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("classifier", clf)])


def make_scaler_only() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])


def cv_scores(task: dict, features: list, y: pd.Series, X: pd.DataFrame,
              alg: str, skf: RepeatedStratifiedKFold) -> dict:
    out = {m: [] for m in ["roc_auc", "pr_auc", "f1", "recall", "precision", "accuracy", "brier"]}
    for tr, te in skf.split(X, y):
        mdl = make_model(alg)
        mdl.fit(X.iloc[tr], y.iloc[tr])
        prob = mdl.predict_proba(X.iloc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        yt = y.iloc[te]
        out["roc_auc"].append(roc_auc_score(yt, prob))
        out["pr_auc"].append(average_precision_score(yt, prob))
        out["f1"].append(f1_score(yt, pred, zero_division=0))
        out["recall"].append(recall_score(yt, pred, zero_division=0))
        out["precision"].append(precision_score(yt, pred, zero_division=0))
        out["accuracy"].append(accuracy_score(yt, pred))
        out["brier"].append(brier_score_loss(yt, prob))
    res = {}
    for k, v in out.items():
        a = np.asarray(v, dtype=float)
        res[k] = round(float(a.mean()), 4)
        res[k + "_std"] = round(float(a.std()), 4)
        res[k + "_ci95"] = [round(float(np.percentile(a, 2.5)), 4),
                            round(float(np.percentile(a, 97.5)), 4)]
    return res


def main() -> None:
    t0 = time.time()
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS), exist_ok=True)

    df = pd.read_csv(DATA)
    print(f"Loaded {DATA}  shape={df.shape}")
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    ALGS = ["LogisticRegression", "RandomForest", "XGBoost"]

    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {
            "file": DATA.replace("\\", "/"),
            "source": "CDC NHANES 2017-March 2020 pre-pandemic (P_*.XPT, released May 2021)",
            "n_participants": int(len(df)),
            "design": "RepeatedStratifiedKFold 5 folds x 3 repeats (no single hold-out split)",
            "note": "Deployed models are refit on the full dataset after cross-validated "
                    "performance estimation. Outputs are screening-risk estimates, not diagnoses.",
        },
        "tasks": {},
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for task_name, task in TASKS.items():
        print(f"\n=== {task_name} ===")
        d = df.dropna(subset=[task["target"]]).copy()
        y = d[task["target"]].astype(int)
        entry = {
            "label": task["label"],
            "definition": task["definition"],
            "n": int(len(d)),
            "prevalence": round(float(y.mean()), 4),
            "majority_baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
            "features_model_a": task["model_a"],
            "features_model_b": task["model_b"],
            "cv": {},
            "ablation": {},
        }

        for split_name, feats in (("model_a", task["model_a"]), ("model_b", task["model_b"])):
            for alg in ALGS:
                t = time.time()
                s = cv_scores(task, feats, y, d[feats], alg, skf)
                entry["cv"].setdefault(alg, {})[split_name] = s
                print(f"  {alg:20s} [{split_name}] ROC-AUC={s['roc_auc']:.4f} "
                      f"PR-AUC={s['pr_auc']:.4f} F1={s['f1']:.4f} "
                      f"({time.time() - t:.1f}s)")

        # Ablation at the best algorithm (by Model B ROC-AUC)
        best = max(ALGS, key=lambda a: entry["cv"][a]["model_b"]["roc_auc"])
        a_auc = entry["cv"][best]["model_a"]["roc_auc"]
        b_auc = entry["cv"][best]["model_b"]["roc_auc"]
        entry["best_model"] = best
        entry["ablation"] = {
            "algorithm": best,
            "model_a_demographics_roc_auc": a_auc,
            "model_b_plus_sleep_roc_auc": b_auc,
            "delta": round(b_auc - a_auc, 4),
            "delta_pr_auc": round(entry["cv"][best]["model_b"]["pr_auc"]
                                  - entry["cv"][best]["model_a"]["pr_auc"], 4),
        }
        entry["metrics_model_b"] = entry["cv"][best]["model_b"]
        results["tasks"][task_name] = entry

        # ---- ROC figure for this task -------------------------------------
        plt.figure(figsize=(7, 6))
        for alg in ALGS:
            # mean ROC curve over folds for the figure
            fprs, tprs = [], []
            probs_true = []
            for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42).split(d[task["model_b"]], y):
                m = make_model(alg)
                m.fit(d[task["model_b"]].iloc[tr], y.iloc[tr])
                p = m.predict_proba(d[task["model_b"]].iloc[te])[:, 1]
                f, t, _ = roc_curve(y.iloc[te], p)
                fprs.append(np.interp(np.linspace(0, 1, 100), f, t))
                probs_true.append(entry["cv"][alg]["model_b"]["roc_auc"])
            plt.plot(np.linspace(0, 1, 100), np.mean(fprs, axis=0), lw=2,
                     label=f"{alg} (AUC={np.mean(probs_true):.3f})")
        plt.plot([0, 1], [0, 1], "k:", lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {task['label']}")
        plt.legend(loc="lower right"); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES, f"roc_{task_name}.png"), dpi=140)
        plt.close()

        # ---- Deploy the winning model (fit on all data) -------------------
        final = make_model(best)
        final.fit(d[task["model_b"]], y)
        joblib.dump(final, os.path.join(MODELS, f"{task_name}_model.pkl"))
        print(f"  -> best={best}  delta(A+sleep - A)={b_auc - a_auc:+.4f}  "
              f"saved {task_name}_model.pkl")

        # ---- SHAP explanation figure (tree or linear model) ----------------
        try:
            import shap
            from sklearn.ensemble import RandomForestClassifier as _RF
            from xgboost import XGBClassifier as _XGB
            Xs = np.asarray(make_scaler_only().fit_transform(d[task["model_b"]]))
            mdl = final.named_steps["classifier"]
            if isinstance(mdl, (_RF, _XGB)):
                explainer = shap.TreeExplainer(mdl)
            else:  # LogisticRegression
                explainer = shap.LinearExplainer(mdl, Xs)
            vals = explainer.shap_values(Xs[:300])
            if isinstance(vals, list):
                vals = vals[1]
            elif getattr(vals, "ndim", 0) == 3:
                vals = vals[:, :, 1]
            # population-level figure: no single person's value applies, so labels
            # are shown without the bracketed value used on the per-patient bars
            readable = [display_feature(f) for f in task["model_b"]]
            shap.summary_plot(np.asarray(vals), Xs[:300],
                              feature_names=readable, show=False,
                              max_display=12)
            plt.title(f"SHAP (log-odds) - {task['label']}", fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES, f"shap_{task_name}.png"), dpi=140)
            plt.close()
            print(f"  -> shap_{task_name}.png")
        except Exception as e:  # pragma: no cover
            print(f"  ! SHAP skipped: {e}")

    with open(METRICS, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {METRICS}  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()

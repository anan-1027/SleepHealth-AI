"""
Adds rigor numbers for the three LEGACY models so that every number shown in the app has a
confidence interval, and so the Kaggle demo is scored honestly.

Part A - Diabetes & Hypertension (NHANES, original Model A / Model B feature sets)
         repeated stratified CV (5x3), same algorithms as the deployed pickles.

Part B - Kaggle sleep-pattern demo, scored TWO ways:
         (i)  ordinary row-wise stratified CV  -> reproduces the optimistic number
         (ii) StratifiedGroupKFold where a "group" = an identical feature pattern
              -> no duplicate pattern can leak between train and test.
         Difference between (i) and (ii) quantifies the duplicate-row leakage.

Writes into results/metrics/metrics.json:
    tasks.diabetes, tasks.hypertension   (same schema as the new tasks)
    legacy_demo                          (Part B)
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

METRICS = os.path.join("results", "metrics", "metrics.json")
NH = os.path.join("data", "processed", "nhanes_sleep_health.csv")
KG = os.path.join("data", "processed", "sleep_cleansed.csv")

SKF = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)


def pipe(name):
    if name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    else:
        clf = XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, eval_metric="logloss", n_jobs=-1)
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("classifier", clf)])


def cv_binary(X, y, alg):
    out = {k: [] for k in ("roc_auc", "pr_auc", "f1", "recall", "precision",
                           "accuracy", "brier")}
    for tr, te in SKF.split(X, y):
        m = pipe(alg).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        yt = y.iloc[te]
        out["roc_auc"].append(roc_auc_score(yt, p))
        out["pr_auc"].append(average_precision_score(yt, p))
        out["f1"].append(f1_score(yt, pred, zero_division=0))
        out["recall"].append(recall_score(yt, pred, zero_division=0))
        out["precision"].append(precision_score(yt, pred, zero_division=0))
        out["accuracy"].append(accuracy_score(yt, pred))
        out["brier"].append(brier_score_loss(yt, p))
    res = {}
    for k, v in out.items():
        a = np.asarray(v, float)
        res[k] = round(float(a.mean()), 4)
        res[k + "_std"] = round(float(a.std()), 4)
        res[k + "_ci95"] = [round(float(np.percentile(a, 2.5)), 4),
                            round(float(np.percentile(a, 97.5)), 4)]
    return res


def stats(vals):
    a = np.asarray(vals, float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "ci95": [round(float(np.percentile(a, 2.5)), 4),
                     round(float(np.percentile(a, 97.5)), 4)]}


def main():
    with open(METRICS, encoding="utf-8") as fh:
        results = json.load(fh)

    # ---------------- Part A : legacy cardiometabolic -----------------------
    df = pd.read_csv(NH)
    A = ["age", "gender_female", "bmi", "waist_circumference"]
    B = A + ["sleep_duration_hrs", "trouble_sleeping"]
    legacy = {
        "diabetes": ("target_diabetes", "Diabetes (self-reported doctor diagnosis)",
                     "DIQ010 == 1, CDC NHANES 2017-March 2020 pre-pandemic. Legacy ablation study.",
                     "RandomForest"),
        "hypertension": ("target_hypertension", "Hypertension (self-reported)",
                         "BPQ020 == 1, CDC NHANES 2017-March 2020 pre-pandemic. Legacy ablation study.",
                         "XGBoost"),
    }
    for name, (target, label, definition, alg) in legacy.items():
        y = df[target].astype(int)
        entry = {
            "label": label, "definition": definition, "n": int(len(y)),
            "prevalence": round(float(y.mean()), 4),
            "majority_baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 4),
            "features_model_a": A, "features_model_b": B,
            "cv": {alg: {"model_a": cv_binary(df[A], y, alg),
                         "model_b": cv_binary(df[B], y, alg)}},
            "best_model": alg,
        }
        a = entry["cv"][alg]["model_a"]["roc_auc"]
        b = entry["cv"][alg]["model_b"]["roc_auc"]
        entry["ablation"] = {"algorithm": alg,
                             "model_a_demographics_roc_auc": a,
                             "model_b_plus_sleep_roc_auc": b,
                             "delta": round(b - a, 4),
                             "delta_pr_auc": round(
                                 entry["cv"][alg]["model_b"]["pr_auc"]
                                 - entry["cv"][alg]["model_a"]["pr_auc"], 4)}
        entry["metrics_model_b"] = entry["cv"][alg]["model_b"]
        results["tasks"][name] = entry
        print(f"{name:14s} A={a:.4f} B={b:.4f} d={b - a:+.4f} "
              f"AUC={b:.4f} CI={entry['metrics_model_b']['roc_auc_ci95']}")

    # ---------------- Part B : Kaggle demo, honest scoring ------------------
    kg = pd.read_csv(KG)
    yk = kg["Target_Sleep_Disorder"].astype(int)
    Xk = kg.drop(columns=["Target_Sleep_Disorder"])
    groups = pd.util.hash_pandas_object(Xk, index=False).values  # identical pattern = 1 group

    cat = ["Gender", "Occupation", "BMI Category"]
    num = [c for c in Xk.columns if c not in cat]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    demo_pipe = Pipeline([("pre", pre),
                          ("classifier", RandomForestClassifier(random_state=42))])

    def rowwise():
        accs, f1s = [], []
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
        for tr, te in skf.split(Xk, yk):
            demo_pipe.fit(Xk.iloc[tr], yk.iloc[tr])
            p = demo_pipe.predict(Xk.iloc[te])
            accs.append(accuracy_score(yk.iloc[te], p))
            f1s.append(f1_score(yk.iloc[te], p, average="macro"))
        return accs, f1s

    def groupwise():
        accs, f1s = [], []
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in gkf.split(Xk, yk, groups):
            demo_pipe.fit(Xk.iloc[tr], yk.iloc[tr])
            p = demo_pipe.predict(Xk.iloc[te])
            accs.append(accuracy_score(yk.iloc[te], p))
            f1s.append(f1_score(yk.iloc[te], p, average="macro"))
        return accs, f1s

    ra, rf1 = rowwise()
    ga, gf1 = groupwise()
    results["legacy_demo"] = {
        "dataset": "Kaggle sleep-health & lifestyle (educational, synthetic labels)",
        "n_rows": int(len(kg)),
        "n_unique_feature_patterns": int(pd.unique(groups).size),
        "note": ("Row-wise CV lets identical feature patterns appear in both train and test "
                 "and therefore overestimates performance; StratifiedGroupKFold keeps each "
                 "unique pattern in a single fold."),
        "rowwise_cv_accuracy": stats(ra),
        "rowwise_cv_macro_f1": stats(rf1),
        "groupaware_cv_accuracy": stats(ga),
        "groupaware_cv_macro_f1": stats(gf1),
        "optimistic_bias_accuracy": round(float(np.mean(ra) - np.mean(ga)), 4),
    }
    print(f"demo row-wise acc={np.mean(ra):.4f}  group-aware acc={np.mean(ga):.4f}  "
          f"bias={np.mean(ra) - np.mean(ga):+.4f}")

    with open(METRICS, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"updated {METRICS}")


if __name__ == "__main__":
    main()

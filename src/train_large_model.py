"""
Large-scale sleep-risk model (100,000 records).

Dataset : data/raw/large_sleep_data.csv  ("Sleep Health & Daily Performance Dataset",
          100,000 records x 32 columns). This file is SYNTHETIC / generated data and is
          labelled as such everywhere it is presented.

Target  : sleep_disorder_risk  (ordinal: Healthy / Mild / Moderate / Severe)

Leakage control: person_id is an identifier; cognitive_performance_score and
felt_rested are sibling TARGET columns of the same dataset and are excluded.

Outputs : models/large_sleep_risk_model.pkl
          results/metrics/large_model_metrics.json
          results/figures/large_confusion_matrix.png
          results/figures/large_class_distribution.png
"""
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA = os.path.join("data", "raw", "large_sleep_data.csv")
MODEL_OUT = os.path.join("models", "large_sleep_risk_model.pkl")
METRICS_OUT = os.path.join("results", "metrics", "large_model_metrics.json")
FIG = "results/figures"

EXCLUDE = ["person_id", "sleep_disorder_risk",
           "cognitive_performance_score", "felt_rested"]   # id + sibling targets
CLASSES = ["Healthy", "Mild", "Moderate", "Severe"]


def main() -> None:
    t0 = time.time()
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    df = pd.read_csv(DATA)
    print(f"Loaded {DATA}  shape={df.shape}")
    y_raw = df["sleep_disorder_risk"]
    print("class distribution:\n", y_raw.value_counts().reindex(CLASSES).to_string())
    # XGBoost requires integer class codes; LabelEncoder sorts alphabetically,
    # which matches CLASSES order (Healthy < Mild < Moderate < Severe).
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw), index=df.index, name="sleep_disorder_risk")
    assert list(le.classes_) == CLASSES, list(le.classes_)
    X = df.drop(columns=EXCLUDE)
    cat = X.select_dtypes(include=["object", "str"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    print(f"features={X.shape[1]}  ({len(num)} numeric, {len(cat)} categorical)")

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", XGBClassifier(n_estimators=350, max_depth=8, learning_rate=0.08,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                              random_state=42, n_jobs=-1))])

    # ---- Repeated stratified CV (5 x 2) for headline numbers ---------------
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    accs, f1s, baccs = [], [], []
    for i, (tr, te) in enumerate(skf.split(X, y)):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], p))
        baccs.append(balanced_accuracy_score(y.iloc[te], p))
        f1s.append(f1_score(y.iloc[te], p, average="macro"))
        print(f"  fold {i + 1:2d}: acc={accs[-1]:.4f} bal_acc={baccs[-1]:.4f} "
              f"macroF1={f1s[-1]:.4f}")

    def stats(a):
        a = np.asarray(a, float)
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
                "ci95": [round(float(np.percentile(a, 2.5)), 4),
                         round(float(np.percentile(a, 97.5)), 4)]}

    def _py(o):
        """JSON-safe conversion for numpy scalars nested in dicts/lists."""
        if isinstance(o, dict):
            return {k: _py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    # ---- Hold-out report (20%) --------------------------------------------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print("\nHold-out classification report:")
    print(classification_report(yte, pred, labels=[0, 1, 2, 3], target_names=CLASSES, digits=3))

    # ---- Figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(yte, pred, labels=[0, 1, 2, 3], normalize="true")
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm[i, j] > 0.5 else "black")
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=20)
    plt.yticks(range(len(CLASSES)), CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Large sleep-risk model - hold-out confusion matrix (normalised)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "large_confusion_matrix.png"), dpi=140)
    plt.close()

    vc = y_raw.value_counts().reindex(CLASSES)
    plt.figure(figsize=(7, 4))
    plt.bar(vc.index, vc.values, color=["#00CC96", "#FFA15A", "#FFA15A", "#EF553B"])
    plt.title(f"Class distribution (n={len(y):,})")
    plt.ylabel("records"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "large_class_distribution.png"), dpi=140)
    plt.close()

    # ---- Deploy: refit on full data ---------------------------------------
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_OUT)

    metrics = {
        "model": "large_sleep_risk_model",
        "dataset": DATA.replace("\\", "/"),
        "dataset_note": "SYNTHETIC / generated benchmark dataset (Sleep Health & Daily "
                        "Performance Dataset, 100,000 records, 32 columns, year 2026).",
        "n_records": int(len(df)),
        "n_features": int(X.shape[1]),
        "target": "sleep_disorder_risk (ordinal: Healthy < Mild < Moderate < Severe)",
        "excluded_columns": EXCLUDE,
        "design": "RepeatedStratifiedKFold 5 folds x 2 repeats + 20% stratified hold-out",
        "cv_accuracy": stats(accs),
        "cv_balanced_accuracy": stats(baccs),
        "cv_macro_f1": stats(f1s),
        "holdout_report": _py(classification_report(
            yte, pred, labels=[0, 1, 2, 3], target_names=CLASSES,
            output_dict=True, digits=4)),
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\nSaved {MODEL_OUT}\nSaved {METRICS_OUT}\n({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

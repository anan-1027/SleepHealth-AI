"""Regenerate ONE saved SHAP figure without retraining.

WHY THIS EXISTS
    train_models.py fits 3 algorithms x 5 folds x 3 tasks, which needs a lot of
    RAM. On a memory-constrained machine that can fail with an OpenBLAS
    allocation error, even though the model pickles on disk are perfectly good.

    Changing a plot label is not a reason to refit a model. This script loads the
    already-deployed pickle and rewrites only its SHAP figure, so a cosmetic
    change costs seconds instead of minutes and touches no metrics.

USAGE
    python src/regen_shap_figure.py osa_risk
"""
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings
from sklearn.ensemble import RandomForestClassifier as _RF
from xgboost import XGBClassifier as _XGB

from feature_labels import display_feature
from train_models import (DATA, FIGURES, MODELS, TASKS, make_scaler_only)

warnings.filterwarnings("ignore")


def regen(task_name: str) -> str:
    if task_name not in TASKS:
        raise SystemExit(f"unknown task {task_name!r}; choose from {list(TASKS)}")
    task = TASKS[task_name]

    # Load the deployed model - it is a full Pipeline (imputer -> scaler -> model).
    path = os.path.join(MODELS, f"{task_name}_model.pkl")
    final = joblib.load(path)

    # Reproduce the exact feature matrix the training script used.
    d = pd.read_csv(DATA)
    feats = task["model_b"]
    Xs = np.asarray(make_scaler_only().fit_transform(d[feats]))

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

    readable = [display_feature(f) for f in feats]
    shap.summary_plot(np.asarray(vals), Xs[:300], feature_names=readable,
                      show=False, max_display=12)
    plt.title(f"SHAP (log-odds) - {task['label']}", fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIGURES, f"shap_{task_name}.png")
    plt.savefig(out, dpi=140)
    plt.close()
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "osa_risk"
    print("wrote", regen(which))

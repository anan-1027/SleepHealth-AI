"""Verify the numbers written to calibration.json by recomputing them independently."""
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import train_models as tm

cal = json.load(open("results/metrics/calibration.json", encoding="utf-8"))
df = pd.read_csv(tm.DATA)


def ece_freq(y, p, n=10):
    edges = np.quantile(p, np.linspace(0, 1, n + 1))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n - 1)
    tot, e = len(y), 0.0
    for b in range(n):
        m = idx == b
        if m.sum():
            # abs per bin BEFORE weighting (errors must not cancel)
            e += m.sum() / tot * abs(y[m].mean() - p[m].mean())
    return e


for task_name, task in tm.TASKS.items():
    d = df.dropna(subset=[task["target"]]).copy()
    y = d[task["target"]].astype(int).to_numpy()
    X = d[task["model_b"]]
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    oof = np.zeros(len(d))
    for tr, te in skf.split(X, y):
        m = tm.make_model("LogisticRegression")
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    c = np.zeros(len(d))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=1,
                                          random_state=0).split(oof, y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(oof[tr], y[tr])
        c[te] = iso.predict(oof[te])
    print(f"\n{task_name}")
    print(f"  independent  brier {brier_score_loss(y,oof):.4f}->{brier_score_loss(y,c):.4f} "
          f"ece_freq {ece_freq(y,oof):.4f}->{ece_freq(y,c):.4f} "
          f"auc {roc_auc_score(y,oof):.4f}->{roc_auc_score(y,c):.4f}")
    r = cal[task_name]
    print(f"  json         brier {r['brier_before']}->{r['brier_after']} "
          f"ece_freq {r['ece_equalfreq_before']}->{r['ece_equalfreq_after']} "
          f"auc {r['roc_auc_before']}->{r['roc_auc_after']}")
    print(f"  distinct values in calibrated array: {len(np.unique(c))}")

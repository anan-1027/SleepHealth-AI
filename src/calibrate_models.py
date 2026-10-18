"""
Probability calibration for the three NHANES screening models.

WHY THIS EXISTS
    LogisticRegression / RandomForest / XGBoost output *scores* that behave like
    probabilities but are usually miscalibrated: a model that says "30% risk" may
    actually be right only 22% of the time. Our app draws those numbers as risk
    GAUGES, so an uncalibrated score would mislead users about their risk.

METHOD: cross-fitted (out-of-fold) isotonic calibration
    1. Produce out-of-fold predictions with the SAME repeated-CV scheme used for
       the headline metrics (so no row is ever predicted by a model trained on it).
    2. Fit an IsotonicRegression mapping raw score -> observed event rate on those
       honest out-of-fold predictions.
    3. Also record Brier score + Expected Calibration Error (ECE) before/after, and
       a 10-bin reliability table, so the improvement is verifiable rather than
       asserted.

NOTE: calibration can only ever IMPROVE calibration (Brier/ECE), never ranking
(ROC-AUC), so the headline AUCs stay untouched.
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import train_models as tm

warnings.filterwarnings("ignore")

CAL_DIR = os.path.join("models", "calibration")
N_BINS = 10


def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS,
        scheme: str = "width") -> float:
    """Expected Calibration Error.

    scheme="width"  -> equal-width bins (0.0-0.1, 0.1-0.2, ...). With a rare outcome
                       most rows land in the lowest bin, so this number is dominated by
                       that bin and can look implausibly small.
    scheme="freq"   -> equal-frequency (quantile) bins. The more honest diagnostic when
                       prevalence is far from 50%, because every bin has similar n.
    Both are reported so neither can flatter the model on its own.
    """
    edges = (np.linspace(0.0, 1.0, n_bins + 1) if scheme == "width"
             else np.quantile(prob, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    total = len(y_true)
    ece_val = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        # absolute gap per bin, THEN weighted: taking abs after summing would let
        # over- and under-prediction cancel and understate the error.
        ece_val += mask.sum() / total * abs(y_true[mask].mean() - prob[mask].mean())
    return float(ece_val)


def reliability_table(y_true: np.ndarray, prob: np.ndarray, n_bins: int = N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                     "n": int(mask.sum()),
                     "mean_predicted": round(float(prob[mask].mean()), 4),
                     "observed_rate": round(float(y_true[mask].mean()), 4)})
    return rows


def main() -> None:
    os.makedirs(CAL_DIR, exist_ok=True)
    df = pd.read_csv(tm.DATA)
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # the winning algorithm per task was decided during training; reuse it here
    with open("results/metrics/metrics.json", encoding="utf-8") as fh:
        trained = json.load(fh)["tasks"]
    best = {k: v.get("best_model", "LogisticRegression") for k, v in trained.items()}

    report = {"method": "cross-fitted isotonic regression on out-of-fold predictions",
              "n_bins": N_BINS,
              "note": ("Calibration fixes the reliability of the reported risk level. "
                       "It cannot change ROC-AUC.")}

    for task_name, task in tm.TASKS.items():
        d = df.dropna(subset=[task["target"]]).copy()
        y = d[task["target"]].astype(int).to_numpy()
        feats = task["model_b"]
        X = d[feats]

        oof = np.zeros(len(d))
        for tr, te in skf.split(X, y):
            mdl = tm.make_model(best.get(task_name, "LogisticRegression"))
            mdl.fit(X.iloc[tr], y[tr])
            oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]

        raw_brier = brier_score_loss(y, oof)
        raw_ece = ece(y, oof, scheme="width")
        raw_ece_freq = ece(y, oof, scheme="freq")
        raw_auc = roc_auc_score(y, oof)

        # Cross-fitted calibration: fit the isotonic map on 4/5 of the out-of-fold
        # predictions and score it on the untouched 1/5, then rotate. Fitting and
        # scoring the calibrator on the same predictions would be in-sample and
        # would report an artificially perfect ECE.
        cal = np.zeros(len(d))
        cal_idx = np.arange(len(d))
        for tr, te in RepeatedStratifiedKFold(
                n_splits=5, n_repeats=1, random_state=0).split(oof, y):
            iso_fold = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso_fold.fit(oof[tr], y[tr])
            cal[te] = iso_fold.predict(oof[te])

        # A single deployable isotonic map fitted on ALL out-of-fold predictions is
        # saved for the app; its reported metrics above come from the stricter
        # cross-fitted version, not from this fit.
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(oof, y)

        cal_brier = brier_score_loss(y, cal)
        cal_ece = ece(y, cal, scheme="width")
        cal_ece_freq = ece(y, cal, scheme="freq")
        cal_auc = roc_auc_score(y, cal)

        path = os.path.join(CAL_DIR, f"{task_name}_isotonic.pkl")
        joblib.dump(iso, path)

        report[task_name] = {
            "roc_auc_before": round(raw_auc, 4),
            "roc_auc_after": round(cal_auc, 4),
            "brier_before": round(raw_brier, 4),
            "brier_after": round(cal_brier, 4),
            "ece_before": round(raw_ece, 4),
            "ece_after": round(cal_ece, 4),
            "ece_equalfreq_before": round(raw_ece_freq, 4),
            "ece_equalfreq_after": round(cal_ece_freq, 4),
            "ece_note": ("equal-frequency ECE is the honest diagnostic at a ~10% base "
                         "rate; equal-width ECE is dominated by the large low-risk bin"),
            "n": int(len(y)),
            "reliability_before": reliability_table(y, oof),
            "reliability_after": reliability_table(y, cal),
            "model_file": path.replace("\\", "/"),
        }
        print(f"{task_name:15s} Brier {raw_brier:.4f}->{cal_brier:.4f}  "
              f"ECE(w/freq) {raw_ece:.4f}/{raw_ece_freq:.4f}->"
              f"{cal_ece:.4f}/{cal_ece_freq:.4f}  AUC {raw_auc:.4f}->{cal_auc:.4f}")

    with open("results/metrics/calibration.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("wrote results/metrics/calibration.json")


if __name__ == "__main__":
    main()

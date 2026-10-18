"""
NHANES 2017-2018 (Cycle J) -> analysis-ready sleep-disorder / cardiovascular table.

Sources (CDC public use files, all stored in data/raw/):
  DEMO_J : demographics          BMX_J  : body measures
  SLQ_J  : sleep disorders       DIQ_J  : diabetes
  BPQ_J  : hypertension          MCQ_J  : medical conditions (heart disease labels)

Builds three NEW prediction targets that were previously unused:
  target_heart_disease : self-reported CHD / CHF / angina / heart attack  (MCQ160B/C/D/E)
  target_insomnia_risk  : ever told a doctor about trouble sleeping       (SLQ050)
  target_osa_risk       : symptom-based obstructive sleep-apnea risk      (SLQ030 + SLQ040)

IMPORTANT / anti-leakage rule: the symptom items used to BUILD a label are
excluded from that label's feature list (see MODEL_FEATURES in train_models.py).

Output: data/processed/nhanes_sleep_health.csv
"""
import os
import pandas as pd
import numpy as np

RAW = os.path.join("data", "raw")
OUT = os.path.join("data", "processed", "nhanes_sleep_health.csv")

FILES = ["DEMO_J.XPT", "BMX_J.XPT", "SLQ_J.XPT", "DIQ_J.XPT", "BPQ_J.XPT", "MCQ_J.XPT"]

# CDC codes that mean Refused(7) / Don't know(9) -> treat as missing
INVALID = [7.0, 9.0]


def load_merged() -> pd.DataFrame:
    merged = None
    for f in FILES:
        path = os.path.join(RAW, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing NHANES file: {path}")
        df = pd.read_sas(path)
        # read_sas decodes the XPT integer "0" as a denormalised float
        # (5.397605346934028e-79). Normalise it back to 0.0 while leaving
        # genuine NaN missing values untouched.
        num = df.select_dtypes(include=[np.number])
        df[num.columns] = num.mask(num.notna() & (num.abs() < 1e-70), 0.0)
        print(f"  {f:12s} {df.shape}")
        merged = df if merged is None else pd.merge(merged, df, on="SEQN", how="inner")
    return merged


def main() -> None:
    print("--- Merging NHANES 2017-2018 modules ---")
    n = load_merged()
    print(f"Merged shape: {n.shape}")

    # ---- Adult cohort (identical criteria to notebooks/03) --------------------
    c = n[(n["RIDAGEYR"] >= 20) & n["DIQ010"].isin([1, 2]) & n["BPQ020"].isin([1, 2])].copy()
    print(f"Adult cohort (age>=20, valid DIQ010 & BPQ020): {c.shape[0]}")

    out = pd.DataFrame(index=c.index)

    # ---- Features: demographics & anthropometrics ---------------------------
    out["age"] = c["RIDAGEYR"]
    out["gender_female"] = (c["RIAGENDR"] == 2).astype(int)
    out["bmi"] = c["BMXBMI"]
    out["waist_circumference"] = c["BMXWAIST"]

    # ---- Features: sleep pattern -------------------------------------------
    out["sleep_duration_hrs"] = c["SLD012"]          # weekday/workday hours
    out["weekend_sleep_hrs"] = c["SLD013"]           # weekend/non-workday hours
    out["social_jetlag_hrs"] = (c["SLD013"] - c["SLD012"]).round(2)

    # Frequency items: 0=Never, 1=Rarely, 2=Occasionally, 3=Frequently (SLQ030/040)
    # SLQ120: 0=Never .. 4=Almost always.  7/9 (Refused/Don't know) -> NaN.
    def freq(col: str, max_code: int) -> pd.Series:
        s = c[col].replace(INVALID, np.nan).where(c[col].replace(INVALID, np.nan) <= max_code)
        return s.astype(float)

    out["snore_freq"] = freq("SLQ030", 3)
    out["witnessed_apnea_freq"] = freq("SLQ040", 3)
    out["daytime_sleepiness"] = freq("SLQ120", 4)
    # SLQ050: 1=Yes, 2=No
    out["trouble_sleeping"] = c["SLQ050"].replace(INVALID, np.nan).map({1.0: 1, 2.0: 0})

    # ---- Labels -------------------------------------------------------------
    # 1) Heart disease (self-report, doctor-diagnosed): CHF, CHD, angina, MI.
    heart_cols = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E"]
    heart = c[heart_cols].replace(INVALID, np.nan)
    answered = heart.notna().any(axis=1)
    out["target_heart_disease"] = ((heart == 1).any(axis=1)).astype(int).where(answered, np.nan)

    # 2) Insomnia risk: ever told a doctor about trouble sleeping.
    out["target_insomnia_risk"] = out["trouble_sleeping"]

    # 3) Obstructive sleep-apnea risk: symptom based (frequent snoring
    #    >=5 nights/week OR occasional-or-worse witnessed breathing pauses).
    osa = ((out["snore_freq"] == 3) | (out["witnessed_apnea_freq"] >= 2)).astype(int)
    out["target_osa_risk"] = osa.where(out["snore_freq"].notna() & out["witnessed_apnea_freq"].notna())

    # ---- Legacy targets (unchanged, kept for backward compatibility) --------
    out["target_diabetes"] = (c["DIQ010"] == 1).astype(int)
    out["target_hypertension"] = (c["BPQ020"] == 1).astype(int)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    # ---- Report -------------------------------------------------------------
    print(f"\nSaved -> {OUT}  shape={out.shape}")
    print("\nMissing values per column:")
    print(out.isnull().sum().to_string())
    print("\nLabel prevalence (1 = positive):")
    for lab in ["target_heart_disease", "target_insomnia_risk", "target_osa_risk",
                "target_diabetes", "target_hypertension"]:
        s = out[lab].dropna()
        print(f"  {lab:24s} n={len(s):5d}  positive={s.mean():.3f}")
    print("\nSleep feature summary:")
    print(out[["sleep_duration_hrs", "weekend_sleep_hrs", "social_jetlag_hrs", "snore_freq",
               "witnessed_apnea_freq", "daytime_sleepiness"]].describe().T[
        ["count", "mean", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()

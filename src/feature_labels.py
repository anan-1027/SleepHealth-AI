"""
Human-readable labels for model features.

WHY THIS EXISTS
    Models are trained on internal column names (gender_female, snore_freq, ...).
    Those names are correct for code but confusing in a chart: selecting "Male"
    and then seeing "gender_female" in a SHAP plot reads like a bug.

    This module is the single source of truth for feature display names, used by
    BOTH the Streamlit app (SHAP bars) and train_models.py (SHAP summary PNGs),
    so the wording can never drift between the UI and the report figures.

DESIGN RULE
    Never surface a raw snake_case identifier to a reader. display_feature()
    appends the user's actual selected value where one is known, so a SHAP bar
    reads like an explanation rather than a schema dump.
"""
from __future__ import annotations

import math

FEATURE_LABELS = {
    "age": "Age",
    "gender_female": "Sex",
    "bmi": "BMI",
    "waist_circumference": "Waist circumference",
    "sleep_duration_hrs": "Weekday sleep duration",
    "weekend_sleep_hrs": "Weekend sleep duration",
    "social_jetlag_hrs": "Social jetlag (weekend - weekday)",
    "snore_freq": "Snoring frequency",
    "witnessed_apnea_freq": "Witnessed breathing pauses",
    "daytime_sleepiness": "Daytime sleepiness",
    "trouble_sleeping": "Trouble sleeping (doctor)",
}

# Questionnaire response scales, matching the NHANES codebook.
_FREQ_0_3 = {0: "Never", 1: "Rarely (1-2/wk)", 2: "Occasionally (3-4/wk)",
             3: "Frequently (5+/wk)"}
_SLEEPY_0_4 = {0: "Never", 1: "Rarely (1x/mo)", 2: "Sometimes (2-4/mo)",
               3: "Often (5-15/mo)", 4: "Almost always"}
_YES_NO = {0: "No", 1: "Yes"}


def format_value(name: str, value) -> str | None:
    """Render a single feature value the way a person would say it."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        return None

    if name == "gender_female":
        return "Female" if int(value) == 1 else "Male"
    if name in ("snore_freq", "witnessed_apnea_freq"):
        return _FREQ_0_3.get(int(value), str(value))
    if name == "daytime_sleepiness":
        return _SLEEPY_0_4.get(int(value), str(value))
    if name == "trouble_sleeping":
        return _YES_NO.get(int(value), str(value))
    if name == "social_jetlag_hrs":
        return f"{float(value):+.1f} h"
    if name in ("sleep_duration_hrs", "weekend_sleep_hrs"):
        return f"{float(value):.1f} h"
    if name == "age":
        return f"{float(value):.0f} years"
    if name == "bmi":
        return f"{float(value):.1f} kg/m2"
    if name == "waist_circumference":
        return f"{float(value):.0f} cm"
    return None


def display_feature(name: str, value=None) -> str:
    """'gender_female' + 0 -> 'Sex: Male'; without a value -> 'Sex'."""
    base = FEATURE_LABELS.get(name, name.replace("_", " ").strip().capitalize())
    rendered = format_value(name, value)
    return f"{base}: {rendered}" if rendered is not None else base


def display_features(names, row=None) -> list[str]:
    """Label every feature for a chart, pulling values from a one-row frame."""
    out = []
    for n in names:
        value = None
        if row is not None and n in row.columns:
            value = row.iloc[0][n]
        out.append(display_feature(n, value))
    return out

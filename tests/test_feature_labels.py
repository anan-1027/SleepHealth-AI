"""Unit checks for the readable feature labels used in SHAP charts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from feature_labels import display_feature, display_features

FAIL = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# The exact user complaint: male selected, chart must not say gender_female
check("male -> 'Sex: Male'", display_feature("gender_female", 0) == "Sex: Male",
      display_feature("gender_female", 0))
check("female -> 'Sex: Female'", display_feature("gender_female", 1) == "Sex: Female",
      display_feature("gender_female", 1))

check("no raw identifier leaked for sex",
      "gender_female" not in display_feature("gender_female", 0))

# questionnaire scales read back in the user's own words
check("snore=3 -> Frequently (5+/wk)",
      display_feature("snore_freq", 3) == "Snoring frequency: Frequently (5+/wk)",
      display_feature("snore_freq", 3))
check("trouble=0 -> No",
      display_feature("trouble_sleeping", 0) == "Trouble sleeping (doctor): No",
      display_feature("trouble_sleeping", 0))
check("jetlag signed", display_feature("social_jetlag_hrs", -1.5) ==
      "Social jetlag (weekend - weekday): -1.5 h",
      display_feature("social_jetlag_hrs", -1.5))

# no raw snake_case identifier may reach a chart, for any feature we actually use
import json
from feature_labels import FEATURE_LABELS

metrics = json.loads((ROOT / "results/metrics/metrics.json").read_text(encoding="utf-8"))
all_feats = set()
for t in metrics["tasks"].values():
    all_feats.update(t.get("features_model_b", []))
    all_feats.update(t.get("features_model_a", []))

missing = sorted(f for f in all_feats if f not in FEATURE_LABELS)
check("every modelled feature has a curated label", not missing, str(missing))

# and no curated label may itself contain a raw identifier
leaky = [f for f in sorted(all_feats) if "_" in FEATURE_LABELS.get(f, "")]
check("no curated label contains snake_case", not leaky, str(leaky))

# NaN / None must degrade gracefully, not crash
check("None value -> plain label", display_feature("bmi", None) == "BMI",
      display_feature("bmi", None))
check("NaN value -> plain label", display_feature("bmi", float("nan")) == "BMI")

# batch helper pulls values out of a one-row frame
row = pd.DataFrame([{"age": 45, "gender_female": 0, "bmi": 27.3}])
out = display_features(["age", "gender_female", "bmi"], row)
check("display_features uses row values",
      out == ["Age: 45 years", "Sex: Male", "BMI: 27.3 kg/m2"], str(out))

print("\n" + "=" * 50)
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL LABEL CHECKS PASSED")

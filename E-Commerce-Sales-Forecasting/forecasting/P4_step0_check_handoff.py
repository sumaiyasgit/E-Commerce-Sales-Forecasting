"""
PERSON 4 - STEP 0: CHECK WHAT YOUR TEAMMATES GAVE YOU
Run this FIRST.  It prints OK / WARNING lines so you know the files are usable.

Run:  python P4_step0_check_handoff.py
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error

from forecast_utils import FEATURE_COLUMNS, make_features, load_series, load_model

problems = []


def ok(msg):
    print("  OK       ", msg)


def warn(msg):
    print("  WARNING  ", msg)
    problems.append(msg)


print("\n1) Your computer")
print("   Python", sys.version.split()[0], "| scikit-learn", sklearn.__version__, "| pandas", pd.__version__)

print("\n2) Files from teammates")
needed = ["data/daily_sales.csv", "data/test_data.csv", "models/final_model.joblib",
          "models/random_forest_tuned.joblib", "models/feature_columns.joblib"]
for f in needed:
    ok(f) if os.path.exists(f) else warn(f"missing file: {f}")
if any(not os.path.exists(f) for f in needed):
    raise SystemExit("\nPut the missing files in the folders shown above, then run this again.")

print("\n3) Daily sales data (Person 1)")
s = load_series()
print(f"   {len(s)} days, {s.index.min().date()} to {s.index.max().date()}, "
      f"average {s.mean():,.0f}, min {s.min():,.0f}, max {s.max():,.0f}")
ok("no missing values") if s.isna().sum() == 0 else warn("there are missing values")
gaps = (s.index.to_series().diff().dropna() != pd.Timedelta(days=1)).sum()
ok("every calendar day is present") if gaps == 0 else warn(f"{gaps} gaps in the calendar")

print("\n4) Feature list (Person 3)")
saved_cols = list(joblib.load("models/feature_columns.joblib"))
ok("feature names and order match this kit") if saved_cols == FEATURE_COLUMNS else warn(
    f"feature list differs!\n   theirs: {saved_cols}\n   kit:    {FEATURE_COLUMNS}\n   -> edit FEATURE_COLUMNS in forecast_utils.py")

print("\n5) Do my features equal Person 3's features?")
t = pd.read_csv("data/test_data.csv", parse_dates=["order_date"])
vals = s.to_numpy()
mine = pd.concat([make_features(vals[:s.index.get_loc(d)], d) for d in t["order_date"]], ignore_index=True)
diff = np.abs(mine.to_numpy() - t[FEATURE_COLUMNS].to_numpy().astype(float)).max()
ok(f"identical (largest difference {diff:.1e})") if diff < 1e-6 else warn(f"features differ by up to {diff:,.2f}")

print("\n6) Models")
tuned = load_model("models/random_forest_tuned.joblib")
final = load_model("models/final_model.joblib")
y = t["daily_revenue"].to_numpy()
X = mine
mae_tuned = mean_absolute_error(y, tuned.predict(X))
mae_final = mean_absolute_error(y, final.predict(X))
print(f"   test-period MAE, random_forest_tuned (trained on training months only): {mae_tuned:,.0f}")
print(f"   test-period MAE, final_model (trained on ALL months):                   {mae_final:,.0f}")
if os.path.exists("data/final_test_residuals.csv"):
    r = pd.read_csv("data/final_test_residuals.csv")
    d = np.abs(tuned.predict(X) - r["predicted_revenue"].to_numpy()).max()
    ok("random_forest_tuned reproduces Person 3's reported predictions") if d < 1 else warn(
        f"random_forest_tuned does not reproduce Person 3's predictions (diff {d:,.0f}). Run P3_rebuild_models.py")
if mae_final < 0.8 * mae_tuned:
    print("   IMPORTANT: final_model scores much better on the test months because it was trained ON them.")
    print("              Use random_forest_tuned to MEASURE accuracy, final_model only to FORECAST.")

print("\nResult:", "all good, go to P4_step1_evaluate.py" if not problems else f"{len(problems)} warning(s) above, fix them first")

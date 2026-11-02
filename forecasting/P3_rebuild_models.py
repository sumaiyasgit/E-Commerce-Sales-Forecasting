"""
FALLBACK - only run this if loading Person 3's .joblib files fails or warns about versions.

It repeats Person 3's training recipe (copied from their ML_Engineering notebook) on YOUR
computer, so the saved models match YOUR scikit-learn version.  Same data, same settings,
same random_state=42, so the results are the same.

It rebuilds two files:
  models/random_forest_tuned.joblib  trained ONLY on the training months -> use for honest accuracy tests
  models/final_model.joblib          trained on ALL rows                    -> use to forecast the future
Your teammate's originals are copied to models/original_from_person3/ first.

Run:  python P3_rebuild_models.py
"""
import os
import shutil
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

os.makedirs("models/original_from_person3", exist_ok=True)
for f in ("random_forest_tuned.joblib", "final_model.joblib"):
    src, dst = f"models/{f}", f"models/original_from_person3/{f}"
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)

d = pd.read_csv("data/daily_sales.csv", parse_dates=["order_date"]).sort_values("order_date").reset_index(drop=True)
rev = d["daily_revenue"]
d["day_of_week"] = d["order_date"].dt.dayofweek
d["day_of_month"] = d["order_date"].dt.day
d["month"] = d["order_date"].dt.month
d["quarter"] = d["order_date"].dt.quarter
d["year"] = d["order_date"].dt.year
d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
for k in (1, 7, 14, 30):
    d[f"lag_{k}"] = rev.shift(k)
d["rolling_mean_7"] = rev.shift(1).rolling(7).mean()
d["rolling_mean_30"] = rev.shift(1).rolling(30).mean()
data = d.dropna().reset_index(drop=True)

features = ["day_of_week", "day_of_month", "month", "quarter", "year", "is_weekend",
            "lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30"]
split = int(len(data) * 0.80)                       # first 80% of days = training
X, y = data[features], data["daily_revenue"]
settings = dict(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)

train_only = RandomForestRegressor(**settings).fit(X.iloc[:split], y.iloc[:split])
final = RandomForestRegressor(**settings).fit(X, y)
joblib.dump(train_only, "models/random_forest_tuned.joblib", compress=3)
joblib.dump(final, "models/final_model.joblib", compress=3)
joblib.dump(features, "models/feature_columns.joblib")
print(f"Training rows: {split}, test rows: {len(data) - split}  (test starts {data.order_date[split].date()})")
print("Rebuilt models/random_forest_tuned.joblib and models/final_model.joblib")

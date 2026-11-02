"""
PERSON 4 - STEP 1: EVALUATION
 1. One-step-ahead accuracy on the test months (how Person 3 measured it)
 2. Compare against simple baselines  (a model must beat these)
 3. Recursive back-test: pretend we forecast 7 / 30 / 90 days from many past dates
 4. Work out "likely ranges" from past mistakes
 5. Save charts (outputs/*.png) and numbers (outputs/*.csv, app/data/*.json)

IMPORTANT: accuracy is measured with models/random_forest_tuned.joblib, which never saw the
test months.  models/final_model.joblib was trained on ALL months, so testing it would be cheating.

Run:  python P4_step1_evaluate.py        (takes about a minute)
"""
import json
import os
import matplotlib
matplotlib.use("Agg")                       # draw charts to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_utils import (load_series, load_model, evaluate, one_step_predictions,
                            recursive_forecast, baselines_one_step)

os.makedirs("outputs", exist_ok=True)
os.makedirs("app/data", exist_ok=True)

series = load_series()
model = load_model("models/random_forest_tuned.joblib")          # trained on training months only
test_start = pd.read_csv("data/test_data.csv", parse_dates=["order_date"])["order_date"].min()
test_end = series.index.max()
actual = series[series.index >= test_start]
print(f"Test period: {test_start.date()} to {test_end.date()}  ({len(actual)} days)")

# ---------------------------------------------------------------- 1 + 2: one-step vs baselines
preds = pd.DataFrame(baselines_one_step(series, test_start))
preds["Random Forest (Person 3)"] = one_step_predictions(model, series, test_start)
rows = {name: evaluate(actual, preds[name]) for name in preds.columns}
onestep = pd.DataFrame(rows).T.round(3)
print("\nONE-STEP-AHEAD ACCURACY (lower MAE / RMSE / MAPE / WAPE is better; R2: 0 = no better than the average)")
print(onestep.to_string())
onestep.to_csv("outputs/metrics_one_step.csv")

# ---------------------------------------------------------------- 3: recursive back-test
# Stand on every test date as if it were "today" (we only know days BEFORE it), forecast up to 90 days
# with the recursive method, then compare with what really happened.
print("\nRunning recursive back-test (this is the slow part)...")
MAX_H = 90
recs = []
for origin in actual.index:
    days_left = (test_end - origin).days + 1
    L = min(MAX_H, days_left)
    if L < 7:
        continue
    past = series[series.index < origin]
    fc = recursive_forecast(model, past.to_numpy(), past.index[-1], L)
    flat = past.iloc[-30:].mean()                               # baseline: "the recent average, flat"
    recs.append(pd.DataFrame({"origin": origin, "horizon": np.arange(1, L + 1),
                              "actual": series.reindex(fc.index).to_numpy(),
                              "model": fc.to_numpy(), "flat_avg30": flat}))
bt = pd.concat(recs, ignore_index=True)

recursive = {}
for cap in (7, 30, 90):
    sub = bt[bt["horizon"] <= cap]
    recursive[str(cap)] = {"model": evaluate(sub["actual"], sub["model"]),
                           "flat_avg30": evaluate(sub["actual"], sub["flat_avg30"])}
print("\nRECURSIVE ACCURACY, forecasting N days ahead (pooled over many start dates)")
for cap, v in recursive.items():
    print(f"  next {cap:>2} days | model MAE {v['model']['MAE']:8,.0f}  MAPE {v['model']['MAPE']:5.1f}% "
          f"| recent-average baseline MAE {v['flat_avg30']['MAE']:8,.0f}  MAPE {v['flat_avg30']['MAPE']:5.1f}%")
pd.DataFrame({f"{c}d_{k}": v[k] for c, v in recursive.items() for k in v}).T.round(3).to_csv("outputs/metrics_recursive.csv")

# ---------------------------------------------------------------- 4: likely ranges from past mistakes
# ratio = what really happened / what we forecast.  1.00 = perfect.  10th-90th percentile = "80% of the time".
bt["ratio"] = bt["actual"] / bt["model"]
bands = {"day": {"low": float(bt["ratio"].quantile(0.10)), "high": float(bt["ratio"].quantile(0.90))}, "total": {}}

# Range for a TOTAL over H days.  Daily mistakes are almost unrelated to each other (their
# autocorrelation is about 0), so good days and bad days cancel out and the percentage error of a
# total shrinks like 1/sqrt(H).  1.2816 = the multiplier that covers the middle 80% of a bell curve.
# We do NOT use the back-test windows for this: there are only ~140 test days, so windows overlap
# almost completely and would make the range look far too narrow.
sigma = float(bt["ratio"].std())
for H in (7, 30, 90):
    half = 1.2816 * sigma / np.sqrt(H)
    bands["total"][str(H)] = {"low": 1 - half, "high": 1 + half}
print("\nLikely range (multiply the forecast by these; covers about 80% of outcomes):")
print(f"  one day:   x{bands['day']['low']:.2f} to x{bands['day']['high']:.2f}")
for H, b in bands["total"].items():
    print(f"  {H:>2}-day total: x{b['low']:.2f} to x{b['high']:.2f}")
print("  (assumes revenue keeps behaving like the last two years - a real change in the business is not covered)")
json.dump(bands, open("app/data/bands.json", "w"), indent=2)

# a "share with the app" summary
test_avg = float(actual.mean())
json.dump({"one_step": onestep.to_dict(orient="index"), "recursive": recursive,
           "test_period": [str(test_start.date()), str(test_end.date())], "test_average": test_avg},
          open("app/data/metrics.json", "w"), indent=2)

# ---------------------------------------------------------------- charts
BLUE, ORANGE, GREY, INK, GRID = "#3B6FD4", "#C2650F", "#8A94A6", "#1F2937", "#E5E7EB"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.axisbelow": True})

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"outputs/{name}"); plt.close(fig)

model_pred = preds["Random Forest (Person 3)"]

# 1. actual vs predicted
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(actual.index, actual, color=BLUE, lw=1.4, label="Actual")
ax.plot(model_pred.index, model_pred, color=ORANGE, lw=1.4, label="Model prediction")
ax.set(title="Actual vs predicted daily revenue (test period)", ylabel="Revenue per day")
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
ax.set_ylim(top=actual.max() * 1.12)
ax.legend(frameon=False, loc="upper left", ncol=2)
save(fig, "1_actual_vs_predicted.png")

# 2. forecast error
res = actual - model_pred
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2, 1]})
axs[0].bar(res.index, res, color=np.where(res >= 0, BLUE, ORANGE), width=1.0)
axs[0].axhline(0, color=INK, lw=.8)
axs[0].set(title="Forecast error each day (actual - predicted)", ylabel="Error")
from matplotlib.patches import Patch
axs[0].legend(handles=[Patch(color=BLUE, label="Predicted too low"), Patch(color=ORANGE, label="Predicted too high")],
              loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False)
axs[1].hist(res, bins=25, color=GREY)
axs[1].set(title="How big are the mistakes?", xlabel="Error")
save(fig, "2_forecast_error.png")

# 3. model vs baselines (MAE)
mae = onestep["MAE"].sort_values()
fig, ax = plt.subplots(figsize=(9, 3.8))
colors = [ORANGE if "Random Forest" in n else GREY for n in mae.index]
bars = ax.barh(mae.index[::-1], mae.values[::-1], color=colors[::-1], height=0.6)
ax.bar_label(bars, labels=[f"{v:,.0f}" for v in mae.values[::-1]], padding=4, color=INK)
ax.set(title="Average daily error (MAE) - shorter bar is better", xlabel="MAE"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, mae.max() * 1.15)
save(fig, "3_model_vs_baselines.png")

# 4. error vs horizon
curve = bt.assign(m=(bt["actual"] - bt["model"]).abs(), f=(bt["actual"] - bt["flat_avg30"]).abs()).groupby("horizon")[["m", "f"]].mean()
curve = curve.rolling(7, center=True, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(curve.index, curve["m"], color=ORANGE, lw=2, label="Model (recursive)")
ax.plot(curve.index, curve["f"], color=GREY, lw=2, label="Recent average, flat")
ax.set(title="Average daily error by how far ahead we forecast (7-day smoothed)", xlabel="Days ahead", ylabel="MAE")
ax.set_ylim(0, curve.max().max() * 1.25); ax.legend(frameon=False, loc="lower right", ncol=2)
save(fig, "4_error_vs_horizon.png")

print("\nSaved: outputs/1..4 charts, outputs/metrics_one_step.csv, outputs/metrics_recursive.csv, app/data/metrics.json, app/data/bands.json")

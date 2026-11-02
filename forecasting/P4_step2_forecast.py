"""
PERSON 4 - STEP 2: THE REAL FORECAST (days after the last date in the data)
Uses models/final_model.joblib (trained on ALL days) and the recursive method to forecast 90 days.
Saves everything the web app needs into app/, so the app never has to run the model itself.

Run:  python P4_step2_forecast.py        (run P4_step1_evaluate.py first - it creates bands.json)
"""
import json
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast_utils import load_series, load_model, recursive_forecast

HORIZON = 90
os.makedirs("outputs", exist_ok=True)
os.makedirs("app/data", exist_ok=True)
os.makedirs("app/models", exist_ok=True)

series = load_series()
model = load_model("models/final_model.joblib")                      # trained on all days: right for FUTURE forecasts
bands = json.load(open("app/data/bands.json"))

fc = recursive_forecast(model, series.to_numpy(), series.index[-1], HORIZON)
out = fc.to_frame("forecast").reset_index(names="date")
out["lower"] = out["forecast"] * bands["day"]["low"]                 # likely range for ONE day
out["upper"] = out["forecast"] * bands["day"]["high"]
out[["forecast", "lower", "upper"]] = out[["forecast", "lower", "upper"]].round(0)
out.to_csv("app/data/forecast.csv", index=False)

hist = series.reset_index().rename(columns={"order_date": "date"})
hist.to_csv("app/data/history.csv", index=False)
shutil.copy("models/final_model.joblib", "app/models/final_model.joblib")
shutil.copy("forecast_utils.py", "app/forecast_utils.py")

print(f"Last day of data: {series.index[-1].date()}   Forecast: {out['date'].iloc[0].date()} to {out['date'].iloc[-1].date()}")
for H in (7, 30, 90):
    b = bands["total"][str(H)]
    tot = out["forecast"].head(H).sum()
    print(f"  next {H:>2} days: total {tot:>12,.0f}   (likely {tot * b['low']:>12,.0f} to {tot * b['high']:>12,.0f})")
spread = out["forecast"].std() / out["forecast"].mean() * 100
print(f"\nThe forecast moves only {spread:.1f}% around its own average. That is expected: past revenue is")
print("mostly random, so the best forecast is close to the typical level. Say this in your report.")

# ---- charts for your report
BLUE, ORANGE, INK, GRID = "#3B6FD4", "#C2650F", "#1F2937", "#E5E7EB"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.axisbelow": True})
fmt = matplotlib.ticker.StrMethodFormatter("{x:,.0f}")

fig, ax = plt.subplots(figsize=(11, 4))
h = series.iloc[-120:]
ax.plot(h.index, h, color=BLUE, lw=1.3, label="Past revenue")
ax.fill_between(out["date"], out["lower"], out["upper"], color=ORANGE, alpha=.15, label="Likely range for a single day")
ax.plot(out["date"], out["forecast"], color=ORANGE, lw=2, label="Forecast")
ax.axvline(series.index[-1], color=INK, lw=.8, ls="--")
ax.set(title="90-day revenue forecast", ylabel="Revenue per day"); ax.yaxis.set_major_formatter(fmt)
ax.legend(frameon=False, loc="upper left", ncol=3)
ax.set_ylim(top=max(h.max(), out["upper"].max()) * 1.12)
fig.tight_layout(); fig.savefig("outputs/5_future_forecast.png"); plt.close(fig)

monthly = out.assign(month=out["date"].dt.strftime("%b %Y")).groupby("month", sort=False)["forecast"].sum()
days_in = out.assign(month=out["date"].dt.strftime("%b %Y")).groupby("month", sort=False)["forecast"].count()
fig, ax = plt.subplots(figsize=(7, 3.8))
bars = ax.bar(monthly.index, monthly.values, color=ORANGE, width=0.55)
ax.bar_label(bars, labels=[f"{v:,.0f}\n({n} days)" for v, n in zip(monthly.values, days_in.values)], padding=3, color=INK, fontsize=9)
ax.set(title="Forecast revenue by month", ylabel="Revenue"); ax.yaxis.set_major_formatter(fmt)
ax.set_ylim(0, monthly.max() * 1.2); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("outputs/6_monthly_forecast.png"); plt.close(fig)
print("\nSaved app/data/forecast.csv, app/data/history.csv, outputs/5_future_forecast.png, outputs/6_monthly_forecast.png")

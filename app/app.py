"""
PERSON 4 - STEP 3: THE WEB APP
The app only READS files made by P4_step2_forecast.py (forecast.csv, history.csv, metrics.json, bands.json).
It never loads the model, so every click is instant and there is no scikit-learn version problem on the server.

Run on your computer:   python app.py          then open  http://127.0.0.1:5000
Run on Render:          gunicorn app:app
"""
import json
import os

import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))     # works no matter where the app is started from
app = Flask(__name__)

# Put the currency symbol here ONLY after Person 1 confirms what currency the dataset uses
# (for example "₹" or "$").  Leave it empty to show plain numbers.
CURRENCY = ""

HORIZONS = [7, 30, 90]
history = pd.read_csv(os.path.join(BASE, "data", "history.csv"), parse_dates=["date"])
forecast = pd.read_csv(os.path.join(BASE, "data", "forecast.csv"), parse_dates=["date"])
metrics = json.load(open(os.path.join(BASE, "data", "metrics.json")))
bands = json.load(open(os.path.join(BASE, "data", "bands.json")))


def money(value):
    return f"{CURRENCY}{value:,.0f}"


app.jinja_env.filters["money"] = money


def build_result(days):
    fc = forecast.head(days)
    total = float(fc["forecast"].sum())
    b = bands["total"][str(days)]
    avg = float(fc["forecast"].mean())
    recent_avg = float(history["revenue"].tail(30).mean())     # what the last 30 real days averaged

    # second chart: daily bars for a week, weekly totals for longer periods
    if days <= 7:
        bar_title = "Forecast for each day"
        labels = fc["date"].dt.strftime("%a %d %b").tolist()
        values = fc["forecast"].round(0).tolist()
    else:
        bar_title = "Forecast by week (7-day blocks)"
        chunks = [fc.iloc[i:i + 7] for i in range(0, len(fc), 7)]
        labels = [f"Week {i + 1}" + ("" if len(c) == 7 else f" ({len(c)} days)") for i, c in enumerate(chunks)]
        values = [round(float(c["forecast"].sum())) for c in chunks]

    hist = history.tail(90)
    return {
        "days": days,
        "total": total, "low": total * b["low"], "high": total * b["high"],
        "avg": avg, "day_low": avg * bands["day"]["low"], "day_high": avg * bands["day"]["high"],
        "recent_avg": recent_avg,
        "start": fc["date"].iloc[0].strftime("%d %b %Y"), "end": fc["date"].iloc[-1].strftime("%d %b %Y"),
        "chart": {
            "hist_x": hist["date"].dt.strftime("%Y-%m-%d").tolist(), "hist_y": hist["revenue"].tolist(),
            "fc_x": fc["date"].dt.strftime("%Y-%m-%d").tolist(), "fc_y": fc["forecast"].tolist(),
            "lo_y": fc["lower"].tolist(), "hi_y": fc["upper"].tolist(),
            "bar_title": bar_title, "bar_x": labels, "bar_y": values,
        },
    }


@app.route("/")
def index():
    days_raw = request.args.get("days")
    result, error = None, None
    selected = 30
    if days_raw is not None:                           # the form was submitted
        try:
            days = int(days_raw)
        except ValueError:
            days = None
        if days not in HORIZONS:
            error = "Pick a forecast period of 7, 30 or 90 days."
        else:
            selected = days
            result = build_result(days)
    return render_template("index.html", horizons=HORIZONS, selected_days=selected, result=result,
                           error=error, metrics=metrics, last_date=history["date"].max().strftime("%d %b %Y"))


if __name__ == "__main__":
    app.run(debug=True)

"""
YOUR TOOLBOX (Person 4).  Every other script imports from this file.
Read it slowly once.  The ideas in here ARE your part of the project.
"""
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 1. FEATURES  (Person 3's recipe, copied exactly - same names, same ORDER)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "day_of_week", "day_of_month", "month", "quarter", "year", "is_weekend",
    "lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30",
]
MIN_HISTORY = 30          # the model needs the previous 30 days to build its clues


def feature_row(past_values, date):
    """
    The 12 clues for ONE day.  `past_values` = revenue of every day BEFORE `date`
    (oldest first).  Person 3 built the same features with shift()/rolling();
    this version gives identical numbers (P4_step0_check_handoff.py proves it).
    """
    past = np.asarray(past_values, dtype=float)
    if len(past) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} past days, got {len(past)}")
    date = pd.Timestamp(date)
    return {
        "day_of_week": date.dayofweek,          # Monday = 0 ... Sunday = 6
        "day_of_month": date.day,
        "month": date.month,
        "quarter": date.quarter,
        "year": date.year,
        "is_weekend": int(date.dayofweek >= 5),
        "lag_1": past[-1],                      # revenue yesterday
        "lag_7": past[-7],                      # revenue 7 days ago
        "lag_14": past[-14],
        "lag_30": past[-30],
        "rolling_mean_7": past[-7:].mean(),     # average of the last 7 days
        "rolling_mean_30": past[-30:].mean(),   # average of the last 30 days
    }


def make_features(past_values, date):
    """Same as feature_row, but a 1-row table with the columns in the right order."""
    return pd.DataFrame([feature_row(past_values, date)])[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# 2. LOADING
# ---------------------------------------------------------------------------
def load_series():
    """Daily revenue as a pandas Series, one value per day, oldest first."""
    df = pd.read_csv("data/daily_sales.csv", parse_dates=["order_date"])
    df = df.sort_values("order_date")
    return df.set_index("order_date")["daily_revenue"].rename("revenue")


def load_model(path):
    """Load a saved model and explain clearly if the library versions do not match."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = joblib.load(path)
        except Exception as e:
            raise SystemExit(
                f"\nCould not open {path}\n  reason: {type(e).__name__}: {e}\n"
                "This usually means your scikit-learn version differs from Person 3's.\n"
                "Fix: run  python P3_rebuild_models.py  (it rebuilds the models on your computer).\n")
    if any("InconsistentVersion" in type(w.message).__name__ for w in caught):
        print(f"  note: {path} was saved with a different scikit-learn version. "
              "It still loaded, but if numbers look odd run P3_rebuild_models.py.")
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1          # one row at a time is much faster on a single core
    return model


# ---------------------------------------------------------------------------
# 3. METRICS
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    """The numbers your report needs."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nonzero = y_true != 0
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),                       # average rupees/dollars wrong per day
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),              # like MAE, punishes big misses
        "MAPE": float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100),  # average % wrong
        "WAPE": float(np.abs(y_true - y_pred).sum() / y_true.sum() * 100),       # total error / total revenue, in %
        "R2": float(r2_score(y_true, y_pred)),                                   # 1 = perfect, 0 = no better than the average
    }


# ---------------------------------------------------------------------------
# 4. PREDICTING
# ---------------------------------------------------------------------------
def one_step_predictions(model, series, start_date):
    """
    'Easy mode': for each day from `start_date` on, the model may look at the REAL
    revenue of all earlier days.  This is how Person 3 measured accuracy.
    """
    values = series.to_numpy()
    out = {}
    for i, day in enumerate(series.index):
        if day >= pd.Timestamp(start_date):
            out[day] = float(model.predict(make_features(values[:i], day))[0])
    return pd.Series(out, name="predicted")


def recursive_forecast(model, past_values, last_date, horizon):
    """
    'Real mode' - forecasting days that have NO data yet.
    Predict day 1, pretend that prediction is real, use it to build the clues for
    day 2, and so on.  Small mistakes get carried forward - that is why long
    forecasts are less reliable.
    """
    history = list(map(float, past_values))
    days, values = [], []
    for step in range(1, horizon + 1):
        day = pd.Timestamp(last_date) + pd.Timedelta(days=step)
        yhat = max(float(model.predict(make_features(history, day))[0]), 0.0)   # revenue cannot be negative
        history.append(yhat)                       # <-- the "recursive" part
        days.append(day)
        values.append(yhat)
    return pd.Series(values, index=days, name="forecast")


# ---------------------------------------------------------------------------
# 5. BASELINES  (dumb forecasts the model must beat to be worth using)
# ---------------------------------------------------------------------------
def baselines_one_step(series, start_date):
    start = pd.Timestamp(start_date)
    train_average = series[series.index < start].mean()
    table = pd.DataFrame({
        "Yesterday's revenue": series.shift(1),
        "Same weekday last week": series.shift(7),
        "Average of last 30 days": series.shift(1).rolling(30).mean(),
    })
    table["Training-period average"] = train_average
    return table[table.index >= start]

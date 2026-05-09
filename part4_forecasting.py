# -*- coding: utf-8 -*-
"""
Part 4: Future AQI Forecasting
- Aggregates ds2 into monthly city-level series
- Trains SARIMA, Prophet, and XGBoost models per city
- Evaluates on 2024 hold-out, runs Diebold-Mariano test
- Builds ensemble forecast for next 18 months
- Feeds ensemble AQI into the health model for future health burden projections
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel

os.makedirs("outputs_part4", exist_ok=True)

# =========================================================
# LOAD DATA (from Part 3 pipeline)
# =========================================================

model        = joblib.load("outputs_part1/final_xgb_pipeline.pkl")
feature_cols = joblib.load("outputs_part1/feature_columns.pkl")

ds2 = pd.read_csv("data-set/indian_aqi_health_impact_2019_2024.csv")

# Repair date column
for col in ds2.columns:
    if "date" in col.lower():
        ds2["Date"] = pd.to_datetime(ds2[col])
        break
if "Date" not in ds2.columns:
    ds2["Date"] = pd.date_range("2019-01-01", periods=len(ds2), freq="D")

ds2["Month"] = ds2["Date"].dt.month
ds2["Year"]  = ds2["Date"].dt.year

# We need X from part1 for the health template row; rebuild a minimal version
ds1 = pd.read_excel("data-set/air_quality_health_dataset.csv.xlsx")
ds1['date'] = pd.to_datetime(ds1['date'])

# =========================================================
# HELPER: MASE + EVALUATE
# =========================================================

def mase(y_true, y_pred, y_train):
    naive = np.mean(np.abs(np.diff(y_train)))
    return np.mean(np.abs(y_true - y_pred)) / naive

def evaluate_ts(y_true, y_pred, y_train):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "MASE": mase(y_true, y_pred, y_train)
    }

def dm_test(e1, e2):
    return ttest_rel(e1, e2)

# =========================================================
# SARIMA AUTO-SELECT
# =========================================================

def sarima_auto(series):
    best_aic   = np.inf
    best_model = None

    for p in range(2):
        for d in range(2):
            for q in range(2):
                try:
                    m   = SARIMAX(series, order=(p, d, q), seasonal_order=(1, 1, 1, 12))
                    res = m.fit(disp=False)
                    if res.aic < best_aic:
                        best_aic   = res.aic
                        best_model = res
                except Exception:
                    continue

    return best_model

# =========================================================
# PROPHET WRAPPER
# =========================================================

def prophet_model(df):
    m = Prophet(yearly_seasonality=True)
    m.fit(df)
    return m

# =========================================================
# XGBOOST FEATURE CREATOR
# =========================================================

def create_features(df):
    df = df.copy()
    df['month'] = df['ds'].dt.month
    df['sin']   = np.sin(2 * np.pi * df['month'] / 12)
    df['cos']   = np.cos(2 * np.pi * df['month'] / 12)

    for lag in [1, 3, 6, 12]:
        df[f'lag_{lag}'] = df['y'].shift(lag)

    df['roll3'] = df['y'].rolling(3).mean()
    df['roll6'] = df['y'].rolling(6).mean()

    return df.dropna()

# =========================================================
# MONTHLY AGGREGATION + GAP FILL
# =========================================================

monthly = ds2.groupby(['City', pd.Grouper(key='Date', freq='ME')])['AQI'].mean().reset_index()
monthly.rename(columns={'Date': 'ds', 'AQI': 'y'}, inplace=True)

monthly_list = []
for city, group in monthly.groupby('City'):
    g          = group.set_index('ds').asfreq('ME')
    g['City']  = city
    g['y']     = g['y'].interpolate()
    monthly_list.append(g.reset_index())

monthly = pd.concat(monthly_list, ignore_index=True)
print(monthly.head())

# =========================================================
# CITY SELECTION (high pollution + increasing trend)
# =========================================================

trend_dict = {
    "Delhi": "Increasing", "Jaipur": "Increasing", "Chennai": "Increasing",
    "Pune": "Increasing", "Surat": "Increasing", "Rajkot": "Increasing",
    "Bhopal": "Increasing", "Srinagar": "Increasing", "Nashik": "Increasing",
    "Indore": "Increasing"
}

top_polluted = monthly.groupby('City')['y'].mean().sort_values(ascending=False)
top_cities   = [c for c in top_polluted.index if trend_dict.get(c) == "Increasing"][:5]

print(f"Selected {len(top_cities)} cities for forecasting: {top_cities}")

# =========================================================
# TRAIN + EVALUATE + FORECAST
# =========================================================

results    = []
forecasts  = {}
best_models = {}

for city in top_cities:
    df_city = monthly[monthly['City'] == city][['ds', 'y']].set_index('ds').sort_index()
    train   = df_city[df_city.index.year < 2024]
    test    = df_city[df_city.index.year == 2024]

    # --- SARIMA ---
    sar_model = sarima_auto(train['y'])
    sar_pred  = sar_model.forecast(len(test))

    # --- Prophet ---
    m         = prophet_model(train.reset_index().rename(columns={'ds': 'ds', 'y': 'y'}))
    prop_pred = (
        m.predict(m.make_future_dataframe(periods=len(test), freq='ME'))
        .set_index('ds')['yhat']
        .loc[test.index]
    )

    # --- XGBoost ---
    df_feat  = create_features(df_city.reset_index())
    train_f  = df_feat[df_feat['ds'].dt.year < 2024]
    test_f   = df_feat[df_feat['ds'].dt.year == 2024]
    xgb      = XGBRegressor(n_estimators=300)
    xgb.fit(train_f.drop(columns=['y', 'ds']), train_f['y'])
    xgb_pred = pd.Series(xgb.predict(test_f.drop(columns=['y', 'ds'])), index=test_f['ds'])

    # --- Align & Evaluate ---
    common_idx  = test.index.intersection(xgb_pred.index)
    y_true      = test.loc[common_idx, 'y']
    sar_aligned  = sar_pred.loc[common_idx]
    prop_aligned = prop_pred.loc[common_idx]
    xgb_aligned  = xgb_pred.loc[common_idx]

    sar_metrics  = evaluate_ts(y_true, sar_aligned,  train['y'])
    prop_metrics = evaluate_ts(y_true, prop_aligned, train['y'])
    xgb_metrics  = evaluate_ts(y_true, xgb_aligned,  train['y'])

    results.extend([
        [city, "SARIMA",  *sar_metrics.values()],
        [city, "Prophet", *prop_metrics.values()],
        [city, "XGBoost", *xgb_metrics.values()]
    ])

    # --- DM Test ---
    dm_stat, dm_p    = dm_test(y_true - xgb_aligned, y_true - sar_aligned)
    best_models[city] = "XGBoost" if dm_p < 0.05 else "Ensemble"

    # --- Recursive XGBoost 18-month forecast ---
    last        = df_city.reset_index().copy()
    xgb_future  = []
    for _ in range(18):
        pred = xgb.predict(create_features(last).iloc[-1:].drop(columns=['y', 'ds']))[0]
        last = pd.concat([
            last,
            pd.DataFrame({'ds': [last['ds'].max() + pd.DateOffset(months=1)], 'y': [pred]})
        ])
        xgb_future.append(pred)

    forecasts[city] = {
        "sarima":  sar_model.forecast(18).values,
        "prophet": m.predict(m.make_future_dataframe(periods=18, freq='ME')).set_index('ds')['yhat'].tail(18).values,
        "xgb":     np.array(xgb_future)
    }

results_df = pd.DataFrame(results, columns=['City', 'Model', 'RMSE', 'MAE', 'MAPE', 'MASE'])
print(results_df)

# =========================================================
# ENSEMBLE
# =========================================================

ensemble = {}
for city in forecasts:
    ensemble[city] = (
        forecasts[city]['sarima'] +
        forecasts[city]['prophet'] +
        forecasts[city]['xgb']
    ) / 3

# =========================================================
# HEALTH FORECAST (plug ensemble AQI → health model)
# =========================================================

# Build a minimal feature template from ds1
health_targets  = ['hospital_visits', 'respiratory_admissions', 'emergency_visits']
template_df     = ds1.dropna().reset_index(drop=True)

# Use the first row of ds1 as template (all non-AQI features stay fixed)
template_row    = pd.DataFrame([template_df.iloc[0]])
# Align columns to feature_cols
for col in feature_cols:
    if col not in template_row.columns:
        template_row[col] = 0
template_row = template_row[feature_cols]

health_outputs = []

for city in ensemble:
    for i, aqi_val in enumerate(ensemble[city]):
        row = template_row.copy()

        if 'AQI'   in row.columns: row['AQI']   = aqi_val
        if 'PM2.5' in row.columns: row['PM2.5'] = aqi_val * 0.6
        if 'PM10'  in row.columns: row['PM10']  = aqi_val * 0.8

        month = (i % 12) + 1
        if 'month_sin' in row.columns: row['month_sin'] = np.sin(2 * np.pi * month / 12)
        if 'month_cos' in row.columns: row['month_cos'] = np.cos(2 * np.pi * month / 12)

        pred = model.predict(row)[0]
        health_outputs.append([city, i, aqi_val, pred[0], pred[1], pred[2]])

health_df = pd.DataFrame(
    health_outputs,
    columns=['City', 'Month', 'AQI', 'Hospital', 'Resp', 'Emergency']
)
print(health_df.head())

# Save for Part 5
health_df.to_csv("outputs_part4/health_forecast.csv", index=False)

# =========================================================
# FINAL FORECAST PLOTS
# =========================================================

for city in ensemble:
    df_city = monthly[monthly['City'] == city]

    plt.figure(figsize=(10, 5))
    plt.plot(df_city['ds'], df_city['y'], label='Historical')

    future_dates = pd.date_range(df_city['ds'].max(), periods=18, freq='ME')
    plt.plot(future_dates, ensemble[city], label='Forecast')

    plt.title(city)
    plt.legend()
    plt.savefig(f"outputs_part4/{city}_forecast.png")
    plt.close()

print("Part 4 complete. Outputs saved to outputs_part4/")

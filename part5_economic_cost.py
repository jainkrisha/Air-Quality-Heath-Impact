# -*- coding: utf-8 -*-
"""
Part 5: Economic Cost Conversion
- Converts predicted health burden into INR costs (direct, indirect, mortality)
- Rebuilds scenarios A–D and calculates policy savings
- Applies costs to the 18-month health forecast from Part 4
- Compares all models (Ridge, RandomForest, LightGBM, XGBoost) on accuracy
- Saves all outputs to outputs_part5/
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

os.makedirs("outputs_part5", exist_ok=True)

# =========================================================
# LOAD ARTIFACTS FROM PREVIOUS PARTS
# =========================================================

final_model_pipeline = joblib.load("outputs_part1/final_xgb_pipeline.pkl")
feature_cols         = joblib.load("outputs_part1/feature_columns.pkl")

# Reconstruct X and y (same pipeline as Part 1/2)

def build_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['region', 'date'])

    pollutants = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    lags       = [1, 2, 3, 5, 7, 10, 14]

    for col in pollutants:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('region')[col].shift(lag)

    for col in ['AQI', 'PM2.5', 'PM10']:
        df[f'{col}_roll3'] = df.groupby('region')[col].transform(lambda x: x.rolling(3).mean())
        df[f'{col}_roll7'] = df.groupby('region')[col].transform(lambda x: x.rolling(7).mean())

    df['PM25_cum7']  = df.groupby('region')['PM2.5'].transform(lambda x: x.rolling(7).sum())
    df['PM25_cum14'] = df.groupby('region')['PM2.5'].transform(lambda x: x.rolling(14).sum())
    df['AQI_cum7']   = df.groupby('region')['AQI'].transform(lambda x: x.rolling(7).sum())
    df['AQI_cum14']  = df.groupby('region')['AQI'].transform(lambda x: x.rolling(14).sum())

    df['AQI_PM25_interact']    = df['AQI'] * df['PM2.5']
    df['PM25_PM10_ratio_safe'] = df['PM2.5'] / (df['PM10'] + 1e-6)
    df['AQI_humidity_interact']= df['AQI'] * df['humidity']

    df['month_num'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    df = df.dropna().reset_index(drop=True)
    return df

raw_df      = pd.read_excel("data-set/air_quality_health_dataset.csv.xlsx")
raw_targets = ['hospital_visits', 'respiratory_admissions', 'emergency_visits']

df = build_features(raw_df)

df['hospital_rate']  = df['hospital_visits']         / (df['population_density'] + 1e-6)
df['resp_rate']      = df['respiratory_admissions']   / (df['population_density'] + 1e-6)
df['emergency_rate'] = df['emergency_visits']         / (df['population_density'] + 1e-6)

targets = ['hospital_rate', 'resp_rate', 'emergency_rate']

y_raw = df[targets].copy()
y     = np.log1p(y_raw)

X = df.drop(columns=raw_targets + targets + ['date'])

# =========================================================
# COST PARAMETERS (INR)
# =========================================================

COST_HOSPITAL        = 12000
COST_EMERGENCY       = 3500
COST_OUTPATIENT      = 1200
OUTPATIENT_MULTIPLIER = 3

LOST_WORKDAY  = 650
SICK_DAYS     = 5
CAREGIVER_DAYS = 2

VSL            = 3.2e7
MORTALITY_RATE = 0.003

# =========================================================
# BASELINE COST (FROM PART 1 MODEL)
# =========================================================

baseline_preds  = final_model_pipeline.predict(X)
baseline_df     = pd.DataFrame(baseline_preds, columns=targets)
baseline_df['region'] = df['region'].values

hospital_col  = [c for c in baseline_df.columns if 'hosp' in c.lower()][0]
emergency_col = [c for c in baseline_df.columns if 'emer' in c.lower()][0]

def compute_costs(cost_df):
    hosp   = cost_df[hospital_col]
    emer   = cost_df[emergency_col]
    outpat = hosp * OUTPATIENT_MULTIPLIER

    direct   = (hosp * COST_HOSPITAL) + (emer * COST_EMERGENCY) + (outpat * COST_OUTPATIENT)
    indirect = (hosp * SICK_DAYS * LOST_WORKDAY) + (hosp * CAREGIVER_DAYS * LOST_WORKDAY)
    mortality = (hosp * MORTALITY_RATE) * VSL

    return pd.DataFrame({
        'direct':   direct,
        'indirect': indirect,
        'mortality': mortality,
        'total':    direct + indirect + mortality
    })

cost_df     = compute_costs(baseline_df)
baseline_df = pd.concat([baseline_df, cost_df], axis=1)
region_cost = baseline_df.groupby('region')[['direct', 'indirect', 'mortality', 'total']].sum()
print(region_cost)

# =========================================================
# COST BREAKDOWN PLOT
# =========================================================

plt.figure(figsize=(10, 6))
region_cost.plot(kind='bar', ax=plt.gca())
plt.title("Cost Breakdown by Region")
plt.ylabel("₹")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs_part5/cost_breakdown_region.png")
plt.close()

# =========================================================
# REBUILD SCENARIOS
# =========================================================

scenario_predictions = {}
X_base = X.copy()

X_A = X_base.copy()
for col, factor in zip(['AQI', 'PM2.5', 'PM10', 'NO2'], [0.9, 0.92, 0.92, 0.95]):
    if col in X_A.columns: X_A[col] *= factor
scenario_predictions["A"] = pd.DataFrame(final_model_pipeline.predict(X_A), columns=targets)

X_B = X_base.copy()
if 'PM2.5' in X_B.columns: X_B['PM2.5'] = np.minimum(X_B['PM2.5'], 60)
scenario_predictions["B"] = pd.DataFrame(final_model_pipeline.predict(X_B), columns=targets)

X_C = X_base.copy()
if 'vehicle_count'       in X_C.columns: X_C['vehicle_count']       *= 0.7
if 'industrial_activity' in X_C.columns: X_C['industrial_activity'] *= 0.8
scenario_predictions["C"] = pd.DataFrame(final_model_pipeline.predict(X_C), columns=targets)

X_D = X_base.copy()
for col, factor in zip(['AQI', 'PM2.5', 'PM10', 'NO2'], [0.9, 0.92, 0.92, 0.95]):
    if col in X_D.columns: X_D[col] *= factor
if 'vehicle_count'       in X_D.columns: X_D['vehicle_count']       *= 0.7
if 'industrial_activity' in X_D.columns: X_D['industrial_activity'] *= 0.8
scenario_predictions["D"] = pd.DataFrame(final_model_pipeline.predict(X_D), columns=targets)

# =========================================================
# POLICY SCENARIO SAVINGS
# =========================================================

scenario_summary = []

for scen, df_scen in scenario_predictions.items():
    delta      = baseline_df[[hospital_col, emergency_col]].values - df_scen.values[:, [0, 2]]
    h_avoided  = delta[:, 0].sum()
    e_avoided  = delta[:, 1].sum()
    o_avoided  = h_avoided * OUTPATIENT_MULTIPLIER

    direct_save   = (h_avoided * COST_HOSPITAL) + (e_avoided * COST_EMERGENCY) + (o_avoided * COST_OUTPATIENT)
    indirect_save = (h_avoided * SICK_DAYS * LOST_WORKDAY) + (h_avoided * CAREGIVER_DAYS * LOST_WORKDAY)
    mortality_save = h_avoided * MORTALITY_RATE * VSL

    scenario_summary.append([scen, h_avoided, direct_save, indirect_save, direct_save + indirect_save + mortality_save])

scenario_df = pd.DataFrame(
    scenario_summary,
    columns=['Scenario', 'Avoided Admissions', 'Direct ₹', 'Indirect ₹', 'Total ₹']
)
print(scenario_df)
scenario_df.to_csv("outputs_part5/scenario_savings.csv", index=False)

# =========================================================
# SCENARIO SAVINGS BAR PLOT
# =========================================================

scenario_df.plot(x='Scenario', y='Total ₹', kind='bar', figsize=(8, 5))
plt.title("Scenario Savings")
plt.ylabel("Total Savings (₹)")
plt.xlabel("Policy Scenario")
plt.tight_layout()
plt.savefig("outputs_part5/scenario_savings.png")
plt.close()

# =========================================================
# FORECAST COST (from Part 4 health_forecast.csv)
# =========================================================

health_df = pd.read_csv("outputs_part4/health_forecast.csv")

h_col = [c for c in health_df.columns if 'hosp' in c.lower()][0]
e_col = [c for c in health_df.columns if 'emer' in c.lower()][0]

health_df['outpatient'] = health_df[h_col] * OUTPATIENT_MULTIPLIER
health_df['direct']     = (health_df[h_col] * COST_HOSPITAL) + (health_df[e_col] * COST_EMERGENCY) + (health_df['outpatient'] * COST_OUTPATIENT)
health_df['indirect']   = (health_df[h_col] * SICK_DAYS * LOST_WORKDAY) + (health_df[h_col] * CAREGIVER_DAYS * LOST_WORKDAY)
health_df['total']      = health_df['direct'] + health_df['indirect']

print(health_df.head())
health_df.to_csv("outputs_part5/forecast_costs.csv", index=False)

# =========================================================
# MODEL COMPARISON (Ridge, RF, LightGBM, XGBoost)
# =========================================================

categorical_cols = ['region']
numerical_cols   = [col for col in X.columns if col not in categorical_cols]

multi_preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                        numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore'),  categorical_cols)
])

models = {
    "Ridge": Pipeline([
        ("prep",  multi_preprocessor),
        ("model", MultiOutputRegressor(Ridge()))
    ]),
    "RandomForest": Pipeline([
        ("prep",  multi_preprocessor),
        ("model", MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
    ]),
    "LightGBM": Pipeline([
        ("prep",  multi_preprocessor),
        ("model", MultiOutputRegressor(LGBMRegressor(n_estimators=100)))
    ]),
    "XGBoost": Pipeline([
        ("prep",  multi_preprocessor),
        ("model", MultiOutputRegressor(XGBRegressor(n_estimators=100, random_state=42)))
    ])
}

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

comparison_results = []

for name, mdl in models.items():
    print(f"\n🔹 Training {name}...")
    mdl.fit(X, y)
    preds = mdl.predict(X)

    rmse_list, mae_list, r2_list = [], [], []
    for i in range(len(targets)):
        rmse, mae, r2 = compute_metrics(y.iloc[:, i], preds[:, i])
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    comparison_results.append({
        "Model":       name,
        "RMSE":        np.mean(rmse_list),
        "MAE":         np.mean(mae_list),
        "R2":          np.mean(r2_list),
        "Accuracy (%)": np.mean(r2_list) * 100
    })

results_df = pd.DataFrame(comparison_results).sort_values(by="Accuracy (%)", ascending=False)

print("\n📊 MODEL ACCURACY TABLE:")
print(results_df)

best_model = results_df.iloc[0]
print("\n✅ BEST MODEL:")
print(best_model)

results_df.to_csv("outputs_part5/model_comparison.csv", index=False)

print("Part 5 complete. All outputs saved to outputs_part5/")

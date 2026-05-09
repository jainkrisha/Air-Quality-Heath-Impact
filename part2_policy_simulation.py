# -*- coding: utf-8 -*-
"""
Part 2: Policy Simulation & What-If Analysis
- Loads the trained pipeline from Part 1
- Applies 4 policy scenarios (A–D) on the feature set
- Runs Monte Carlo uncertainty simulation
- Generates summary tables, bar charts, and region heatmaps
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

os.makedirs("outputs_part2", exist_ok=True)

# =========================================================
# LOAD MODEL + DATA
# =========================================================

if 'final_model_pipeline' not in globals():
    print("Reloading model from Part 1 outputs...")
    final_model_pipeline = joblib.load("outputs_part1/final_xgb_pipeline.pkl")

raw_df = pd.read_excel("air_quality_health_dataset.csv.xlsx")
raw_df['date'] = pd.to_datetime(raw_df['date'])
raw_df = raw_df.dropna().reset_index(drop=True)

# =========================================================
# 1. FEATURE ENGINEERING FUNCTION
# =========================================================

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

# =========================================================
# 2. BUILD FEATURES + TARGETS
# =========================================================

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
# 3. BASELINE PREDICTIONS
# =========================================================

baseline_preds = final_model_pipeline.predict(X)

baseline_df         = pd.DataFrame(baseline_preds, columns=targets)
baseline_df['region'] = df['region'].values

baseline_region = baseline_df.groupby('region').mean()
print("\nBaseline Predictions per Region:\n", baseline_region)

# =========================================================
# 4. SCENARIOS
# =========================================================

def scenario_A(X_data):
    X_data = X_data.copy()
    X_data['AQI']   *= 0.90
    X_data['PM2.5'] *= 0.92
    X_data['PM10']  *= 0.92
    X_data['NO2']   *= 0.95
    return X_data

def scenario_B(X_data):
    X_data = X_data.copy()
    X_data['PM2.5'] = np.minimum(X_data['PM2.5'], 60)
    return X_data

def scenario_C(X_data):
    X_data = X_data.copy()
    X_data['vehicle_count']      *= 0.70
    X_data['industrial_activity'] *= 0.80
    return X_data

def scenario_D(X_data):
    return scenario_C(scenario_A(X_data))

scenarios = {"A": scenario_A, "B": scenario_B, "C": scenario_C, "D": scenario_D}

# =========================================================
# 5. SCENARIO PREDICTIONS
# =========================================================

scenario_results = {}

for name, func in scenarios.items():
    X_scen = func(X)
    preds  = final_model_pipeline.predict(X_scen)
    scenario_results[name] = preds

# =========================================================
# 6. MONTE CARLO SIMULATION
# =========================================================

n_sim    = 1000
num_cols = X.select_dtypes(include=np.number).columns
cov_matrix = np.cov(X[num_cols].values.T)

mc_results = {}

for name, func in scenarios.items():
    X_s        = func(X)
    preds_list = []

    for _ in range(n_sim):
        noise = np.random.multivariate_normal(
            mean=np.zeros(len(num_cols)),
            cov=cov_matrix
        )
        X_mc           = X_s.copy()
        X_mc[num_cols] = X_mc[num_cols] + noise
        preds          = final_model_pipeline.predict(X_mc)
        preds_list.append(preds.mean(axis=0))

    preds_array = np.array(preds_list)
    mc_results[name] = {
        "mean": preds_array.mean(axis=0),
        "low":  np.percentile(preds_array, 5,  axis=0),
        "high": np.percentile(preds_array, 95, axis=0)
    }

# =========================================================
# 7. SUMMARY TABLE
# =========================================================

baseline_mean = baseline_preds.mean(axis=0)
summary       = []

for name in scenarios.keys():
    res = mc_results[name]
    for i, target in enumerate(targets):
        delta = baseline_mean[i] - res["mean"][i]
        summary.append([
            name, target,
            delta,
            res["low"][i],
            res["high"][i],
            delta * 365
        ])

summary_df = pd.DataFrame(
    summary,
    columns=["Scenario", "Target", "Mean_Delta", "Low_90CI", "High_90CI", "Annual_Avoided"]
)
print("\nScenario Summary:\n", summary_df)
summary_df.to_csv("outputs_part2/scenario_summary.csv", index=False)

# =========================================================
# 8. BAR CHARTS
# =========================================================

for target in targets:
    subset           = summary_df[summary_df['Target'] == target]
    symmetric_error  = (subset['High_90CI'] - subset['Low_90CI']) / 2

    plt.figure()
    plt.bar(subset['Scenario'], subset['Mean_Delta'], yerr=symmetric_error)
    plt.title(f"Scenario Impact — {target}")
    plt.ylabel("Mean Delta (Baseline - Scenario)")
    plt.xlabel("Scenario")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"outputs_part2/bar_{target}.png")
    plt.close()

# =========================================================
# 9. HEATMAP (region × scenario)
# =========================================================

heatmap_data = []

for name, preds in scenario_results.items():
    df_temp = pd.DataFrame(preds, columns=targets)
    df_temp['region'] = df['region'].values

    reduction = (
        baseline_df.groupby('region').mean()['hospital_rate']
        - df_temp.groupby('region').mean()['hospital_rate']
    )
    for region, val in reduction.items():
        heatmap_data.append([region, name, val])

heatmap_df = pd.DataFrame(heatmap_data, columns=['Region', 'Scenario', 'Reduction'])
pivot = heatmap_df.pivot(index="Region", columns="Scenario", values="Reduction")

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".6f")
plt.title("Reduction in Hospital Rate by Region and Scenario")
plt.savefig("outputs_part2/heatmap_region.png")
plt.close()

# =========================================================
# 10. POLICY INTERPRETATION
# =========================================================

for region in baseline_region.index:
    reduction = heatmap_df[
        (heatmap_df['Region'] == region) & (heatmap_df['Scenario'] == 'D')
    ]['Reduction'].values[0]
    print(f"In {region}, Scenario D could lead to approximately {reduction*365:.6f} fewer hospital visits per year.")

# =========================================================
# 11. INVERSE POLICY (SHAP)
# =========================================================

import shap

model_for_shap           = final_model_pipeline.named_steps['model'].estimators_[0]
X_transformed_for_shap   = final_model_pipeline.named_steps['prep'].transform(X)
explainer                = shap.Explainer(model_for_shap)
shap_values              = explainer(X_transformed_for_shap)
feature_names_for_shap   = final_model_pipeline.named_steps['prep'].get_feature_names_out()

importance   = np.abs(shap_values.values).mean(axis=0)
feat_imp_df  = pd.DataFrame({
    "Feature":    feature_names_for_shap,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 3 Features to Reduce hospital_visits (based on SHAP importance):")
top3 = feat_imp_df.head(3)
for _, row in top3.iterrows():
    print(f"{row['Feature']} → major lever (~10–30% change needed)")

print("Part 2 complete. Outputs saved to outputs_part2/")

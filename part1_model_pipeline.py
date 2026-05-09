# -*- coding: utf-8 -*-
"""
Part 1: Air Quality & Health Model Pipeline
- Loads data, engineers features, trains Ridge / RandomForest / XGBoost models
- Evaluates with GroupKFold cross-validation
- Generates SHAP explanations
- Saves the final pipeline to outputs_part1/
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import joblib
import scipy.stats as stats

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, spearmanr

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

os.makedirs("outputs_part1", exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================

data_file_path = r"D:\Air-Quality-Heath-Impact\data-set\air_quality_health_dataset.csv.xlsx"
raw_health_data = pd.read_excel(data_file_path)

raw_health_data['date'] = pd.to_datetime(raw_health_data['date'])
health_df = raw_health_data.sort_values(['region', 'date']).reset_index(drop=True)

health_targets = ['hospital_visits', 'respiratory_admissions', 'emergency_visits']

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

air_pollutants = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
lag_days = [1, 2, 3, 5, 7, 10, 14]

# Lagged features
for p_col in air_pollutants:
    for lag_val in lag_days:
        health_df[f'{p_col}_lag{lag_val}'] = health_df.groupby('region')[p_col].shift(lag_val)

# Rolling means
for roll_col in ['AQI', 'PM2.5', 'PM10']:
    health_df[f'{roll_col}_roll3'] = health_df.groupby('region')[roll_col].transform(lambda x: x.rolling(3).mean())
    health_df[f'{roll_col}_roll7'] = health_df.groupby('region')[roll_col].transform(lambda x: x.rolling(7).mean())

# Cumulative exposure
health_df['PM25_cum7']  = health_df.groupby('region')['PM2.5'].transform(lambda x: x.rolling(7).sum())
health_df['PM25_cum14'] = health_df.groupby('region')['PM2.5'].transform(lambda x: x.rolling(14).sum())
health_df['AQI_cum7']   = health_df.groupby('region')['AQI'].transform(lambda x: x.rolling(7).sum())
health_df['AQI_cum14']  = health_df.groupby('region')['AQI'].transform(lambda x: x.rolling(14).sum())

# Interaction terms
health_df['AQI_PM25_interact']    = health_df['AQI'] * health_df['PM2.5']
health_df['PM25_PM10_ratio_safe'] = health_df['PM2.5'] / (health_df['PM10'] + 1e-6)
health_df['AQI_humidity_interact']= health_df['AQI'] * health_df['humidity']

# Seasonal features
health_df['month_num'] = health_df['date'].dt.month
health_df['month_sin'] = np.sin(2 * np.pi * health_df['month_num'] / 12)
health_df['month_cos'] = np.cos(2 * np.pi * health_df['month_num'] / 12)

health_df = health_df.dropna().reset_index(drop=True)

# =========================================================
# 3. TARGET TRANSFORMATION
# =========================================================

health_df['hospital_rate']  = health_df['hospital_visits']         / (health_df['population_density'] + 1e-6)
health_df['resp_rate']      = health_df['respiratory_admissions']   / (health_df['population_density'] + 1e-6)
health_df['emergency_rate'] = health_df['emergency_visits']         / (health_df['population_density'] + 1e-6)

health_outcome_rates = ['hospital_rate', 'resp_rate', 'emergency_rate']
raw_health_rates     = health_df[health_outcome_rates].copy()
transformed_health_y = np.log1p(raw_health_rates)

features_for_model = health_df.drop(columns=[
    'hospital_visits', 'respiratory_admissions', 'emergency_visits',
    'hospital_rate', 'resp_rate', 'emergency_rate',
    'date'
])

region_grouping   = health_df['region']
categorical_feats = ['region']
numeric_feats     = [c for c in features_for_model.columns if c not in categorical_feats]

final_data_preprocessor = ColumnTransformer([
    ('cat_process', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_feats),
    ('num_process', 'passthrough', numeric_feats)
])

# Set globals for modeling
X          = features_for_model.copy()
y          = transformed_health_y.copy()
groups     = region_grouping.copy()
preprocessor = final_data_preprocessor

# =========================================================
# 4. METRICS FUNCTION
# =========================================================

def evaluate_health(true_logs, pred_logs):
    """Calculates RMSE, MAE, MAPE, R2 on original and log scale."""
    actual_rates    = np.expm1(true_logs)
    predicted_rates = np.expm1(pred_logs)
    all_metrics = {}

    for idx, target_name in enumerate(health_outcome_rates):
        rmse_val          = np.sqrt(mean_squared_error(actual_rates.iloc[:, idx], predicted_rates[:, idx]))
        mae_val           = mean_absolute_error(actual_rates.iloc[:, idx], predicted_rates[:, idx])
        mape_val          = np.mean(np.abs((actual_rates.iloc[:, idx] - predicted_rates[:, idx]) / (actual_rates.iloc[:, idx] + 1e-6))) * 100
        r2_on_orig_scale  = r2_score(actual_rates.iloc[:, idx], predicted_rates[:, idx])
        r2_on_log_scale   = r2_score(true_logs.iloc[:, idx], pred_logs[:, idx])
        all_metrics[target_name] = [rmse_val, mae_val, mape_val, r2_on_orig_scale, r2_on_log_scale]

    return all_metrics

# =========================================================
# 5. BASE MODELS
# =========================================================

our_models = {
    "Ridge":        MultiOutputRegressor(Ridge()),
    "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
}

# =========================================================
# 6. XGBOOST OPTUNA TUNING
# =========================================================

def objective(trial):
    xgb_tune_params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": 42,
        "tree_method":  "hist",
        "verbosity":    0
    }
    temp_xgb_model = MultiOutputRegressor(XGBRegressor(**xgb_tune_params))
    cv_folds = GroupKFold(n_splits=5)
    rmses = []

    for train_idx, val_idx in cv_folds.split(X, y, groups):
        pipe = Pipeline([("data_prep", preprocessor), ("model_runner", temp_xgb_model)])
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], pipe.predict(X.iloc[val_idx])))
        rmses.append(fold_rmse)

    return np.mean(rmses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

best_xgb_params = study.best_params
final_tuned_xgb_model = MultiOutputRegressor(
    XGBRegressor(**best_xgb_params, random_state=42, tree_method="hist", verbosity=0)
)
our_models["XGBoost"] = final_tuned_xgb_model
print(f"Best XGBoost parameters: {best_xgb_params}")

# =========================================================
# 7. TRAIN + EVALUATE ALL MODELS
# =========================================================

results = []

for name, model in our_models.items():
    gkf       = GroupKFold(n_splits=5)
    all_preds = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        all_preds[test_idx] = pipe.predict(X.iloc[test_idx])

    metrics = evaluate_health(y, all_preds)
    for target in health_outcome_rates:
        results.append([name, target] + metrics[target])

results_df = pd.DataFrame(results, columns=["Model", "Target", "RMSE", "MAE", "MAPE", "R2_original", "R2_log"])
print(results_df)

# =========================================================
# 8. FINAL TRAINING
# =========================================================

final_model_pipeline = Pipeline([
    ("prep",  preprocessor),
    ("model", final_tuned_xgb_model)
])
final_model_pipeline.fit(X, y)

joblib.dump(final_model_pipeline,    "outputs_part1/final_xgb_pipeline.pkl")
joblib.dump(X.columns.tolist(),      "outputs_part1/feature_columns.pkl")

# =========================================================
# 8b. PREDICTED VS ACTUAL PLOTS
# =========================================================

preds = final_model_pipeline.predict(X)

for i, target in enumerate(health_outcome_rates):
    plt.figure(figsize=(6, 6))
    plt.scatter(y.iloc[:, i], preds[:, i], alpha=0.5)

    min_val = min(y.iloc[:, i].min(), preds[:, i].min())
    max_val = max(y.iloc[:, i].max(), preds[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.xlabel("Actual (Log-Transformed Rate)")
    plt.ylabel("Predicted (Log-Transformed Rate)")
    plt.title(f"Predicted vs Actual — {target} (Log Scale)")
    plt.savefig(f"outputs_part1/pred_vs_actual_{target}.png")
    plt.close()

# =========================================================
# 9. SHAP EXPLAINABILITY
# =========================================================

X_transformed = preprocessor.transform(X)

sample_size = min(500, X_transformed.shape[0])
sample_idx  = np.random.choice(X_transformed.shape[0], sample_size, replace=False)

X_sample = X_transformed[sample_idx]
y_sample = y.iloc[sample_idx]

feature_names = (
    preprocessor.named_transformers_['cat_process']
    .get_feature_names_out(categorical_feats).tolist()
    + numeric_feats
)

xgb_single  = final_model_pipeline.named_steps['model'].estimators_[0]
explainer   = shap.Explainer(xgb_single, X_sample)
shap_values = explainer(X_sample)

shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.savefig("outputs_part1/shap_summary_hospital_rate.png")
plt.close()

if "AQI" in feature_names:
    shap.dependence_plot("AQI", shap_values.values, X_sample, feature_names=feature_names, show=False)
    plt.savefig("outputs_part1/shap_dependence_AQI.png")
    plt.close()

scenario_modified_features = ['AQI', 'PM2.5', 'PM10', 'NO2', 'vehicle_count', 'industrial_activity']

for feature in scenario_modified_features:
    if feature in feature_names:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values.values, X_sample, feature_names=feature_names, show=False)
        plt.title(f"SHAP Dependence Plot for {feature}")
        plt.savefig(f"outputs_part1/shap_dependence_{feature}.png")
        plt.close()
    else:
        print(f"Warning: Feature '{feature}' not found in `feature_names`. Skipping dependence plot.")

print("Generated SHAP dependence plots for scenario-modified features.")

# Force plots for top 3 high-burden samples
top_idx = np.argsort(y_sample['hospital_rate'])[-3:]

for i, idx in enumerate(top_idx):
    shap.force_plot(
        explainer.expected_value,
        shap_values.values[idx],
        X_sample[idx],
        matplotlib=True,
        show=False
    )
    plt.savefig(f"outputs_part1/force_plot_{i}.png")
    plt.close()

print("Part 1 complete. Outputs saved to outputs_part1/")

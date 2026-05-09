# -*- coding: utf-8 -*-
"""
Part 3: Real-World Indian City Data Validation
- Loads the Indian AQI dataset (ds2) and the original dataset (ds1)
- Visualises AQI distributions, seasonal trends, and city rankings
- Runs distribution-shift tests (KS, Jensen-Shannon)
- Transfers the Part-1 model to ds2 and validates with Bland-Altman & Spearman
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, spearmanr

os.makedirs("outputs_part3", exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================

ds1 = pd.read_excel("data-set/air_quality_health_dataset.csv.xlsx")
ds2 = pd.read_csv("data-set/indian_aqi_health_impact_2019_2024.csv")

model        = joblib.load("outputs_part1/final_xgb_pipeline.pkl")
feature_cols = joblib.load("outputs_part1/feature_columns.pkl")

# =========================================================
# 2. CLEAN DATE COLUMN IN DS2
# =========================================================

for col in ds2.columns:
    if "date" in col.lower():
        ds2["Date"] = pd.to_datetime(ds2[col])
        break

if "Date" not in ds2.columns:
    print("Warning: No explicit date column found. Generating default daily dates.")
    ds2["Date"] = pd.date_range("2019-01-01", periods=len(ds2), freq="D")

ds2["Month"] = ds2["Date"].dt.month
ds2["Year"]  = ds2["Date"].dt.year

# =========================================================
# 3. AQI DISTRIBUTION BY CITY
# =========================================================

city_order = ds2.groupby("City")["AQI"].median().sort_values().index

plt.figure(figsize=(12, 6))
sns.boxplot(data=ds2, x="City", y="AQI", order=city_order)
plt.xticks(rotation=90)
plt.title("AQI Distribution by City")
plt.tight_layout()
plt.savefig("outputs_part3/city_boxplot.png")
plt.close()

# =========================================================
# 4. MONTHLY SEASONAL AQI
# =========================================================

monthly = ds2.groupby("Month")["AQI"].mean()

plt.figure(figsize=(8, 5))
monthly.plot(marker="o")
plt.title("Monthly Seasonal AQI Trend")
plt.ylabel("AQI")
plt.savefig("outputs_part3/monthly_aqi.png")
plt.close()

# =========================================================
# 5. TOP / BOTTOM 5 CITIES
# =========================================================

ranked  = ds2.groupby("City")["AQI"].mean().sort_values()
top5    = ranked.tail(5)
bottom5 = ranked.head(5)

print("\nTop 5 Most Polluted Cities:\n",  top5)
print("\nBottom 5 Cleanest Cities:\n", bottom5)

# =========================================================
# 6. CORRELATION MATRIX
# =========================================================

pollutants = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

plt.figure(figsize=(8, 6))
sns.heatmap(ds2[pollutants].corr(), annot=True, cmap="coolwarm")
plt.title("Pollutant Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs_part3/correlation_matrix.png")
plt.close()

# =========================================================
# 7. HEALTH BURDEN SCORE
# =========================================================

def burden_band(aqi):
    if aqi <= 50:   return 0
    elif aqi <= 100: return 1
    elif aqi <= 150: return 2
    elif aqi <= 200: return 3
    elif aqi <= 300: return 4
    else:            return 5

ds2["health_burden"]          = ds2["AQI"].apply(burden_band)
ds2["health_burden_weighted"] = ds2["health_burden"] * (ds2["PM2.5"] / 60)

# =========================================================
# 8. DISTRIBUTION SHIFT TESTS
# =========================================================

shared = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']

shift_results = []
for col in shared:
    stat, p = ks_2samp(ds1[col].dropna(), ds2[col].dropna())
    shift_results.append([col, stat, p])

shift_df = pd.DataFrame(shift_results, columns=["Feature", "KS_Stat", "p_value"])
print("\nDistribution Shift Test:\n", shift_df)

h1 = np.histogram(ds1["AQI"], bins=30, density=True)[0]
h2 = np.histogram(ds2["AQI"], bins=30, density=True)[0]
js = jensenshannon(h1, h2)
print("\nAQI Jensen-Shannon Divergence:", js)

# =========================================================
# 9. VIOLIN PLOTS
# =========================================================

for col in shared:
    temp = pd.DataFrame({
        col:       pd.concat([ds1[col], ds2[col]], ignore_index=True),
        "Dataset": ["DS1"] * len(ds1) + ["DS2"] * len(ds2)
    })
    plt.figure()
    sns.violinplot(data=temp, x="Dataset", y=col)
    plt.title(f"{col} Distribution")
    plt.savefig(f"outputs_part3/violin_{col}.png")
    plt.close()

# =========================================================
# 10. MODEL TRANSFER VALIDATION
# =========================================================

for col in feature_cols:
    if col not in ds2.columns:
        if col == 'region':
            ds2[col] = 'unknown_region'
        else:
            ds2[col] = 0

if 'region' in ds2.columns:
    ds2['region'] = ds2['region'].astype(str)

X2   = ds2[feature_cols]
preds = model.predict(X2)

pred_df = pd.DataFrame(preds, columns=[
    "hospital_visits_pred",
    "respiratory_pred",
    "emergency_pred"
])
ds2 = pd.concat([ds2, pred_df], axis=1)

city_pred = ds2.groupby("City")["hospital_visits_pred"].mean().sort_values(ascending=False)
print("\nHighest Predicted Burden Cities:\n", city_pred.head(10))

rho, p = spearmanr(ds2["AQI"], ds2["hospital_visits_pred"])
print("\nSpearman Correlation AQI vs Predicted Burden:")
print("rho =", rho, "p =", p)

# =========================================================
# 11. BLAND-ALTMAN PLOT
# =========================================================

mean_vals = (ds2["AQI"] + ds2["hospital_visits_pred"]) / 2
diff_vals = ds2["AQI"] - ds2["hospital_visits_pred"]

plt.figure(figsize=(8, 5))
plt.scatter(mean_vals, diff_vals, alpha=0.4)
plt.axhline(diff_vals.mean(), linestyle="--")
plt.title("Bland-Altman Plot")
plt.xlabel("Mean")
plt.ylabel("Difference")
plt.savefig("outputs_part3/bland_altman.png")
plt.close()

# =========================================================
# 12. TEMPORAL TREND (TOP 5 CITIES)
# =========================================================

top5cities = top5.index.tolist()

for city in top5cities:
    temp   = ds2[ds2["City"] == city]
    yearly = temp.groupby("Year")["AQI"].mean()

    plt.figure()
    yearly.plot(marker="o")
    plt.title(f"{city} AQI Trend")

    if 2020 in yearly.index:
        plt.axvline(2020, linestyle="--")
        plt.text(2020, yearly.max(), "COVID Lockdown")

    plt.savefig(f"outputs_part3/trend_{city}.png")
    plt.close()

# =========================================================
# 13. TREND DIRECTION (ALL CITIES)
# =========================================================

print("\nTrend Direction:")

for city in ds2["City"].unique():
    temp   = ds2[ds2["City"] == city]
    yearly = temp.groupby("Year")["AQI"].mean()

    if len(yearly) >= 3:
        slope = np.polyfit(yearly.index, yearly.values, 1)[0]
        trend = "Increasing" if slope > 0 else ("Decreasing" if slope < 0 else "Stable")
        print(city, ":", trend)

print("Part 3 complete. Outputs saved to outputs_part3/")

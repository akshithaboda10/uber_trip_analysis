# uber_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import warnings
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# --- Load and Concatenate Data ---
print("🔄 Loading data from ../data/")
files = glob.glob("../data/uber-raw-data-*.csv")
if not files:
    raise FileNotFoundError("🚫 No CSV files found in ../data/. Please check file paths.")
dfs = [pd.read_csv(file) for file in files]
df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {df.shape[0]} rows from {len(files)} files")

# --- Preprocess ---
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df['Hour'] = df['Date/Time'].dt.hour
df['DayOfWeek'] = df['Date/Time'].dt.dayofweek
df['Date'] = df['Date/Time'].dt.floor('H')

# --- Aggregate Hourly ---
hourly_trips = df.groupby('Date').size().reset_index(name='Count')
hourly_trips.set_index('Date', inplace=True)

# --- Seasonal Decomposition Plot ---
print("📈 Creating seasonal decomposition plot")
decompose_result = seasonal_decompose(hourly_trips['Count'], model='additive', period=24)
fig = decompose_result.plot()
plt.suptitle("Seasonal Decomposition", fontsize=16)
plt.tight_layout()
os.makedirs("../output/plots", exist_ok=True)
plt.savefig("../output/plots/seasonal_decomposition.png")
plt.close()

# --- Time Series Forecasting ---
train = hourly_trips.loc[:'2014-09-15']
test = hourly_trips.loc['2014-09-16':]

def create_lagged_features(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

window = 24
X_train, y_train = create_lagged_features(train['Count'].values, window)
test_data = np.concatenate([train['Count'].values[-window:], test['Count'].values])
X_test, y_test = create_lagged_features(test_data, window)

# --- Train Models ---
print("🤖 Training models...")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

gbr = GradientBoostingRegressor(n_estimators=300)
gbr.fit(X_train, y_train)
gbr_preds = gbr.predict(X_test)

# --- Evaluate ---
xgb_mape = mean_absolute_percentage_error(y_test, xgb_preds)
rf_mape = mean_absolute_percentage_error(y_test, rf_preds)
gbr_mape = mean_absolute_percentage_error(y_test, gbr_preds)

print(f"📊 XGBoost MAPE: {xgb_mape:.2%}")
print(f"📊 Random Forest MAPE: {rf_mape:.2%}")
print(f"📊 Gradient Boosting MAPE: {gbr_mape:.2%}")

# --- Ensemble ---
weights = np.array([1/xgb_mape, 1/rf_mape, 1/gbr_mape])
weights /= weights.sum()
ensemble_preds = weights[0]*xgb_preds + weights[1]*rf_preds + weights[2]*gbr_preds
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_preds)

print(f"⭐ Ensemble MAPE: {ensemble_mape:.2%}")

# --- Plot Predictions ---
print("📉 Saving prediction comparison plot")
plt.figure(figsize=(14, 6))
plt.plot(test.index[:len(y_test)], y_test, label="Actual", color="black")
plt.plot(test.index[:len(xgb_preds)], xgb_preds, '--', label="XGBoost")
plt.plot(test.index[:len(rf_preds)], rf_preds, '--', label="Random Forest")
plt.plot(test.index[:len(gbr_preds)], gbr_preds, '--', label="GBR")
plt.plot(test.index[:len(ensemble_preds)], ensemble_preds, label="Ensemble", color="purple")
plt.title("Actual vs Predicted Uber Trips (Hourly)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../output/plots/model_predictions.png")
plt.show()

# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose

import os
import glob
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# --- Streamlit page setup ---
st.set_page_config(page_title="Uber Trip Analysis", layout="wide")
st.title("🚕 Uber Trip Analysis (NYC 2014)")
st.markdown("Analyze and forecast Uber trip demand using 2014 pickup data.")

# --- File loader ---
st.sidebar.header("📂 Data Source")
files = glob.glob("../data/uber-raw-data-*.csv")
st.sidebar.write("Files found:", files)


if not files:
    st.sidebar.warning("No CSVs found in /data/. Please upload them.")
    uploaded_files = st.sidebar.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with open(f"data/{file.name}", "wb") as f:
                f.write(file.read())
        st.sidebar.success("Files uploaded. Refresh the page.")
        st.stop()
else:
    st.sidebar.success(f"{len(files)} files loaded from /data/")

# --- Load data ---
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# --- Preprocess ---
df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df['Hour'] = df['Date/Time'].dt.hour
df['DayOfWeek'] = df['Date/Time'].dt.dayofweek
df['Date'] = df['Date/Time'].dt.floor('H')

# --- EDA ---
st.subheader("📊 Exploratory Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Trips by Hour**")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.countplot(x='Hour', data=df, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.markdown("**Trips by Day of Week**")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.countplot(x='DayOfWeek', data=df, ax=ax2)
    st.pyplot(fig2)

# --- Time Series Aggregation ---
hourly_trips = df.groupby('Date').size().reset_index(name='Count')
hourly_trips.set_index('Date', inplace=True)

# --- Forecasting Section ---
st.subheader("⏳ Forecasting Trip Demand")

# Train/test split
train = hourly_trips.loc[:'2014-09-15']
test = hourly_trips.loc['2014-09-16':]

def create_lagged_features(data, window=24):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

window = 24
X_train, y_train = create_lagged_features(train['Count'].values, window)
test_data = np.concatenate([train['Count'].values[-window:], test['Count'].values])
X_test, y_test = create_lagged_features(test_data, window)

# Model training
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

gbr = GradientBoostingRegressor(n_estimators=300)
gbr.fit(X_train, y_train)
gbr_preds = gbr.predict(X_test)

# Evaluation
xgb_mape = mean_absolute_percentage_error(y_test, xgb_preds)
rf_mape = mean_absolute_percentage_error(y_test, rf_preds)
gbr_mape = mean_absolute_percentage_error(y_test, gbr_preds)

st.markdown(f"📈 **Model Performance:**")
st.markdown(f"- XGBoost MAPE: `{xgb_mape:.2%}`")
st.markdown(f"- Random Forest MAPE: `{rf_mape:.2%}`")
st.markdown(f"- Gradient Boosting MAPE: `{gbr_mape:.2%}`")

# Ensemble
weights = np.array([1/xgb_mape, 1/rf_mape, 1/gbr_mape])
weights /= weights.sum()
ensemble_preds = weights[0]*xgb_preds + weights[1]*rf_preds + weights[2]*gbr_preds
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_preds)
st.markdown(f"- **Ensemble MAPE:** `{ensemble_mape:.2%}`")

# Plot
st.markdown("📉 **Actual vs Predicted Trips**")
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(test.index[:len(y_test)], y_test, label="Actual", color="black")
ax3.plot(test.index[:len(xgb_preds)], xgb_preds, label="XGBoost", linestyle="--")
ax3.plot(test.index[:len(rf_preds)], rf_preds, label="Random Forest", linestyle="--")
ax3.plot(test.index[:len(gbr_preds)], gbr_preds, label="GBR", linestyle="--")
ax3.plot(test.index[:len(ensemble_preds)], ensemble_preds, label="Ensemble", color="purple")
ax3.set_title("Actual vs Predicted Uber Trips")
ax3.legend()
plt.xticks(rotation=45)
st.pyplot(fig3)

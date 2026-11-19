
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Earthquake Forecasting Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Earthquake Forecasting Dashboard")

st.markdown("""
Analyze historical earthquake data and forecast future activity
using Time Series Analysis.
""")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload Earthquake CSV Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload your earthquake dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")

st.dataframe(df.head())

# Date column detection
date_columns = [
    c for c in df.columns
    if c.lower() in ["date", "time", "datetime", "timestamp"]
]

if not date_columns:
    st.error("Date column not found.")
    st.stop()

date_column = st.selectbox(
    "Select Date Column",
    date_columns
)

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

df = df.dropna(subset=[date_column])

# Numeric target selection
numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if not numeric_columns:
    st.error("No numeric columns found.")
    st.stop()

target_column = st.selectbox(
    "Select Forecast Variable",
    numeric_columns
)

# Prepare time series
ts = (
    df.set_index(date_column)[target_column]
    .resample("D")
    .mean()
    .interpolate()
    .dropna()
)

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Observations", len(ts))

col2.metric("Start Date", str(ts.index.min().date()))

col3.metric("End Date", str(ts.index.max().date()))

# Historical chart
st.subheader("Historical Time Series")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(ts.index, ts.values)

ax.set_xlabel("Date")

ax.set_ylabel(target_column)

ax.set_title("Historical Data")

ax.grid(True)

st.pyplot(fig)

# Forecast settings
st.subheader("Forecast Settings")

forecast_days = st.slider(
    "Forecast Horizon (Days)",
    min_value=7,
    max_value=180,
    value=60
)

if len(ts) < 30:

    st.warning("At least 30 observations are recommended.")

else:

    try:

        # Train/test split
        train_size = int(len(ts) * 0.8)

        train = ts.iloc[:train_size]

        test = ts.iloc[train_size:]

        # ARIMA model
        model = ARIMA(
            train,
            order=(2, 1, 1)
        )

        model_fit = model.fit()

        predictions = model_fit.forecast(
            steps=len(test)
        )

        # Evaluation
        mae = mean_absolute_error(
            test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                test,
                predictions
            )
        )

        col1, col2 = st.columns(2)

        col1.metric("MAE", round(mae, 4))

        col2.metric("RMSE", round(rmse, 4))

        # Actual vs predicted
        st.subheader("Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(train.index, train, label="Training")

        ax.plot(test.index, test, label="Actual")

        ax.plot(test.index, predictions, label="Predicted")

        ax.legend()

        ax.grid(True)

        st.pyplot(fig)

        # Future forecasting
        final_model = ARIMA(
            ts,
            order=(2, 1, 1)
        )

        final_model_fit = final_model.fit()

        future_forecast = final_model_fit.forecast(
            steps=forecast_days
        )

        future_dates = pd.date_range(
            start=ts.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": np.asarray(future_forecast)
        })

        # Forecast graph
        st.subheader("Future Forecast")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            ts.index,
            ts.values,
            label="Historical"
        )

        ax.plot(
            future_dates,
            future_forecast,
            linestyle="--",
            label="Forecast"
        )

        ax.legend()

        ax.grid(True)

        st.pyplot(fig)

        # Table
        st.subheader("Forecast Data")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        # Download forecast
        csv = forecast_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Forecast CSV",
            csv,
            "earthquake_forecast.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(f"Forecast Error: {e}")

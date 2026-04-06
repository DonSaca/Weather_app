import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ----------------------------------------
# CONFIG
# ----------------------------------------
LATITUDE = 38.72
LONGITUDE = -9.14

FEATURE_SCALER_PATH = "scaler_features.joblib"
TARGET_SCALER_PATH = "scaler_target.joblib"

LSTM_MODEL_PATH = "weather_predictor_LSTM.keras"
TRANSFORMER_MODEL_PATH = "weather_predictor_transformer.keras"

MAE_LSTM = 1.99
MAE_TRANSFORMER = 2.30  # adjust if needed


# ----------------------------------------
# DATA FETCHING
# ----------------------------------------
def fetch_data():
    """Fetch last 10+ days of hourly weather data from Open-Meteo API"""
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=15)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,precipitation,windspeed_10m",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "timezone": "Europe/Lisbon"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame({
            "date": pd.to_datetime(data["hourly"]["time"]),
            "temperature": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "pressure": data["hourly"]["pressure_msl"],
            "precipitation": data["hourly"]["precipitation"],
            "wind_speed": data["hourly"]["windspeed_10m"]
        })

        return df

    except Exception as e:
        st.error(f"Erro ao obter dados da API: {e}")
        return None


# ----------------------------------------
# PREPROCESSING
# ----------------------------------------
def preprocess_data(df):
    """Convert hourly data to daily + feature engineering"""

    df["date_only"] = df["date"].dt.date

    df_daily = df.groupby("date_only").agg({
        "temperature": ["max", "min", "mean"],
        "humidity": "mean",
        "pressure": "mean",
        "precipitation": "sum",
        "wind_speed": "mean"
    })

    df_daily.columns = [
        "temp_max", "temp_min", "temp_mean",
        "humidity", "pressure", "precipitation", "wind_speed"
    ]

    df_daily = df_daily.reset_index()

    # Feature Engineering
    df_daily["temp_range"] = df_daily["temp_max"] - df_daily["temp_min"]
    df_daily["temp_max_mean_3d"] = df_daily["temp_max"].rolling(window=3).mean()

    # Lag Features
    for lag in range(1, 8):
        df_daily[f"temp_max_lag_{lag}"] = df_daily["temp_max"].shift(lag)

    df_daily = df_daily.dropna()

    # Select last 7 days
    df_model = df_daily.tail(7)

    feature_cols = [
        "temp_max", "temp_min", "temp_mean",
        "humidity", "pressure", "precipitation", "wind_speed",
        "temp_range", "temp_max_mean_3d",
        "temp_max_lag_1", "temp_max_lag_2", "temp_max_lag_3",
        "temp_max_lag_4", "temp_max_lag_5"
    ]

    X = df_model[feature_cols].values

    return X, df_daily


# ----------------------------------------
# LOAD MODEL
# ----------------------------------------
def load_model(model_name):
    if model_name == "LSTM":
        return tf.keras.models.load_model(LSTM_MODEL_PATH)
    else:
        return tf.keras.models.load_model(TRANSFORMER_MODEL_PATH)


# ----------------------------------------
# PREDICTION
# ----------------------------------------
def predict(model, X, scaler_X, scaler_y):
    """Scale, predict, inverse scale"""

    X_scaled = scaler_X.transform(X)
    X_scaled = np.reshape(X_scaled, (1, 7, 14))

    pred_scaled = model.predict(X_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)

    return float(pred[0][0])


# ----------------------------------------
# PLOT
# ----------------------------------------
def plot_results(df_daily, prediction):
    """Plot last 30 days + prediction"""

    df_plot = df_daily.tail(30)

    dates = list(df_plot["date_only"])
    temps = list(df_plot["temp_max"])

    # Add prediction
    next_day = dates[-1] + timedelta(days=1)
    dates.append(next_day)
    temps.append(prediction)

    plt.figure()
    plt.plot(dates[:-1], temps[:-1], label="Histórico")
    plt.plot(dates[-2:], temps[-2:], linestyle="--", label="Previsão")

    plt.xlabel("Data")
    plt.ylabel("Temperatura Máx (°C)")
    plt.title("Previsão de Temperatura")
    plt.legend()

    st.pyplot(plt)


# ----------------------------------------
# UI
# ----------------------------------------
def main():
    st.title("🌤️ Weather Prediction App")
    st.write("Previsão da temperatura máxima para Lisboa (Próximo Dia)")

    # Sidebar
    st.sidebar.header("Modelo")
    model_choice = st.sidebar.radio(
        "Seleciona o modelo:",
        ["LSTM (Mais Estável)", "Transformer (Experimental)"]
    )

    st.sidebar.write(f"LSTM MAE: {MAE_LSTM}°C")
    st.sidebar.write(f"Transformer MAE: {MAE_TRANSFORMER}°C")

    # Load scalers
    scaler_X = joblib.load(FEATURE_SCALER_PATH)
    scaler_y = joblib.load(TARGET_SCALER_PATH)

    # Fetch data
    df = fetch_data()
    if df is None:
        return

    # Preprocess
    X, df_daily = preprocess_data(df)

    # Load both models for confidence comparison
    lstm_model = load_model("LSTM")
    transformer_model = load_model("Transformer")

    pred_lstm = predict(lstm_model, X, scaler_X, scaler_y)
    pred_transformer = predict(transformer_model, X, scaler_X, scaler_y)

    # Selected model
    if "LSTM" in model_choice:
        prediction = pred_lstm
    else:
        prediction = pred_transformer

    # Display prediction
    st.subheader("📊 Resultado")
    st.metric("Temperatura Máxima Prevista", f"{prediction:.2f} °C")

    # Confidence indicator
    diff = abs(pred_lstm - pred_transformer)
    if diff < 0.5:
        confidence = "Alta"
    else:
        confidence = "Moderada"

    st.write(f"🔍 Confiança: {confidence}")
    st.write("Margem de erro histórica: ±1.99°C")

    # Plot
    plot_results(df_daily, prediction)


# ----------------------------------------
# RUN
# ----------------------------------------
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import openmeteo_requests
from retry_requests import retry
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- CONFIGURATION & CONSTANTS ---
STATION_LAT = 38.7167
STATION_LON = -9.1333
WINDOW_SIZE = 7
NUM_FEATURES = 14
MODEL_PATH_LSTM = 'models/weather_predictor_LSTM.keras'
MODEL_PATH_TRANS = 'models/weather_predictor_transformer.keras'
SCALER_FEAT_PATH = 'models/scaler_features.joblib'
SCALER_TARG_PATH = 'models/scaler_target.joblib'

# Feature order must strictly match training
FEATURES = [
    'temp_max', 'temp_min', 'temp_mean', 'humidity_mean', 
    'pressure_mean', 'precip_total', 'wind_max', 
    'temp_max_lag_1', 'temp_max_lag_2', 'temp_max_lag_3',
    'temp_max_mean_3d', 'temp_range', 'sin_month', 'cos_month'
]

st.set_page_config(page_title="Lisbon AI Weather Predictor", layout="wide")

# --- CORE FUNCTIONS ---

@st.cache_resource
def load_ml_assets():
    """Load pretrained models and scalers into memory."""
    try:
        lstm = tf.keras.models.load_model(MODEL_PATH_LSTM)
        transformer = tf.keras.models.load_model(MODEL_PATH_TRANS)
        s_feat = joblib.load(SCALER_FEAT_PATH)
        s_targ = joblib.load(SCALER_TARG_PATH)
        return lstm, transformer, s_feat, s_targ
    except Exception as e:
        st.error(f"Erro ao carregar ficheiros ML: {e}")
        return None, None, None, None

def fetch_weather_data(days_back=45):
    """Fetch hourly data from Open-Meteo and resample to daily."""
    try:
        retry_session = retry(backoff_factor=0.2, retries=5)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            "latitude": STATION_LAT,
            "longitude": STATION_LON,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "precipitation", "wind_speed_10m"],
            "timezone": "UTC"
        }
        
        responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                periods=len(hourly.Variables(0).ValuesAsNumpy()),
                freq='H'
            ),
            "temp": hourly.Variables(0).ValuesAsNumpy(),
            "hum": hourly.Variables(1).ValuesAsNumpy(),
            "pres": hourly.Variables(2).ValuesAsNumpy(),
            "prec": hourly.Variables(3).ValuesAsNumpy(),
            "wind": hourly.Variables(4).ValuesAsNumpy()
        }
        
        df_hourly = pd.DataFrame(data)
        
        # Resample to Daily
        df_daily = df_hourly.resample('D', on='date').agg({
            'temp': ['max', 'min', 'mean'],
            'hum': 'mean',
            'pres': 'mean',
            'prec': 'sum',
            'wind': 'max'
        })
        
        df_daily.columns = ['temp_max', 'temp_min', 'temp_mean', 'humidity_mean', 'pressure_mean', 'precip_total', 'wind_max']
        return df_daily.reset_index()
    except Exception as e:
        st.error(f"Erro na API Open-Meteo: {e}")
        return None

def preprocess_pipeline(df):
    """Apply feature engineering and scaling."""
    df = df.copy()
    
    # 1. Lags
    df['temp_max_lag_1'] = df['temp_max'].shift(1)
    df['temp_max_lag_2'] = df['temp_max'].shift(2)
    df['temp_max_lag_3'] = df['temp_max'].shift(3)
    
    # 2. Rolling Mean & Range
    df['temp_max_mean_3d'] = df['temp_max'].rolling(window=3).mean()
    df['temp_range'] = df['temp_max'] - df['temp_min']
    
    # 3. Cyclical Encoding
    df['month'] = df['date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Drop NaNs created by shifts/rolling
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

def prepare_input_tensor(df, scaler):
    """Extract last 7 days and reshape for LSTM/Transformer (1, 7, 14)."""
    last_7_days = df[FEATURES].tail(WINDOW_SIZE)
    scaled_data = scaler.transform(last_7_days)
    return scaled_data.reshape(1, WINDOW_SIZE, NUM_FEATURES)

def get_prediction(model, tensor, scaler_target):
    """Run inference and inverse scale result."""
    pred_scaled = model.predict(tensor, verbose=0)
    return scaler_target.inverse_transform(pred_scaled)[0][0]

def plot_results(df_historic, prediction_val):
    """Visualize last 30 days + prediction."""
    plot_df = df_historic.tail(30).copy()
    last_date = plot_df['date'].iloc[-1]
    next_date = last_date + timedelta(days=1)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df['date'], plot_df['temp_max'], label='Histórico (30 dias)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    
    # Connect last point to prediction
    ax.plot([last_date, next_date], [plot_df['temp_max'].iloc[-1], prediction_val], color='#d62728', linestyle='--')
    ax.scatter(next_date, prediction_val, color='#d62728', s=100, label='Previsão Amanhã', zorder=5)
    
    ax.set_title("Evolução da Temperatura Máxima em Lisboa", fontsize=12)
    ax.set_ylabel("Temperatura (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    return fig

# --- UI LAYOUT ---

st.title("🌤️ Previsão de Temperatura com Deep Learning")
st.markdown("---")

# Load Assets
m_lstm, m_trans, s_feat, s_targ = load_ml_assets()

# Sidebar
st.sidebar.header("🕹️ Painel de Controlo")
model_choice = st.sidebar.radio(
    "Selecione o Modelo:",
    ["LSTM (Mais Estável)", "Transformer (Experimental)"]
)

st.sidebar.markdown("### 📊 Performance Histórica")
st.sidebar.write("**LSTM MAE:** 1.99°C")
st.sidebar.write("**Transformer MAE:** 2.25°C")

# Main Execution
if st.button("🚀 Gerar Previsão para Amanhã"):
    if m_lstm and m_trans:
        with st.spinner("A processar dados em tempo real..."):
            # 1. Fetch
            raw_data = fetch_weather_data()
            if raw_data is not None:
                # 2. Preprocess
                processed_data = preprocess_pipeline(raw_data)
                
                # 3. Prepare Tensor
                input_tensor = prepare_input_tensor(processed_data, s_feat)
                
                # 4. Predict with both (for confidence check)
                res_lstm = get_prediction(m_lstm, input_tensor, s_targ)
                res_trans = get_prediction(m_trans, input_tensor, s_targ)
                
                # Selection
                final_pred = res_lstm if "LSTM" in model_choice else res_trans
                
                # 5. Display Results
                st.subheader(f"Resultado: {model_choice.split(' ')[0]}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Previsão Amanhã", f"{final_pred:.1f} °C")
                c2.metric("Margem de Erro", "±1.99°C")
                
                # Confidence Logic
                diff = abs(res_lstm - res_trans)
                conf_level = "Alta" if diff < 0.5 else "Moderada"
                c3.metric("Confiança", conf_level, delta=f"Diff: {diff:.2f}°C", delta_color="inverse")
                
                # 6. Plot
                st.markdown("### 📈 Visualização de Tendência")
                fig = plot_results(processed_data, final_pred)
                st.pyplot(fig)
                
                st.success("Cálculo concluído com sucesso!")
    else:
        st.error("Não foi possível inicializar os modelos. Verifique a pasta 'models/'.")

st.markdown("---")
st.caption("Projeto de Aptidão Profissional (PAP) - 12º Ano | Dados: Open-Meteo ERA5")
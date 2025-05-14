
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
import keras
import tensorflow
import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os

# API Key
API_KEY = "Key"
LOCATION = "Illinois"
base_url = "http://api.weatherapi.com/v1/history.json"


now = datetime.now()
dates_needed = list(set([
    now.strftime("%Y-%m-%d"),
    (now - timedelta(hours=10)).strftime("%Y-%m-%d")
]))

weather_data = []

for date_str in dates_needed:
    url = f"{base_url}?key={API_KEY}&q={LOCATION}&dt={date_str}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"error: {e}")
        continue

    if "forecast" in data:
        try:
            hour_data_list = data["forecast"]["forecastday"][0]["hour"]
            for hour_data in hour_data_list:
                time_str = hour_data["time"]
                hour_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")

                if 0 <= (now - hour_dt).total_seconds() <= 3600 * 10:
                    weather_data.append({
                        "Temperature_C": hour_data["temp_c"],
                        "Humidity_%": hour_data["humidity"],
                        "Is_Snowing": 1 if "snow" in hour_data["condition"]["text"].lower() else 0,
                        "Pressure_mb": hour_data["pressure_mb"],
                        "Wind_Speed_kph": hour_data["wind_kph"],
                        "Visibility_km": hour_data["vis_km"],
                        "Wind_Bearing_deg": hour_data["wind_degree"]
                    })
        except KeyError:
            print(f"error: {data}")
            continue

if not weather_data:
    print("cannot find weather")
    exit()


df = pd.DataFrame(weather_data).to_numpy()


scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Reshape the data to match the LSTM model input shape
df_reshaped = df_scaled.reshape((df_scaled.shape[0], 1, df_scaled.shape[1]))  # Shape (10, 1, 7)

try:
    mc = joblib.load("weather_lstm.pkl")
except FileNotFoundError:
    print("model not found")
    exit()

dapre = mc.predict(df_reshaped)


def inverse_scale_temp(scaled_temp, scaler, feature_index=0):
    dummy = np.zeros((scaled_temp.shape[0], df.shape[1]))
    dummy[:, feature_index] = scaled_temp.flatten()
    return scaler.inverse_transform(dummy)[:, feature_index].reshape(-1, 1)

result = inverse_scale_temp(dapre, scaler, feature_index=0)
print(result)

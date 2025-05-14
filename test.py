
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

# 安全存储 API Key
API_KEY = "a9e680c901b64650acd211526250304"
LOCATION = "Illinois"
base_url = "http://api.weatherapi.com/v1/history.json"

# 当前时间
now = datetime.now()
dates_needed = list(set([
    now.strftime("%Y-%m-%d"),
    (now - timedelta(hours=10)).strftime("%Y-%m-%d")
]))

weather_data = []

# 获取天气数据
for date_str in dates_needed:
    url = f"{base_url}?key={API_KEY}&q={LOCATION}&dt={date_str}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"请求失败: {e}")
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
            print(f"数据结构异常: {data}")
            continue

if not weather_data:
    print("未获取到天气数据")
    exit()

# 转换为 NumPy 数组
df = pd.DataFrame(weather_data).to_numpy()

# 归一化
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Reshape the data to match the LSTM model input shape
df_reshaped = df_scaled.reshape((df_scaled.shape[0], 1, df_scaled.shape[1]))  # Shape (10, 1, 7)

# 加载模型
try:
    mc = joblib.load("weather_lstm.pkl")
except FileNotFoundError:
    print("模型文件未找到")
    exit()

# 预测
dapre = mc.predict(df_reshaped)

# 反归一化温度
def inverse_scale_temp(scaled_temp, scaler, feature_index=0):
    dummy = np.zeros((scaled_temp.shape[0], df.shape[1]))
    dummy[:, feature_index] = scaled_temp.flatten()
    return scaler.inverse_transform(dummy)[:, feature_index].reshape(-1, 1)

result = inverse_scale_temp(dapre, scaler, feature_index=0)
print(result)

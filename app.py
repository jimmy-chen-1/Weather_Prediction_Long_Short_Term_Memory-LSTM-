from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
import json

# 安全存储 API Key
API_KEY = "a9e680c901b64650acd211526250304"
LOCATION = "Illinois"
base_url = "http://api.weatherapi.com/v1/history.json"

# 创建 Flask 应用
app = Flask(__name__)

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

# 转换为 Pandas DataFrame
df = pd.DataFrame(weather_data)

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


# 反归一化温度
def inverse_scale_temp(scaled_temp, scaler, feature_index=0):
    dummy = np.zeros((scaled_temp.shape[0], df.shape[1]))
    dummy[:, feature_index] = scaled_temp.flatten()
    return scaler.inverse_transform(dummy)[:, feature_index].reshape(-1, 1)


# 预测天气
def predict_weather(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame")

    # 归一化数据
    df_scaled = scaler.fit_transform(df)

    # Reshape data to match LSTM input
    df_reshaped = df_scaled.reshape((df_scaled.shape[0], 1, df_scaled.shape[1]))

    # 使用 LSTM 模型进行预测
    dapre = mc.predict(df_reshaped)

    # 反归一化温度
    result = inverse_scale_temp(dapre, scaler, feature_index=0)
    return result


# 创建网页路由
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/weather', methods=['POST'])
def weather():
    try:
        # 解析前端发送的JSON数据
        data = request.get_json()
        location = data['location']

        # 更新全局变量并清空旧数据
        global LOCATION, weather_data
        LOCATION = location
        weather_data.clear()

        # 获取当前时间并计算所需日期
        now = datetime.now()
        dates_needed = list(set([
            now.strftime("%Y-%m-%d"),
            (now - timedelta(hours=10)).strftime("%Y-%m-%d")
        ]))

        # 请求天气API数据
        for date_str in dates_needed:
            url = f"{base_url}?key={API_KEY}&q={LOCATION}&dt={date_str}"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                api_data = response.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                print(f"API请求失败: {e}")
                continue

            # 提取有效小时数据
            if "forecast" in api_data:
                try:
                    forecast_day = api_data["forecast"]["forecastday"][0]
                    for hour_data in forecast_day["hour"]:
                        time_str = hour_data["time"]
                        hour_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")

                        # 仅保留最近10小时数据
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
                except KeyError as e:
                    print(f"数据解析失败，缺失字段: {e}")
                    continue

        # 检查数据有效性
        if not weather_data:
            return jsonify({"error": "未获取到有效天气数据，请检查API密钥或城市名称"}), 400

        # 转换为DataFrame并进行预测
        df = pd.DataFrame(weather_data)
        try:
            predicted = predict_weather(df).flatten().tolist()
        except Exception as e:
            print(f"模型预测失败: {e}")
            return jsonify({"error": "温度预测失败"}), 500

        # 生成智能时间标签（处理跨天）
        time_labels = []
        for i in range(10):
            start_time = now + timedelta(hours=i)
            end_time = start_time + timedelta(hours=1)

            # 处理跨天显示逻辑
            day_suffix = "(次日)" if start_time.day != end_time.day else ""
            label = f"{start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')} {day_suffix}".strip()
            time_labels.append(label)

        # 返回结构化数据
        return jsonify({
            "predicted_temperatures": [round(temp, 1) for temp in predicted],
            "time_labels": time_labels
        })

    except KeyError as e:
        return jsonify({"error": f"请求参数错误: {str(e)}"}), 400
    except Exception as e:
        print(f"服务器内部错误: {str(e)}")
        return jsonify({"error": "服务器内部错误"}), 500


if __name__ == '__main__':
    app.run(debug=True)

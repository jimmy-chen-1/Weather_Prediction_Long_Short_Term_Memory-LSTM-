from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
import json

# API Key
API_KEY = "ENTER_YOUR_KEY"
LOCATION = "Illinois"
base_url = "http://api.weatherapi.com/v1/history.json"

# Flask 
app = Flask(__name__)

# time
now = datetime.now()
dates_needed = list(set([
    now.strftime("%Y-%m-%d"),
    (now - timedelta(hours=10)).strftime("%Y-%m-%d")
]))

weather_data = []

# get weather
for date_str in dates_needed:
    url = f"{base_url}?key={API_KEY}&q={LOCATION}&dt={date_str}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"fail: {e}")
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
            print(f"data abnormal: {data}")
            continue

if not weather_data:
    print("weather data not found")
    exit()

#  Pandas DataFrame
df = pd.DataFrame(weather_data)

# normalize
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Reshape the data to match the LSTM model input shape
df_reshaped = df_scaled.reshape((df_scaled.shape[0], 1, df_scaled.shape[1]))  # Shape (10, 1, 7)

# add model
try:
    mc = joblib.load("weather_lstm.pkl")
except FileNotFoundError:
    print("model not find")
    exit()


# anti-normalize
def inverse_scale_temp(scaled_temp, scaler, feature_index=0):
    dummy = np.zeros((scaled_temp.shape[0], df.shape[1]))
    dummy[:, feature_index] = scaled_temp.flatten()
    return scaler.inverse_transform(dummy)[:, feature_index].reshape(-1, 1)


# forecast weather
def predict_weather(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame")

    # normalize
    df_scaled = scaler.fit_transform(df)

    # Reshape data to match LSTM input
    df_reshaped = df_scaled.reshape((df_scaled.shape[0], 1, df_scaled.shape[1]))

    # use LSTM 
    dapre = mc.predict(df_reshaped)

    result = inverse_scale_temp(dapre, scaler, feature_index=0)
    return result


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/weather', methods=['POST'])
def weather():
    try:

        data = request.get_json()
        location = data['location']

        global LOCATION, weather_data
        LOCATION = location
        weather_data.clear()

        now = datetime.now()
        dates_needed = list(set([
            now.strftime("%Y-%m-%d"),
            (now - timedelta(hours=10)).strftime("%Y-%m-%d")
        ]))

        # API data
        for date_str in dates_needed:
            url = f"{base_url}?key={API_KEY}&q={LOCATION}&dt={date_str}"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                api_data = response.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                print(f"API failed: {e}")
                continue

            if "forecast" in api_data:
                try:
                    forecast_day = api_data["forecast"]["forecastday"][0]
                    for hour_data in forecast_day["hour"]:
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
                except KeyError as e:
                    print(f"data lose: {e}")
                    continue

      
        if not weather_data:
            return jsonify({"error": "cannot find weather information"}), 400

       
        df = pd.DataFrame(weather_data)
        try:
            predicted = predict_weather(df).flatten().tolist()
        except Exception as e:
            print(f"model failed: {e}")
            return jsonify({"error": "weather fail to find"}), 500

      
        time_labels = []
        for i in range(10):
            start_time = now + timedelta(hours=i)
            end_time = start_time + timedelta(hours=1)

           
            day_suffix = "(next day)" if start_time.day != end_time.day else ""
            label = f"{start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')} {day_suffix}".strip()
            time_labels.append(label)

     
        return jsonify({
            "predicted_temperatures": [round(temp, 1) for temp in predicted],
            "time_labels": time_labels
        })

    except KeyError as e:
        return jsonify({"error": f"error: {str(e)}"}), 400
    except Exception as e:
        print(f"internal error: {str(e)}")
        return jsonify({"error": "internal error"}), 500


if __name__ == '__main__':
    app.run(debug=True)

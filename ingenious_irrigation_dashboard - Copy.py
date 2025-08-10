import os
import json
import glob
import requests
from flask import Flask, render_template, request
from datetime import datetime
from ultralytics import YOLO  # for AI predictions
from PIL import Image
import matplotlib.pyplot as plt
 
from schedule_manager import save_schedule, load_schedule  # already imported

@app.route('/save_schedule', methods=['POST'])
def save_schedule_route():
    start_time = request.form.get('start_time')
    duration = request.form.get('duration')
    frequency_days = request.form.get('frequency_days')

    schedule = {
        "start_time": start_time,
        "duration": int(duration),
        "frequency_days": int(frequency_days)
    }

    save_schedule(schedule)
    return dashboard()

# --- App Setup ---
app = Flask(__name__)

# --- Folder Paths ---import requests
from schedule_manager import load_schedule, should_water_today
from datetime import datetime
import os

# --- Weather Integration ---
def get_weather_forecast():
    api_key = "YOUR_OPENWEATHER_API_KEY"
    lat, lon = "29.7604", "-95.3698"  # Houston, TX
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()
    return None

def should_skip_watering(weather_data):
    if not weather_data:
        return False
    next_12h = weather_data['list'][:4]
    for period in next_12h:
        if period['rain'].get('3h', 0) > 0.1:
            return True
    return False

def smart_irrigation_decision():
    schedule = load_schedule()
    logs = load_scores()
    last_watered = logs[-1]["timestamp"] if logs else "2000-01-01T00:00:00"
    
    weather_data = get_weather_forecast()
    skip_due_to_rain = should_skip_watering(weather_data)
    
    if should_water_today(last_watered, schedule["frequency_days"]) and not skip_due_to_rain:
        print("💧 Irrigation allowed today!")
        return True
    else:
        print("⛔ Skipping irrigation (either not scheduled or rain expected).")
        return False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, 'logs')
IMAGE_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'predict')
CHART_PATH = os.path.join(BASE_DIR, 'static', 'charts')

# Ensure folders exist
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(IMAGE_PATH, exist_ok=True)
os.makedirs(CHART_PATH, exist_ok=True)

# --- Load Scores ---
def load_scores():
    log_file = os.path.join(LOG_PATH, 'hydration_log.json')
    if not os.path.exists(log_file):
        return []
    with open(log_file, 'r') as f:
        return json.load(f)

# --- Save Score ---
def save_score(score):
    log_file = os.path.join(LOG_PATH, 'hydration_log.json')
    logs = load_scores()
    logs.append({"timestamp": datetime.now().isoformat(), "score": score})
    with open(log_file, 'w') as f:
        json.dump(logs[-100:], f, indent=2)

# --- Generate Chart ---
def generate_chart():
    logs = load_scores()
    if not logs:
        return None
    times = [entry['timestamp'].split('T')[0] for entry in logs]
    scores = [entry['score'] for entry in logs]

    plt.figure(figsize=(10, 4))
    plt.plot(times, scores, marker='o', linestyle='-', color='#00ff88')
    plt.title('Hydration Score Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_file = os.path.join(CHART_PATH, 'hydration_chart.png')
    plt.savefig(chart_file)
    plt.close()
    return chart_file

# --- Routes ---
@app.route('/')
def dashboard():
    generate_chart()
    predictions = sorted(glob.glob(os.path.join(IMAGE_PATH, '*.jpg')), key=os.path.getmtime, reverse=True)
    hydration_log = load_scores()[::-1]  # newest first
    return render_template('dashboard.html',
                           predictions=predictions[:5],
                           chart_file='charts/hydration_chart.png',
                           hydration_log=hydration_log[:10])

@app.route('/submit_score', methods=['POST'])
def submit_score():
    score = request.form.get('score')
    try:
        score = float(score)
        save_score(score)
    except ValueError:
        pass
    return dashboard()

# --- Launch ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

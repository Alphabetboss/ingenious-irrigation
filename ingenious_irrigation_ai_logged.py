
from datetime import datetime
import random
import numpy as np
import cv2
import csv
import os

# ---------------- Configuration ----------------
BASE_WATERING_TIME = 10  # minutes
ZONE_COUNT = 3
CSV_LOG_FILE = "irrigation_log.csv"

# ---------------- Simulated Sensors ----------------
def simulate_image():
    base_color = random.randint(50, 200)
    return np.full((480, 640, 3), (base_color, base_color + 30, base_color), dtype=np.uint8)

def simulate_humidity():
    return random.uniform(30, 90)

def simulate_pressure():
    return random.uniform(0.1, 1.0)

def simulate_weather_forecast():
    return random.choice(["Clear", "Rain", "Clouds", "Thunderstorm"])

# ---------------- Hydration Score Logic ----------------
def calculate_greenness_score(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1] * 255)
    hydration_score = round((1.0 - green_ratio) * 10, 2)
    return min(max(hydration_score, 0), 10)

def adjust_watering_time(base_time, hydration_score):
    if hydration_score == 5:
        return base_time
    elif hydration_score < 5:
        return base_time + int((5 - hydration_score) * 2)
    else:
        return max(0, base_time - int((hydration_score - 5) * 2))

# ---------------- Emergency Detection ----------------
def detect_emergency(pressure, image):
    if pressure < 0.2:
        return "Possible pipe burst (low pressure)"
    muddy_score = calculate_greenness_score(image)
    if muddy_score >= 9.5:
        return "Possible flood or overwatering detected"
    return None

# ---------------- CSV Logger ----------------
def log_to_csv(data):
    file_exists = os.path.isfile(CSV_LOG_FILE)
    with open(CSV_LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Zone", "Hydration Score", "Humidity", "Pressure", "Watering Time", "Emergency"])
        writer.writerow(data)

# ---------------- Irrigation AI Core ----------------
def run_irrigation_ai():
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Starting AI Irrigation System")
    weather = simulate_weather_forecast()
    print("Weather forecast:", weather)

    for zone in range(ZONE_COUNT):
        print(f"\n🌱 Zone {zone + 1}")

        image = simulate_image()
        hydration_score = calculate_greenness_score(image)
        humidity = simulate_humidity()
        pressure = simulate_pressure()

        print(f"Hydration Score: {hydration_score}/10")
        print(f"Humidity: {round(humidity)}%")
        print(f"Pressure: {round(pressure, 2)} bar")

        emergency = detect_emergency(pressure, image)
        if emergency:
            print(f"🚨 EMERGENCY: {emergency}")
            log_to_csv([timestamp, zone + 1, hydration_score, round(humidity), round(pressure, 2), 0, emergency])
            continue

        watering_time = adjust_watering_time(BASE_WATERING_TIME, hydration_score)
        action = f"Watering for {watering_time} minutes" if watering_time > 0 else "Skipping watering (too wet)"
        print(f"💧 Action: {action}")

        log_to_csv([timestamp, zone + 1, hydration_score, round(humidity), round(pressure, 2), watering_time, "None"])

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    run_irrigation_ai()

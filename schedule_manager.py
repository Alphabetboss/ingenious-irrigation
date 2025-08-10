import os
import json

SCHEDULE_PATH = os.path.join(os.path.dirname(__file__), 'logs', 'watering_schedule.json')

def save_schedule(data):
    with open(SCHEDULE_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def load_schedule():
    if not os.path.exists(SCHEDULE_PATH):
        return {
            "start_time": "05:00",
            "duration": 10,
            "frequency_days": 2
        }
    with open(SCHEDULE_PATH, 'r') as f:
        return json.load(f)

def should_water_today(last_timestamp, frequency_days):
    from datetime import datetime, timedelta
    last_date = datetime.fromisoformat(last_timestamp)
    return datetime.now() >= (last_date + timedelta(days=frequency_days))

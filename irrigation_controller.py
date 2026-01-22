import time
from hydration_logic import calculate_hydration_score

def adjust_watering(base_minutes, hydration_score):
    if hydration_score <= 2:
        return base_minutes + 10
    elif hydration_score <= 4:
        return base_minutes + 5
    elif hydration_score <= 6:
        return base_minutes
    elif hydration_score <= 8:
        return base_minutes - 5
    else:
        return 0

def run_zone(duration_minutes):
    print(f"💧 Watering for {duration_minutes} minutes")
    time.sleep(duration_minutes * 60)
    print("✅ Watering complete")
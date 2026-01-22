# config.py

# -------------------
# System Settings
# -------------------
SYSTEM_NAME = "Ingenious Irrigation"
DEBUG = True

# -------------------
# Camera Settings
# -------------------
CAMERA_INDEX = 0
IMAGE_SAVE_PATH = "data/latest_lawn.jpg"

# -------------------
# Hydration Scoring
# 0 = very dry | 5 = optimal | 10 = oversaturated
# -------------------
HYDRATION_OPTIMAL = 5
HYDRATION_MIN = 0
HYDRATION_MAX = 10

# -------------------
# Watering Limits (minutes)
# -------------------
MIN_WATER_TIME = 3
MAX_WATER_TIME = 30
DEFAULT_WATER_TIME = 10

# -------------------
# Scheduling
# -------------------
DEFAULT_WATER_HOUR = 5   # 5 AM
DEFAULT_WATER_MINUTE = 0
WATER_DAYS_INTERVAL = 2  # every other day

# -------------------
# GPIO (Raspberry Pi)
# -------------------
RELAY_PIN = 17  # change per wiring
GPIO_ACTIVE_STATE = True

# -------------------
# Logging
# -------------------
LOG_FILE = "irrigation_log.csv"
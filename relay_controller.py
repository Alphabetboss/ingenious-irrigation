# relay_controller.py

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

from config import RELAY_PIN, GPIO_ACTIVE_STATE
import time

def setup_gpio():
    if not GPIO_AVAILABLE:
        print("GPIO not available (running in dev mode)")
        return

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_PIN, GPIO.OUT)
    GPIO.output(RELAY_PIN, not GPIO_ACTIVE_STATE)


def start_watering(duration_minutes: int):
    print(f"Starting watering for {duration_minutes} minutes")

    if GPIO_AVAILABLE:
        GPIO.output(RELAY_PIN, GPIO_ACTIVE_STATE)

    time.sleep(duration_minutes * 60)

    stop_watering()


def stop_watering():
    print("Stopping watering")

    if GPIO_AVAILABLE:
        GPIO.output(RELAY_PIN, not GPIO_ACTIVE_STATE)
        GPIO.cleanup()
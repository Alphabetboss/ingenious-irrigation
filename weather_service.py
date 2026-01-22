import requests
from config import WEATHER_API_KEY, LOCATION

def get_weather():
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={LOCATION}&appid={WEATHER_API_KEY}&units=imperial"
    )
    return requests.get(url).json()

def should_skip_weather(weather):
    rain = weather.get("rain", {}).get("1h", 0)
    temp = weather["main"]["temp"]

    if rain > 0.2:
        return True
    if temp < 40:
        return True
    return False
"""Location and weather helpers for real-time weather responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass
class Location:
    city: str
    region: str
    country: str
    latitude: float
    longitude: float
    timezone_name: str


WEATHER_CODE_MAP: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _load_json(url: str, timeout: float = 8.0) -> dict:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_ip_location() -> Location | None:
    """Resolve approximate user location from public IP."""
    try:
        data = _load_json("https://ipapi.co/json/")
        return Location(
            city=str(data.get("city") or "Unknown City"),
            region=str(data.get("region") or "Unknown Region"),
            country=str(data.get("country_name") or "Unknown Country"),
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            timezone_name=str(data.get("timezone") or "UTC"),
        )
    except (KeyError, ValueError, URLError, TimeoutError, OSError):
        return None


def get_current_weather(location: Location) -> dict | None:
    """Fetch current weather from Open-Meteo for a resolved location."""
    params = urlencode(
        {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        data = _load_json(url)
        current = data.get("current") or {}
        code = int(current.get("weather_code", -1))
        return {
            "temperature_f": current.get("temperature_2m"),
            "feels_like_f": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_mph": current.get("wind_speed_10m"),
            "weather": WEATHER_CODE_MAP.get(code, f"code {code}"),
            "observed_at": current.get("time"),
            "timezone": data.get("timezone") or location.timezone_name,
        }
    except (ValueError, URLError, TimeoutError, OSError):
        return None


def is_weather_query(text: str) -> bool:
    """Heuristic detector for weather/temperature questions."""
    lower = text.lower()
    keywords = (
        "weather",
        "temperature",
        "forecast",
        "rain",
        "snow",
        "wind",
        "humid",
        "hot",
        "cold",
        "outside",
    )
    return any(keyword in lower for keyword in keywords)


def build_weather_context(location: Location, weather: dict) -> str:
    """Build plain-text context for the LLM."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "Use this live weather data as source of truth for your answer. "
        "Do not invent weather values.\n"
        f"Current UTC time: {now_utc}\n"
        f"User approximate location (IP-based): {location.city}, {location.region}, {location.country}\n"
        f"Observed local time: {weather.get('observed_at')} ({weather.get('timezone')})\n"
        f"Condition: {weather.get('weather')}\n"
        f"Temperature: {weather.get('temperature_f')} F\n"
        f"Feels like: {weather.get('feels_like_f')} F\n"
        f"Humidity: {weather.get('humidity')}%\n"
        f"Wind: {weather.get('wind_mph')} mph"
    )

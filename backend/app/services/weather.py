"""Weather via OpenWeatherMap current-weather API. Key comes from .env only."""
from __future__ import annotations
from datetime import datetime, timezone
import requests
from ..core.config import get_settings
from ..core.errors import WeatherError

CURRENT = "https://api.openweathermap.org/data/2.5/weather"
GEO = "https://api.openweathermap.org/geo/1.0/direct"


def _get(url: str, params: dict) -> requests.Response:
    s = get_settings()
    if not s.weather_api_key:
        raise WeatherError("Weather service is not configured (WEATHER_API_KEY missing). "
                           "You can still continue without weather.", "no_api_key", 503)
    params = {**params, "appid": s.weather_api_key}
    try:
        r = requests.get(url, params=params, timeout=s.weather_timeout_s)
    except requests.Timeout:
        raise WeatherError("The weather service timed out. Please try again.", "timeout", 504)
    except requests.RequestException:
        raise WeatherError("Could not reach the weather service. Check your internet connection.", "network", 502)
    if r.status_code == 401:
        raise WeatherError("The weather API key is invalid.", "invalid_api_key", 503)
    if r.status_code == 404:
        raise WeatherError("Location not found. Please check the location and try again.", "location_not_found", 404)
    if r.status_code == 400:
        raise WeatherError("Invalid location coordinates.", "bad_request", 400)
    if r.status_code >= 400:
        raise WeatherError("The weather service returned an error. Please try again later.", "api_failure", 502)
    return r


def geocode(query: str) -> tuple[float, float, str]:
    r = _get(GEO, {"q": query, "limit": 1})
    try:
        hits = r.json()
    except ValueError:
        raise WeatherError("Unexpected response from the weather service.", "bad_response", 502)
    if not hits:
        raise WeatherError("Location not found. Please check the spelling or try 'City,Country'.",
                           "location_not_found", 404)
    h = hits[0]
    name = ", ".join(x for x in (h.get("name"), h.get("state"), h.get("country")) if x)
    return float(h["lat"]), float(h["lon"]), name


def parse_current(j: dict) -> dict:
    """Convert an OpenWeatherMap payload into our fields. Missing fields stay None."""
    main, wind = j.get("main") or {}, j.get("wind") or {}
    rain = j.get("rain") or {}
    rainfall = rain.get("1h", rain.get("3h"))
    # OWM omits 'rain' when it is not raining: that is a real 0.0 only if the payload is otherwise valid.
    if rainfall is None and main:
        rainfall = 0.0
    desc = (j.get("weather") or [{}])[0].get("description")
    ts = j.get("dt")
    coord = j.get("coord") or {}
    return {
        "temperature_c": main.get("temp"), "humidity_pct": main.get("humidity"),
        "rainfall_mm": rainfall, "wind_speed_ms": wind.get("speed"), "description": desc,
        "location_name": ", ".join(x for x in (j.get("name"), (j.get("sys") or {}).get("country")) if x) or None,
        "latitude": coord.get("lat"), "longitude": coord.get("lon"),
        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
    }


def get_weather(lat: float | None = None, lon: float | None = None, query: str | None = None) -> dict:
    name = None
    if lat is None or lon is None:
        if not query:
            raise WeatherError("Provide coordinates or a location name.", "bad_request", 400)
        lat, lon, name = geocode(query)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise WeatherError("Invalid location coordinates.", "bad_request", 400)
    r = _get(CURRENT, {"lat": lat, "lon": lon, "units": "metric"})
    try:
        data = parse_current(r.json())
    except (ValueError, AttributeError):
        raise WeatherError("Unexpected response from the weather service.", "bad_response", 502)
    if all(data[k] is None for k in ("temperature_c", "humidity_pct")):
        raise WeatherError("The weather service returned incomplete data.", "bad_response", 502)
    data["latitude"], data["longitude"] = lat, lon
    data["location_name"] = name or data["location_name"]
    return data

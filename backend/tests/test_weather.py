import pytest, requests
from app.core.errors import WeatherError
from app.core import config
from app.services import weather

OK = {"main": {"temp": 24.5, "humidity": 80}, "wind": {"speed": 3.1}, "weather": [{"description": "light rain"}],
      "rain": {"1h": 0.6}, "name": "Pune", "sys": {"country": "IN"}, "coord": {"lat": 18.5, "lon": 73.8}, "dt": 1700000000}

class Resp:
    def __init__(self, code=200, data=None): self.status_code, self._d = code, data
    def json(self):
        if self._d is None: raise ValueError("bad json")
        return self._d

def _key(monkeypatch, key="abc"):
    import dataclasses
    s = dataclasses.replace(config.get_settings(), weather_api_key=key, weather_timeout_s=1)
    monkeypatch.setattr(weather, "get_settings", lambda: s)

def test_no_api_key(monkeypatch):
    _key(monkeypatch, "")
    with pytest.raises(WeatherError) as e: weather.get_weather(18.5, 73.8)
    assert e.value.code == "no_api_key"

@pytest.mark.parametrize("status,code", [(401, "invalid_api_key"), (404, "location_not_found"), (500, "api_failure"), (400, "bad_request")])
def test_http_errors(monkeypatch, status, code):
    _key(monkeypatch); monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(status, {}))
    with pytest.raises(WeatherError) as e: weather.get_weather(18.5, 73.8)
    assert e.value.code == code

def test_timeout_and_network(monkeypatch):
    _key(monkeypatch)
    def boom(exc):
        def f(*a, **k): raise exc
        return f
    monkeypatch.setattr(requests, "get", boom(requests.Timeout()))
    with pytest.raises(WeatherError) as e: weather.get_weather(1, 1)
    assert e.value.code == "timeout"
    monkeypatch.setattr(requests, "get", boom(requests.ConnectionError()))
    with pytest.raises(WeatherError) as e: weather.get_weather(1, 1)
    assert e.value.code == "network"

def test_success_parsing(monkeypatch):
    _key(monkeypatch); monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(200, OK))
    d = weather.get_weather(18.5, 73.8)
    assert d["temperature_c"] == 24.5 and d["humidity_pct"] == 80 and d["rainfall_mm"] == 0.6 and d["description"] == "light rain"

def test_invalid_coordinates(monkeypatch):
    _key(monkeypatch)
    with pytest.raises(WeatherError) as e: weather.get_weather(999, 0)
    assert e.value.code == "bad_request"

def test_invalid_location_name(monkeypatch):
    _key(monkeypatch); monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(200, []))
    with pytest.raises(WeatherError) as e: weather.get_weather(query="zzzzqq")
    assert e.value.code == "location_not_found"

def test_missing_fields_not_invented():
    d = weather.parse_current({"main": {"temp": 20}})
    assert d["humidity_pct"] is None and d["wind_speed_ms"] is None

def test_bad_json(monkeypatch):
    _key(monkeypatch); monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(200, None))
    with pytest.raises(WeatherError) as e: weather.get_weather(1, 1)
    assert e.value.code == "bad_response"

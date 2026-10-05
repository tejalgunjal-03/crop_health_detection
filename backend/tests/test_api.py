"""End-to-end API tests. ML predictors and the weather HTTP call are replaced by stubs, because unit
tests must not depend on trained weights or the internet. They verify WIRING and the fusion-input rule.
Requires: fastapi, sqlalchemy, httpx, torch installed (see requirements.txt)."""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.ml.fusion import engine

client = TestClient(app, raise_server_exceptions=False)

def _jpg(size=(300, 300)):
    b = io.BytesIO(); Image.new("RGB", size, (30, 140, 40)).save(b, "JPEG"); return b.getvalue()

def _auth(email="a@b.com"):
    r = client.post("/api/auth/register", json={"email": email, "name": "T", "password": "secret1"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}

LEAF = {"disease": "Common Rust", "confidence": 0.91, "model_arch": "stub",
        "probabilities": {"Common Rust": 0.91, "Blight": 0.04, "Healthy": 0.03, "Gray Leaf Spot": 0.02}}
WX = {"temperature_c": 21.0, "humidity_pct": 88.0, "rainfall_mm": 1.2, "wind_speed_ms": 2.0, "description": "rain",
      "location_name": "Test", "latitude": 18.5, "longitude": 73.8, "timestamp": "2026-01-01T00:00:00+00:00"}
UAV = {"overlay_file": "x.png", "image_width": 900, "image_height": 700, "n_tiles": 20, "class_fractions": {},
       "suspicious_pixel_fraction": 0.2, "regions": [{"id": 1}], "legend": {}, "mean_pixel_confidence": .9, "note": "n"}

def test_unauthenticated_rejected():
    assert client.post("/api/sessions", json={"monitoring_mode": "DIRECT_LEAF"}).status_code == 401

def test_missing_model_gives_clear_message_and_no_fake_prediction():
    h = _auth("m@b.com"); sid = client.post("/api/sessions", json={"monitoring_mode": "DIRECT_LEAF"}, headers=h).json()["session_id"]
    r = client.post("/api/leaf/analyze", data={"session_id": sid}, files={"file": ("l.jpg", _jpg(), "image/jpeg")}, headers=h)
    assert r.status_code == 503 and "Trained model not found" in r.json()["error"]["message"]

def test_invalid_image_rejected_cleanly():
    h = _auth("i@b.com"); sid = client.post("/api/sessions", json={"monitoring_mode": "DIRECT_LEAF"}, headers=h).json()["session_id"]
    r = client.post("/api/leaf/analyze", data={"session_id": sid}, files={"file": ("l.jpg", b"not an image", "image/jpeg")}, headers=h)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"

def test_direct_leaf_workflow_without_uav(monkeypatch):
    import app.services.leaf_inference as li, app.services.weather as wsv
    monkeypatch.setattr(li, "predict_leaf", lambda img: LEAF); monkeypatch.setattr(wsv, "get_weather", lambda *a, **k: dict(WX))
    h = _auth("d@b.com"); sid = client.post("/api/sessions", json={"monitoring_mode": "DIRECT_LEAF"}, headers=h).json()["session_id"]
    assert client.post("/api/leaf/analyze", data={"session_id": sid}, files={"file": ("l.jpg", _jpg(), "image/jpeg")}, headers=h).status_code == 200
    assert client.get(f"/api/weather?lat=18.5&lon=73.8&session_id={sid}", headers=h).status_code == 200
    r = client.post("/api/fusion/analyze", json={"session_id": sid}, headers=h); assert r.status_code == 200, r.text
    j = r.json()
    assert j["inputs_used"] == ["LeafAnalysis", "WeatherData"] and j["field_monitoring"] is None and j["disease"] == "Common Rust"
    assert client.get(f"/api/results/{j['id']}", headers=h).status_code == 200
    assert len(client.get("/api/results/history", headers=h).json()) == 1

def test_uav_workflow_fusion_input_excludes_uav(monkeypatch):
    import app.services.leaf_inference as li, app.services.uav_inference as ui, app.services.weather as wsv
    monkeypatch.setattr(li, "predict_leaf", lambda img: LEAF); monkeypatch.setattr(ui, "analyze_field", lambda img: dict(UAV))
    monkeypatch.setattr(wsv, "get_weather", lambda *a, **k: dict(WX))
    seen = {}
    real = engine.fuse
    def spy(leaf, weather, cfg):
        seen["args"] = (type(leaf).__name__, type(weather).__name__); return real(leaf, weather, cfg)
    import app.api.routes as routes; monkeypatch.setattr(routes, "fuse", spy)
    h = _auth("u@b.com"); sid = client.post("/api/sessions", json={"monitoring_mode": "UAV"}, headers=h).json()["session_id"]
    assert client.post("/api/uav/analyze", data={"session_id": sid}, files={"file": ("f.jpg", _jpg((900, 700)), "image/jpeg")}, headers=h).json()["is_final_diagnosis"] is False
    client.post("/api/leaf/analyze", data={"session_id": sid}, files={"file": ("l.jpg", _jpg(), "image/jpeg")}, headers=h)
    client.get(f"/api/weather?lat=18.5&lon=73.8&session_id={sid}", headers=h)
    j = client.post("/api/fusion/analyze", json={"session_id": sid}, headers=h).json()
    assert seen["args"] == ("LeafAnalysis", "WeatherData")
    assert j["inputs_used"] == ["LeafAnalysis", "WeatherData"]
    assert j["field_monitoring"]["used_in_fusion"] is False
    assert "uav" not in str(j["evidence"]).lower() and "uav" not in j["environmental_context"].lower()

def test_weather_failure_does_not_crash():
    h = _auth("w@b.com")
    r = client.get("/api/weather?q=", headers=h)
    assert r.status_code in (400, 404, 502, 503) and "error" in r.json()

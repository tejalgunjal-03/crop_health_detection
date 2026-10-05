import inspect
from app.core.config import get_settings, load_yaml
from app.ml.fusion.engine import fuse, LeafAnalysis, WeatherData

CFG = load_yaml(get_settings().fusion_config_path)
PROBS = {"Common Rust": 0.91, "Blight": 0.04, "Healthy": 0.03, "Gray Leaf Spot": 0.02}
GOOD = WeatherData(temperature_c=20, humidity_pct=85, rainfall_mm=1.0)
BAD = WeatherData(temperature_c=40, humidity_pct=20, rainfall_mm=0.0)


def test_fusion_signature_has_no_uav_parameter():
    params = list(inspect.signature(fuse).parameters)
    assert params == ["leaf", "weather", "cfg"]
    assert not any("uav" in p.lower() for p in params)


def test_fusion_inputs_recorded_as_leaf_and_weather_only():
    r = fuse(LeafAnalysis("Common Rust", 0.91, PROBS), GOOD, CFG)
    assert r["inputs_used"] == ["LeafAnalysis", "WeatherData"]
    assert set(r["evidence"]) == {"leaf", "weather", "weather_checks", "rules_applied"}
    assert "uav" not in str(r).lower()


def test_favourable_weather_raises_score_over_unfavourable():
    leaf = LeafAnalysis("Common Rust", 0.91, PROBS)
    assert fuse(leaf, GOOD, CFG)["fusion_score"] > fuse(leaf, BAD, CFG)["fusion_score"]


def test_healthy_leaf_is_healthy_regardless_of_weather():
    leaf = LeafAnalysis("Healthy", 0.97, {"Healthy": 0.97, "Common Rust": 0.01, "Blight": 0.01, "Gray Leaf Spot": 0.01})
    assert fuse(leaf, GOOD, CFG)["overall_condition"] == "Healthy"


def test_low_confidence_is_inconclusive():
    leaf = LeafAnalysis("Blight", 0.40, {"Blight": 0.40, "Common Rust": 0.35, "Healthy": 0.2, "Gray Leaf Spot": 0.05})
    assert fuse(leaf, GOOD, CFG)["overall_condition"] == "Inconclusive"


def test_missing_weather_still_works_and_says_so():
    r = fuse(LeafAnalysis("Common Rust", 0.91, PROBS), None, CFG)
    assert r["weather_used"] is False and "not available" in r["environmental_context"]


def test_recommendations_contain_no_dosage_language():
    for cond in ("Healthy", "Mild Stress", "Moderate Stress", "Severe/Critical", "Inconclusive"):
        text = " ".join(CFG["recommendations"][cond] + CFG["recommendations"]["common"]).lower()
        assert not any(w in text for w in ("mg", "ml/", "dose", "dosage", "spray", "fungicide"))

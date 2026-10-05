"""Rule-based, transparent DATA FUSION.

ARCHITECTURE RULE: this function accepts ONLY a LeafAnalysis and WeatherData.
It has no UAV parameter on purpose. UAV output is shown separately in the UI.

All thresholds live in configs/fusion_config.yaml and are PROJECT-DEFINED.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LeafAnalysis:
    disease: str                              # e.g. "Common Rust"
    confidence: float                         # top-class probability from the model
    probabilities: Optional[dict] = None      # {class name: probability}


@dataclass
class WeatherData:
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    rainfall_mm: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    description: Optional[str] = None
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[str] = None


def _favourable_fraction(disease: str, w: WeatherData, cfg: dict):
    """Return (fraction of favourable criteria, list of human-readable checks)."""
    crit = cfg["weather_favourability"].get(disease)
    if not crit:
        return None, []
    checks, hits = [], 0
    values = {"temperature_c": w.temperature_c, "humidity_pct": w.humidity_pct, "rainfall_mm": w.rainfall_mm}
    for key, (lo, hi) in crit.items():
        v = values.get(key)
        if v is None:
            continue
        ok = lo <= v <= hi
        hits += ok
        checks.append({"criterion": key, "value": v, "range": [lo, hi], "favourable": bool(ok)})
    if not checks:
        return None, []
    return hits / len(checks), checks


def fuse(leaf: LeafAnalysis, weather: Optional[WeatherData], cfg: dict) -> dict:
    """Combine leaf + weather into the final crop-health assessment."""
    rules: list[str] = []
    probs = leaf.probabilities or {leaf.disease: leaf.confidence}
    base = cfg["leaf_base_risk"]

    # 1) Leaf risk = expected base risk under the model's probabilities.
    leaf_risk = sum(p * base.get(name, 0.0) for name, p in probs.items())
    rules.append(f"Leaf risk = sum(probability x base_risk) = {leaf_risk:.3f}")
    score = leaf_risk

    # 2) Weather adjustment (only for a diseased leaf; weather alone never creates a disease).
    weather_used = weather is not None and any(
        v is not None for v in (weather.temperature_c, weather.humidity_pct, weather.rainfall_mm))
    checks, favourable = [], None
    if weather_used and leaf.disease != "Healthy":
        favourable, checks = _favourable_fraction(leaf.disease, weather, cfg)
        if favourable is not None:
            adj = cfg["weather_weight"] * (favourable - 0.5)
            score += adj
            rules.append(f"Weather favourability for {leaf.disease} = {favourable:.2f}; adjustment {adj:+.3f}")
        else:
            rules.append("No usable weather criteria for this disease; no weather adjustment")
    elif not weather_used:
        rules.append("Weather unavailable; assessment uses leaf evidence only")
    else:
        rules.append("Leaf predicted Healthy; weather does not raise the score")
    if weather_used and weather.temperature_c is not None and weather.temperature_c >= cfg["heat_stress_c"]:
        score += cfg["heat_stress_bonus"]
        rules.append(f"Temperature >= {cfg['heat_stress_c']} C: +{cfg['heat_stress_bonus']} heat stress")

    score = max(0.0, min(1.0, score))

    # 3) Map to label.
    lab = cfg["labels"]
    if leaf.confidence < cfg["min_reliable_confidence"]:
        condition = "Inconclusive"
        rules.append(f"Top confidence {leaf.confidence:.2f} < {cfg['min_reliable_confidence']}: Inconclusive")
    elif score < lab["healthy_below"]:
        condition = "Healthy"
    elif score < lab["mild_below"]:
        condition = "Mild Stress"
    elif score < lab["moderate_below"]:
        condition = "Moderate Stress"
    else:
        condition = "Severe/Critical"
    rules.append(f"Final score {score:.3f} -> {condition}")

    # 4) Environmental context text (descriptive only).
    if not weather_used:
        env = "Weather data was not available, so environmental context could not be assessed."
    else:
        bits = []
        if weather.temperature_c is not None: bits.append(f"{weather.temperature_c:.1f} C")
        if weather.humidity_pct is not None: bits.append(f"{weather.humidity_pct:.0f}% humidity")
        if weather.rainfall_mm is not None: bits.append(f"{weather.rainfall_mm:.1f} mm recent rain")
        env = "Current conditions: " + ", ".join(bits) + "."
        if favourable is not None:
            env += (" These conditions are favourable to " + leaf.disease + " under the project's rules."
                    if favourable >= 0.67 else
                    " These conditions are only partly favourable to " + leaf.disease + " under the project's rules."
                    if favourable >= 0.34 else
                    " These conditions are mostly unfavourable to " + leaf.disease + " under the project's rules.")

    recs = list(cfg["recommendations"].get(condition, [])) + list(cfg["recommendations"]["common"])
    return {
        "overall_condition": condition,
        "disease": leaf.disease,
        "confidence": leaf.confidence,
        "severity": cfg["severity_by_condition"].get(condition),
        "fusion_score": round(score, 4),
        "environmental_context": env,
        "recommendations": recs,
        "inputs_used": ["LeafAnalysis", "WeatherData"],     # never includes UAV
        "weather_used": bool(weather_used),
        "evidence": {
            "leaf": asdict(leaf),
            "weather": asdict(weather) if weather else None,
            "weather_checks": checks,
            "rules_applied": rules,
        },
        "disclaimer": "Thresholds are project-defined rules, not universal agronomic standards.",
    }

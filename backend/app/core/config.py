"""Central configuration: environment variables (.env) + YAML config files."""
from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[3]          # project root
if load_dotenv:
    load_dotenv(ROOT / ".env")


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    weather_api_key: str
    secret_key: str
    database_url: str
    leaf_model_path: Path
    uav_model_path: Path
    uploads_dir: Path
    results_dir: Path
    fusion_config_path: Path
    uav_classes_path: Path
    max_upload_bytes: int
    cors_origins: list
    weather_timeout_s: float
    token_minutes: int
    uav_max_pixels: int          # reject UAV images larger than this (RAM protection)
    uav_min_region_px: int       # ignore suspicious regions smaller than this


@lru_cache
def get_settings() -> Settings:
    e = os.getenv
    return Settings(
        weather_api_key=e("WEATHER_API_KEY", "").strip(),
        secret_key=e("SECRET_KEY", "dev-only-insecure-key"),
        database_url=e("DATABASE_URL", f"sqlite:///{(ROOT / 'maize_monitor.db').as_posix()}"),
        leaf_model_path=_path(e("LEAF_MODEL_PATH", "models/leaf/best_model.pth")),
        uav_model_path=_path(e("UAV_MODEL_PATH", "models/uav/best_model.pth")),
        uploads_dir=_path(e("UPLOADS_DIR", "uploads")),
        results_dir=_path(e("RESULTS_DIR", "results")),
        fusion_config_path=_path(e("FUSION_CONFIG", "configs/fusion_config.yaml")),
        uav_classes_path=_path(e("UAV_CLASSES", "configs/uav_classes.yaml")),
        max_upload_bytes=int(float(e("MAX_UPLOAD_MB", "15")) * 1024 * 1024),
        cors_origins=[o.strip() for o in e("CORS_ORIGINS", "http://localhost:5173").split(",")],
        weather_timeout_s=float(e("WEATHER_TIMEOUT_S", "8")),
        token_minutes=int(e("TOKEN_MINUTES", "720")),
        uav_max_pixels=int(e("UAV_MAX_PIXELS", str(24_000_000))),
        uav_min_region_px=int(e("UAV_MIN_REGION_PX", "50")),
    )


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

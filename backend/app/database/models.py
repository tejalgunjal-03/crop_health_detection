"""Database schema. NOTE: CropHealthResult links to Leaf + Weather only; UAVAnalysis hangs off the
session for display and has NO link to the result/fusion."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .session import Base

now = lambda: datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class CropField(Base):
    __tablename__ = "crop_fields"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(120), default="My maize field")
    crop: Mapped[str] = mapped_column(String(30), default="maize")


class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    field_id: Mapped[int] = mapped_column(ForeignKey("crop_fields.id"))
    monitoring_mode: Mapped[str] = mapped_column(String(20))            # "UAV" | "DIRECT_LEAF"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    uav_analysis_id: Mapped[Optional[int]] = mapped_column(ForeignKey("uav_analyses.id", use_alter=True, name="fk_session_uav_analysis_id"), nullable=True)
    leaf_analysis_id: Mapped[Optional[int]] = mapped_column(ForeignKey("leaf_analyses.id", use_alter=True, name="fk_session_leaf_analysis_id"), nullable=True)
    weather_data_id: Mapped[Optional[int]] = mapped_column(ForeignKey("weather_data.id", use_alter=True, name="fk_session_weather_data_id"), nullable=True)
    final_result_id: Mapped[Optional[int]] = mapped_column(ForeignKey("crop_health_results.id", use_alter=True, name="fk_session_final_result_id"), nullable=True)
    field = relationship("CropField")


class UAVImage(Base):
    __tablename__ = "uav_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("monitoring_sessions.id"))
    file_path: Mapped[str] = mapped_column(String(500))
    width: Mapped[int] = mapped_column(Integer); height: Mapped[int] = mapped_column(Integer)


class UAVAnalysis(Base):
    __tablename__ = "uav_analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("uav_images.id"))
    overlay_file: Mapped[str] = mapped_column(String(300))
    details: Mapped[dict] = mapped_column(JSON)          # fractions, regions, legend, note
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class LeafImage(Base):
    __tablename__ = "leaf_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("monitoring_sessions.id"))
    file_path: Mapped[str] = mapped_column(String(500))


class LeafAnalysis(Base):
    __tablename__ = "leaf_analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("leaf_images.id"))
    disease: Mapped[str] = mapped_column(String(60))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSON)
    model_arch: Mapped[str] = mapped_column(String(60), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class WeatherRecord(Base):
    __tablename__ = "weather_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    temperature_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rainfall_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_speed_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    location_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    observed_at: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)


class CropHealthResult(Base):
    __tablename__ = "crop_health_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("monitoring_sessions.id"))
    leaf_analysis_id: Mapped[int] = mapped_column(ForeignKey("leaf_analyses.id"))
    weather_data_id: Mapped[Optional[int]] = mapped_column(ForeignKey("weather_data.id"), nullable=True)
    overall_condition: Mapped[str] = mapped_column(String(40))
    severity: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    disease: Mapped[str] = mapped_column(String(60))
    confidence: Mapped[float] = mapped_column(Float)
    fusion_score: Mapped[float] = mapped_column(Float)
    environmental_context: Mapped[str] = mapped_column(Text)
    full: Mapped[dict] = mapped_column(JSON)             # complete fusion output incl. rules applied
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("crop_health_results.id"), index=True)
    text: Mapped[str] = mapped_column(Text)

"""All HTTP endpoints. Route functions stay thin; logic lives in services/ and ml/."""
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..core.config import get_settings, load_yaml
from ..core.errors import AppError
from ..core.security import hash_password, make_token, verify_password
from ..database import models as M
from ..ml.fusion.engine import LeafAnalysis, WeatherData, fuse
from ..schemas.schemas import FusionIn, LoginIn, RegisterIn, SessionIn
from ..services import weather as weather_svc
from ..services.image_validation import save_upload, validate_image_bytes
from .deps import current_user, get_db, own_session

router = APIRouter(prefix="/api")


async def _read_upload(f: UploadFile) -> bytes:
    s = get_settings()
    data = await f.read(s.max_upload_bytes + 1)     # never read more than the limit + 1 byte
    return data


# ---------------- auth ----------------
@router.post("/auth/register")
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if db.scalar(select(M.User).where(M.User.email == body.email.lower())):
        raise AppError("An account with this email already exists.", "duplicate", 409)
    u = M.User(email=body.email.lower(), name=body.name, password_hash=hash_password(body.password))
    db.add(u); db.commit()
    return {"token": make_token(u.id), "user": {"id": u.id, "name": u.name, "email": u.email}}


@router.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    u = db.scalar(select(M.User).where(M.User.email == body.email.lower()))
    if not u or not verify_password(body.password, u.password_hash):
        raise AppError("Incorrect email or password.", "auth", 401)
    return {"token": make_token(u.id), "user": {"id": u.id, "name": u.name, "email": u.email}}


# ---------------- sessions ----------------
@router.post("/sessions")
def create_session(body: SessionIn, db: Session = Depends(get_db), user: M.User = Depends(current_user)):
    field = db.scalar(select(M.CropField).where(M.CropField.user_id == user.id))
    if not field:
        field = M.CropField(user_id=user.id, name=body.field_name or "My maize field"); db.add(field); db.flush()
    s = M.MonitoringSession(user_id=user.id, field_id=field.id, monitoring_mode=body.monitoring_mode)
    db.add(s); db.commit()
    return {"session_id": s.id, "monitoring_mode": s.monitoring_mode}


# ---------------- UAV (localisation only) ----------------
@router.post("/uav/analyze")
async def uav_analyze(session_id: int = Form(...), file: UploadFile = File(...),
                      db: Session = Depends(get_db), user: M.User = Depends(current_user)):
    s = own_session(db, user, session_id)
    if s.monitoring_mode != "UAV":
        raise AppError("This session was started in Direct Leaf mode.", "bad_mode", 400)
    cfg = get_settings()
    data = await _read_upload(file)
    img, ext = validate_image_bytes(data, cfg.max_upload_bytes, cfg.uav_max_pixels)
    from ..services.uav_inference import analyze_field      # imported lazily: torch is only needed here
    result = analyze_field(img)                              # raises ModelNotFoundError if untrained
    path = save_upload(data, cfg.uploads_dir / "uav", ext)
    im = M.UAVImage(session_id=s.id, file_path=str(path), width=img.width, height=img.height); db.add(im); db.flush()
    an = M.UAVAnalysis(image_id=im.id, overlay_file=result["overlay_file"], details=result); db.add(an); db.flush()
    s.uav_analysis_id = an.id; db.commit()
    return {"id": an.id, "overlay_url": f"/files/uav/{an.overlay_file}", **result,
            "next_step": "Inspect the highlighted region and upload a close-up maize leaf image.",
            "is_final_diagnosis": False}


@router.get("/uav/result/{aid}")
def uav_result(aid: int, db: Session = Depends(get_db), user: M.User = Depends(current_user)):
    a = db.get(M.UAVAnalysis, aid)
    owner = db.scalar(select(M.MonitoringSession).where(M.MonitoringSession.uav_analysis_id == aid)) if a else None
    if not a or not owner or owner.user_id != user.id:
        raise AppError("UAV result not found.", "not_found", 404)
    return {"id": a.id, "overlay_url": f"/files/uav/{a.overlay_file}", **a.details, "is_final_diagnosis": False}


# ---------------- leaf ----------------
@router.post("/leaf/analyze")
async def leaf_analyze(session_id: int = Form(...), file: UploadFile = File(...),
                       db: Session = Depends(get_db), user: M.User = Depends(current_user)):
    s = own_session(db, user, session_id)
    cfg = get_settings()
    data = await _read_upload(file)
    img, ext = validate_image_bytes(data, cfg.max_upload_bytes)
    from ..services.leaf_inference import predict_leaf
    r = predict_leaf(img)
    path = save_upload(data, cfg.uploads_dir / "leaf", ext)
    im = M.LeafImage(session_id=s.id, file_path=str(path)); db.add(im); db.flush()
    an = M.LeafAnalysis(image_id=im.id, disease=r["disease"], confidence=r["confidence"],
                        probabilities=r["probabilities"], model_arch=r["model_arch"]); db.add(an); db.flush()
    s.leaf_analysis_id = an.id; db.commit()
    return {"id": an.id, **r}


@router.get("/leaf/result/{aid}")
def leaf_result(aid: int, db: Session = Depends(get_db), user: M.User = Depends(current_user)):
    a = db.get(M.LeafAnalysis, aid)
    owner = db.scalar(select(M.MonitoringSession).where(M.MonitoringSession.leaf_analysis_id == aid)) if a else None
    if not a or not owner or owner.user_id != user.id:
        raise AppError("Leaf result not found.", "not_found", 404)
    return {"id": a.id, "disease": a.disease, "confidence": a.confidence, "probabilities": a.probabilities}


# ---------------- weather ----------------
@router.get("/weather")
def weather(lat: Optional[float] = Query(None), lon: Optional[float] = Query(None), q: Optional[str] = Query(None),
            session_id: Optional[int] = Query(None),
            db: Session = Depends(get_db), user: M.User = Depends(current_user)):
    data = weather_svc.get_weather(lat, lon, q)      # WeatherError -> friendly JSON via handler
    if session_id is not None:
        s = own_session(db, user, session_id)
        rec = M.WeatherRecord(observed_at=data.pop("timestamp"), **data); db.add(rec); db.flush()
        data["timestamp"] = rec.observed_at
        s.weather_data_id = rec.id; db.commit()
    return data


# ---------------- fusion: Leaf + Weather ONLY ----------------
@router.post("/fusion/analyze")
def fusion(body: FusionIn, db: Session = Depends(get_db), user: M.User = Depends(current_user)):
    s = own_session(db, user, body.session_id)
    if not s.leaf_analysis_id:
        raise AppError("Please analyse a leaf image first.", "missing_leaf", 400)
    la = db.get(M.LeafAnalysis, s.leaf_analysis_id)
    wr = db.get(M.WeatherRecord, s.weather_data_id) if s.weather_data_id else None
    # Only these two objects reach the fusion engine. The UAV analysis is never loaded here.
    leaf = LeafAnalysis(la.disease, la.confidence, la.probabilities)
    wx = WeatherData(wr.temperature_c, wr.humidity_pct, wr.rainfall_mm, wr.wind_speed_ms, wr.description,
                     wr.location_name, wr.latitude, wr.longitude, wr.observed_at) if wr else None
    out = fuse(leaf, wx, load_yaml(get_settings().fusion_config_path))
    res = M.CropHealthResult(session_id=s.id, leaf_analysis_id=la.id, weather_data_id=wr.id if wr else None,
                             overall_condition=out["overall_condition"], severity=out["severity"], disease=out["disease"],
                             confidence=out["confidence"], fusion_score=out["fusion_score"],
                             environmental_context=out["environmental_context"], full=out)
    db.add(res); db.flush()
    for t in out["recommendations"]:
        db.add(M.Recommendation(result_id=res.id, text=t))
    s.final_result_id = res.id; db.commit()
    return _result_payload(db, res, s)


def _result_payload(db: Session, res: M.CropHealthResult, s: M.MonitoringSession) -> dict:
    payload = {"id": res.id, "session_id": s.id, "monitoring_mode": s.monitoring_mode,
               "created_at": res.created_at.isoformat(), **res.full}
    # Separate block: shown in the UI as "Field monitoring information". Not part of the fusion output above.
    if s.uav_analysis_id:
        u = db.get(M.UAVAnalysis, s.uav_analysis_id)
        payload["field_monitoring"] = {"overlay_url": f"/files/uav/{u.overlay_file}", **u.details,
                                       "used_in_fusion": False}
    else:
        payload["field_monitoring"] = None
    return payload


@router.get("/results/history")
def history(db: Session = Depends(get_db), user: M.User = Depends(current_user)):
    rows = db.execute(select(M.CropHealthResult, M.MonitoringSession).join(
        M.MonitoringSession, M.CropHealthResult.session_id == M.MonitoringSession.id)
        .where(M.MonitoringSession.user_id == user.id).order_by(M.CropHealthResult.id.desc()).limit(100)).all()
    return [{"id": r.id, "created_at": r.created_at.isoformat(), "monitoring_mode": s.monitoring_mode,
             "overall_condition": r.overall_condition, "disease": r.disease, "confidence": r.confidence} for r, s in rows]


@router.get("/results/{rid}")
def result(rid: int, db: Session = Depends(get_db), user: M.User = Depends(current_user)):
    r = db.get(M.CropHealthResult, rid)
    s = db.get(M.MonitoringSession, r.session_id) if r else None
    if not r or s.user_id != user.id:
        raise AppError("Result not found.", "not_found", 404)
    return _result_payload(db, r, s)


@router.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "leaf_model_present": s.leaf_model_path.exists(), "uav_model_present": s.uav_model_path.exists(),
            "weather_key_configured": bool(s.weather_api_key)}

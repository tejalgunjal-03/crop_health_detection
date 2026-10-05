from fastapi import Depends, Header
from sqlalchemy.orm import Session
from ..core.errors import AppError
from ..core.security import read_token
from ..database.models import MonitoringSession, User
from ..database.session import SessionLocal


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(authorization: str = Header(default=""), db: Session = Depends(get_db)) -> User:
    if not authorization.lower().startswith("bearer "):
        raise AppError("Please sign in first.", "auth", 401)
    user = db.get(User, read_token(authorization[7:].strip()))
    if not user:
        raise AppError("Account not found. Please sign in again.", "auth", 401)
    return user


def own_session(db: Session, user: User, session_id: int) -> MonitoringSession:
    s = db.get(MonitoringSession, session_id)
    if not s or s.user_id != user.id:
        raise AppError("Monitoring session not found.", "not_found", 404)
    return s

"""Password hashing (PBKDF2) and JWT tokens."""
import hashlib, hmac, os
from datetime import datetime, timedelta, timezone
import jwt
from .config import get_settings
from .errors import AppError


def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    return salt.hex() + "$" + hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000).hex()


def verify_password(pw: str, stored: str) -> bool:
    try:
        salt_hex, h = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), 200_000).hex()
        return hmac.compare_digest(calc, h)
    except ValueError:
        return False


def make_token(user_id: int) -> str:
    s = get_settings()
    exp = datetime.now(timezone.utc) + timedelta(minutes=s.token_minutes)
    return jwt.encode({"sub": str(user_id), "exp": exp}, s.secret_key, algorithm="HS256")


def read_token(token: str) -> int:
    try:
        return int(jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])["sub"])
    except Exception:
        raise AppError("Your login has expired. Please sign in again.", "auth", 401)

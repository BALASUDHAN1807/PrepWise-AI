"""Password hashing (bcrypt), JWT tokens and auth dependencies."""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.core.database import get_db

_bearer = HTTPBearer(auto_error=False)
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """Hash a password with bcrypt (salted, adaptive cost)."""
    # bcrypt only uses the first 72 bytes; pre-hash longer inputs to keep entropy.
    pw = password.encode("utf-8")
    if len(pw) > 72:
        pw = hashlib.sha256(pw).hexdigest().encode()
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time bcrypt verification."""
    pw = password.encode("utf-8")
    if len(pw) > 72:
        pw = hashlib.sha256(pw).hexdigest().encode()
    try:
        return bcrypt.checkpw(pw, hashed.encode())
    except ValueError:
        return False


def create_access_token(user_id: int, role: str) -> str:
    """Create a signed JWT for the user."""
    s = get_settings()
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": datetime.now(UTC) + timedelta(minutes=s.jwt_expire_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, get_settings().jwt_secret, algorithms=[ALGORITHM])


def anonymize_id(user_id: int) -> str:
    """Stable, salted pseudonym for a user (used in admin views and exports)."""
    digest = hashlib.sha256(f"{get_settings().anon_salt}:{user_id}".encode()).hexdigest()
    return f"student-{digest[:8]}"


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
):
    """Resolve the logged-in user from the Bearer token or raise 401."""
    from backend.models.db_models import User

    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        payload = decode_token(creds.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token") from exc
    user = db.get(User, int(payload["sub"]))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return user


def require_admin(user=Depends(get_current_user)):
    """Dependency that only lets admins through."""
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return user


def require_consent(user=Depends(get_current_user)):
    """Dependency for endpoints that store personal data (resume, answers)."""
    if not user.consent_given:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Please accept the data storage consent in your profile before using this feature.",
        )
    return user

from __future__ import annotations
"""
server/auth_router.py
M3: /api/auth/* endpoints — register and login.
Owner: Sunishka Sarkar
"""

import uuid
import datetime

import bcrypt
import jwt as pyjwt
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db_handler import read_db, write_db, get_user

# Environment variable key for JWT signing
import os
JWT_SECRET = os.environ.get("JWT_SECRET", "dev_secret_change_in_production")
JWT_ALGO   = "HS256"
JWT_EXP_HOURS = 24

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username:      str
    password:      str
    hospital_name: str
    contact_email: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def create_jwt(user_id: str) -> str:
    exp = datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXP_HOURS)
    payload = {"sub": user_id, "exp": exp}
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def verify_jwt(token: str) -> dict:
    """Decode and validate a JWT; returns the payload dict."""
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except pyjwt.ExpiredSignatureError:
        raise ValueError("Token has expired.")
    except pyjwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/register")
async def register_user(payload: RegisterRequest):
    """POST /api/auth/register — create a new hospital account."""
    db = read_db()

    # Uniqueness check
    existing = next((u for u in db["users"] if u["username"] == payload.username), None)
    if existing:
        return JSONResponse(
            status_code=409,
            content={"detail": f"Username '{payload.username}' is already taken."},
        )

    # Hash password
    pw_hash = bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt()).decode()

    user_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow().isoformat() + "Z"
    new_user = {
        "user_id":          user_id,
        "username":         payload.username,
        "password_hash":    pw_hash,
        "hospital_name":    payload.hospital_name,
        "contact_email":    payload.contact_email,
        "approved_projects": [],
        "pending_projects":  [],
        "created_at":       now,
        "last_active":      now,
    }
    db["users"].append(new_user)
    write_db(db)

    return JSONResponse(
        status_code=201,
        content={"user_id": user_id, "username": payload.username},
    )


@router.post("/login")
async def login_user(payload: LoginRequest):
    """POST /api/auth/login — validate credentials and issue JWT."""
    db = read_db()
    user = next((u for u in db["users"] if u["username"] == payload.username), None)

    if user is None or not bcrypt.checkpw(
        payload.password.encode(), user["password_hash"].encode()
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid username or password."},
        )

    # Update last_active
    user["last_active"] = datetime.datetime.utcnow().isoformat() + "Z"
    write_db(db)

    token = create_jwt(user["user_id"])
    return JSONResponse(
        status_code=200,
        content={
            "access_token":      token,
            "token_type":        "bearer",
            "approved_projects": user.get("approved_projects", []),
            "user_id":           user["user_id"],
        },
    )

"""Authentication: validate demo credentials and issue/verify role-tagged tokens.

Tokens are signed JWTs so the backend stays stateless — the role is carried
(and integrity-protected) inside the token and verified on every ``/chat``
request. The role inside the verified token is what drives RBAC; it cannot be
tampered with client-side without invalidating the signature.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from pydantic import BaseModel

from .config import DEMO_USERS, settings


class AuthError(Exception):
    """Raised when credentials are invalid or a token cannot be verified."""


class Identity(BaseModel):
    username: str
    role: str
    display_name: str


def authenticate(username: str, password: str) -> Identity:
    """Validate credentials, returning the resolved identity or raising."""
    user = DEMO_USERS.get(username)
    if user is None or user.password != password:
        raise AuthError("Invalid username or password.")
    return Identity(
        username=user.username, role=user.role, display_name=user.display_name
    )


def create_token(identity: Identity) -> str:
    """Create a signed JWT carrying the user's role."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": identity.username,
        "role": identity.role,
        "name": identity.display_name,
        "iat": now,
        "exp": now + timedelta(minutes=settings.token_ttl_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def verify_token(token: str) -> Identity:
    """Verify a JWT and return the embedded identity, or raise ``AuthError``."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Session expired. Please log in again.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Invalid session token.") from exc
    return Identity(
        username=payload["sub"], role=payload["role"], display_name=payload.get("name", "")
    )


def parse_bearer(authorization: Optional[str]) -> Identity:
    """Parse an ``Authorization: Bearer <token>`` header into an identity."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthError("Missing or malformed Authorization header.")
    token = authorization.split(" ", 1)[1].strip()
    return verify_token(token)

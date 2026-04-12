"""API key authentication middleware."""
from __future__ import annotations

import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key():
    """Return the API key from the BIKEAI_API_KEY environment variable."""
    key = os.environ.get("BIKEAI_API_KEY")
    if not key:
        raise RuntimeError(
            "BIKEAI_API_KEY env var not set. "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    return key


async def require_api_key(api_key: str | None = Security(_header)) -> str:
    """FastAPI dependency. Add to any endpoint that requires authentication.

    Usage:
        @app.get("/predict/{station_id}", dependencies=[Depends(require_api_key)])
    """
    expected = get_api_key()
    if api_key is None or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

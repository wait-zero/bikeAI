"""bikeAI serving — FastAPI application.

Endpoints:
    GET  /                              → service info (public)
    GET  /health                        → liveness probe (public)
    GET  /stations                      → list stations (requires API key)
    POST /predict                       → single prediction (requires API key)
    POST /predict/batch                 → batch prediction (requires API key)
    GET  /docs                          → Swagger UI (auto-generated)

Start:
    export BIKEAI_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    PYTHONPATH=src uvicorn bikeai.serving.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import socket
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query

from bikeai.serving.artifact import Artifact
from bikeai.serving.auth import require_api_key
from bikeai.serving.inference import predict_one
from bikeai.serving.schemas import (
    BatchPredictRequest,
    PredictRequest,
    PredictResponse,
    StationInfo,
)

START_TIME = datetime.now(timezone.utc)
HOSTNAME = socket.gethostname()

_artifact: Artifact | None = None


def _load_artifact() -> Artifact:
    model_dir = os.environ.get(
        "BIKEAI_MODEL_DIR",
        str(Path(__file__).resolve().parents[3] / "models_out" / "v1"),
    )
    return Artifact.load(Path(model_dir))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifact
    _artifact = _load_artifact()
    print(f"✅ model loaded: {_artifact.version}, {len(_artifact.boosters)} boosters, "
          f"{len(_artifact.station_stats)} stations")
    yield


app = FastAPI(
    title="bikeAI",
    version="1.0.0",
    description="Seoul Ttareungi per-station availability forecasting API",
    lifespan=lifespan,
)


def get_artifact() -> Artifact:
    if _artifact is None:
        raise HTTPException(503, "model not loaded")
    return _artifact


# ── Public endpoints ──────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "bikeAI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    art = _artifact
    uptime_sec = int((datetime.now(timezone.utc) - START_TIME).total_seconds())
    return {
        "status": "ok",
        "hostname": HOSTNAME,
        "uptime_sec": uptime_sec,
        "model_loaded": art is not None,
        "model_version": art.version if art else None,
        "stations": len(art.station_stats) if art else 0,
        "horizons": art.horizons if art else [],
    }


# ── Authenticated endpoints ───────────────────────────────────────

@app.get(
    "/stations",
    response_model=list[StationInfo],
    dependencies=[Depends(require_api_key)],
)
def list_stations(
    limit: int = Query(50, ge=1, le=3000),
    offset: int = Query(0, ge=0),
    artifact: Artifact = Depends(get_artifact),
):
    """Return station metadata (id, name, lat, lon, mean activity)."""
    df = artifact.station_stats.sort_values("st_mean", ascending=False)
    page = df.iloc[offset : offset + limit]
    return [
        StationInfo(
            station_id=r["station_id"],
            station_name=r.get("station_name"),
            lat=r.get("lat"),
            lon=r.get("lon"),
            st_mean=r["st_mean"],
            st_max=r["st_max"],
        )
        for _, r in page.iterrows()
    ]


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(require_api_key)],
)
def predict(request: PredictRequest, artifact: Artifact = Depends(get_artifact)):
    """Predict bike count deltas for a single station.

    The caller must provide `station_id` and `current_bike_count`.
    Optional `history` (list of past observations) improves accuracy.
    Optional `weather` (temp_c, precip_mm, etc.) adds weather features.
    """
    try:
        return predict_one(request, artifact)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post(
    "/predict/batch",
    response_model=list[PredictResponse],
    dependencies=[Depends(require_api_key)],
)
def predict_batch(
    request: BatchPredictRequest,
    artifact: Artifact = Depends(get_artifact),
):
    """Predict for multiple stations in one call."""
    results = []
    for item in request.items:
        try:
            results.append(predict_one(item, artifact))
        except KeyError:
            pass
    return results

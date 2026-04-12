"""Pydantic request/response models for the bikeAI API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HistoryPoint(BaseModel):
    minutes_ago: int = Field(..., ge=1, le=60)
    bike_count: int = Field(..., ge=0)


class PredictRequest(BaseModel):
    """Mode B (stateless) input. Caller provides current count and optional history."""
    station_id: str
    current_bike_count: int = Field(..., ge=0)
    history: list[HistoryPoint] = Field(default_factory=list)
    weather: dict | None = None
    horizons: list[int] = Field(default=[5, 15, 30, 60])


class Prediction(BaseModel):
    horizon_min: int
    predicted_count: int
    predicted_delta: float
    method: str  # "persistence" | "model"


class PredictResponse(BaseModel):
    station_id: str
    station_name: str | None = None
    lat: float | None = None
    lon: float | None = None
    as_of: datetime
    current_bike_count: int
    predictions: list[Prediction]
    warnings: list[str] = Field(default_factory=list)
    model_version: str


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]


class StationInfo(BaseModel):
    station_id: str
    station_name: str | None = None
    lat: float | None = None
    lon: float | None = None
    st_mean: float
    st_max: float

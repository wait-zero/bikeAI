"""Feature building + model inference at serving time.

Given a station_id + current_bike_count (+ optional history/weather), produce
a PredictResponse with per-horizon delta and absolute count predictions.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bikeai.serving.artifact import Artifact
from bikeai.serving.schemas import (
    HistoryPoint,
    Prediction,
    PredictRequest,
    PredictResponse,
)

try:
    from holidayskr import is_holiday
except ImportError:
    def is_holiday(d: str) -> bool:
        return False


def predict_one(request: PredictRequest, artifact: Artifact) -> PredictResponse:
    """Build features from the request and run all horizon boosters."""
    now = datetime.now(timezone.utc)
    info = artifact.station_info(request.station_id)
    warnings: list[str] = []

    if info is None:
        raise KeyError(f"unknown station_id: {request.station_id}")

    # --- build feature row ---
    bc = request.current_bike_count
    st_mean = info["st_mean"]
    st_std = info["st_std"]
    st_max = info["st_max"]

    # Time features (use server local time)
    from zoneinfo import ZoneInfo
    kst = now.astimezone(ZoneInfo("Asia/Seoul"))
    hour = kst.hour
    minute = kst.minute
    dow = kst.weekday()
    month = kst.month
    is_weekend = int(dow >= 5)
    date_str = kst.strftime("%Y-%m-%d")
    try:
        holiday = int(bool(is_holiday(date_str)))
    except Exception:
        holiday = 0

    hour_frac = hour + minute / 60
    hour_sin = math.sin(2 * math.pi * hour_frac / 24)
    hour_cos = math.cos(2 * math.pi * hour_frac / 24)
    dow_sin = math.sin(2 * math.pi * dow / 7)
    dow_cos = math.cos(2 * math.pi * dow / 7)

    # Normalization
    bc_centered = bc - st_mean
    bc_z = bc_centered / st_std if st_std > 0 else 0.0
    bc_pct = bc / st_max if st_max > 0 else 0.0

    # Lag features from history (or fill with current if missing)
    history_map = {hp.minutes_ago: hp.bike_count for hp in request.history}
    lag_1 = history_map.get(1, bc)
    lag_5 = history_map.get(5, bc)
    lag_15 = history_map.get(15, bc)

    if not request.history:
        warnings.append("No history provided; lag features filled with current count (degraded)")

    # Rolling means from history
    hist_values = [history_map.get(i, bc) for i in range(1, 6)]
    roll_mean_5 = float(np.mean(hist_values))
    hist_values_15 = [history_map.get(i, bc) for i in range(1, 16)]
    roll_mean_15 = float(np.mean(hist_values_15))

    # Neighbor mean (from artifact cache or degraded)
    neighbor_ids = artifact.neighbors_of(request.station_id)
    neighbor_mean = float(bc)  # degraded default
    warnings.append("Neighbor mean unavailable in stateless mode; using current count")

    # Weather
    weather = request.weather or {}
    temp_c = weather.get("temp_c", float("nan"))
    precip_mm = weather.get("precip_mm", 0.0)
    wind_ms = weather.get("wind_ms", float("nan"))
    humidity_pct = weather.get("humidity_pct", float("nan"))
    if not request.weather:
        warnings.append("No weather provided; using NaN (model handles gracefully)")

    # --- assemble feature row in trained feature order ---
    schema = artifact.feature_schema
    feature_names = schema["features"]
    categoricals = set(schema.get("categorical", []))

    feature_values = {
        "station_id": request.station_id,
        "hour": hour,
        "dow": dow,
        "month": month,
        "is_weekend": is_weekend,
        "is_holiday": holiday,
        "bike_count": bc,
        "bike_count_centered": bc_centered,
        "bike_count_z": bc_z,
        "bike_count_pct": bc_pct,
        "st_mean": st_mean,
        "st_std": st_std,
        "st_max": st_max,
        "lat": info["lat"] or 0.0,
        "lon": info["lon"] or 0.0,
        "neighbor_mean_count": neighbor_mean,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "temp_c": temp_c,
        "precip_mm": precip_mm,
        "wind_ms": wind_ms,
        "humidity_pct": humidity_pct,
        "lag_1": lag_1,
        "lag_5": lag_5,
        "lag_15": lag_15,
        "roll_mean_5": roll_mean_5,
        "roll_mean_15": roll_mean_15,
    }

    row = pd.DataFrame([{fn: feature_values.get(fn) for fn in feature_names}])
    for c in categoricals:
        if c in row.columns:
            row[c] = row[c].astype("category")

    # --- run boosters ---
    predictions: list[Prediction] = []
    for h in request.horizons:
        if h not in artifact.boosters:
            continue
        if h == 5:
            # Hybrid: 5min = persistence
            predictions.append(
                Prediction(
                    horizon_min=5,
                    predicted_count=bc,
                    predicted_delta=0.0,
                    method="persistence",
                )
            )
            continue
        booster = artifact.boosters[h]
        delta = float(booster.predict(row, num_iteration=booster.best_iteration or None)[0])
        predicted_count = max(0, int(round(bc + delta)))
        predictions.append(
            Prediction(
                horizon_min=h,
                predicted_count=predicted_count,
                predicted_delta=round(delta, 2),
                method="model",
            )
        )

    return PredictResponse(
        station_id=request.station_id,
        station_name=info.get("station_name"),
        lat=info.get("lat"),
        lon=info.get("lon"),
        as_of=now,
        current_bike_count=bc,
        predictions=predictions,
        warnings=warnings,
        model_version=artifact.version,
    )

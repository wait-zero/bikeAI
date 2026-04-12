"""Tests for time/spatial/build_dataset feature builders."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bikeai.features.build_dataset import build_feature_table
from bikeai.features.spatial_features import build_neighbor_index, haversine_m
from bikeai.features.time_features import add_time_features


def test_add_time_features_produces_cyclical_columns():
    df = pd.DataFrame({"ts": pd.to_datetime(["2025-07-15 00:00", "2025-07-15 12:00"])})
    out = add_time_features(df)
    assert {"hour", "dow", "is_weekend", "hour_sin", "hour_cos"}.issubset(out.columns)
    # 00:00 → sin≈0, cos≈1
    assert abs(out["hour_sin"].iloc[0]) < 1e-9
    assert abs(out["hour_cos"].iloc[0] - 1) < 1e-9


def test_haversine_zero_for_same_point():
    d = haversine_m(np.array([37.5]), np.array([127.0]), np.array([37.5]), np.array([127.0]))
    assert d[0] == 0


def test_haversine_known_distance():
    # 1 degree of latitude ≈ 111 km
    d = haversine_m(np.array([37.0]), np.array([127.0]), np.array([38.0]), np.array([127.0]))
    assert 110_000 < d[0] < 112_000


def test_build_neighbor_index_finds_nearby_stations_only():
    stations = pd.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "lat": [37.5000, 37.5010, 37.6000],  # A-B ≈110m, A-C ≈11km
            "lon": [127.0, 127.0, 127.0],
        }
    )
    idx = build_neighbor_index(stations, radius_m=500)
    assert idx["A"] == ["B"]
    assert idx["B"] == ["A"]
    assert idx["C"] == []


def test_build_feature_table_delta_target_default():
    """Default target mode is 'delta': target_h(t) = bike_count(t+h) - bike_count(t)."""
    times = pd.date_range("2025-07-15 14:00", periods=120, freq="1min")
    series = pd.DataFrame(
        {
            "station_id": ["A"] * 120,
            "ts": times,
            "bike_count": np.arange(120, dtype="int64"),
        }
    )
    stations = pd.DataFrame({"station_id": ["A"], "lat": [37.5], "lon": [127.0]})

    df = build_feature_table(series, stations, weather=None)  # default delta
    a = df[df["station_id"] == "A"].sort_values("ts").reset_index(drop=True)

    # Synthetic series increases by 1 each minute, so delta should be exactly h
    assert a["target_5min"].iloc[0] == 5
    assert a["target_60min"].iloc[0] == 60
    # Same for any starting row that has h future rows
    assert a["target_5min"].iloc[10] == 5
    # Last 60 rows have NaN target_60min
    assert pd.isna(a["target_60min"].iloc[-1])


def test_build_feature_table_absolute_mode_legacy():
    """Explicit 'absolute' mode preserves the legacy behavior."""
    times = pd.date_range("2025-07-15 14:00", periods=120, freq="1min")
    series = pd.DataFrame(
        {
            "station_id": ["A"] * 120,
            "ts": times,
            "bike_count": np.arange(120, dtype="int64"),
        }
    )
    stations = pd.DataFrame({"station_id": ["A"], "lat": [37.5], "lon": [127.0]})

    df = build_feature_table(series, stations, weather=None, target_mode="absolute")
    a = df[df["station_id"] == "A"].sort_values("ts").reset_index(drop=True)
    # absolute target equals bike_count(t+h)
    assert a["target_5min"].iloc[0] == a["bike_count"].iloc[5]
    assert a["target_60min"].iloc[0] == a["bike_count"].iloc[60]


def test_delta_target_can_be_negative():
    """Delta should be negative when bike_count decreases (rentals exceed returns)."""
    times = pd.date_range("2025-07-15 14:00", periods=10, freq="1min")
    series = pd.DataFrame(
        {
            "station_id": ["A"] * 10,
            "ts": times,
            "bike_count": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    stations = pd.DataFrame({"station_id": ["A"], "lat": [37.5], "lon": [127.0]})
    df = build_feature_table(series, stations, weather=None)
    a = df[df["station_id"] == "A"].sort_values("ts").reset_index(drop=True)
    # bike_count drops by 1 every minute, so 5min delta = -5
    assert a["target_5min"].iloc[0] == -5


def test_build_feature_table_joins_weather_via_merge_asof():
    times = pd.date_range("2025-07-15 14:00", periods=10, freq="1min")
    series = pd.DataFrame(
        {"station_id": ["A"] * 10, "ts": times, "bike_count": [5] * 10}
    )
    stations = pd.DataFrame({"station_id": ["A"], "lat": [37.5], "lon": [127.0]})
    weather = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-07-15 14:00"]),
            "temp_c": [25.0],
            "precip_mm": [0.0],
            "wind_ms": [1.0],
            "humidity_pct": [50],
        }
    )
    df = build_feature_table(series, stations, weather=weather)
    assert (df["temp_c"] == 25.0).all()

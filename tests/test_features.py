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
            "capacity": [10, 10, 10],
        }
    )
    idx = build_neighbor_index(stations, radius_m=500)
    assert idx["A"] == ["B"]
    assert idx["B"] == ["A"]
    assert idx["C"] == []


def test_build_feature_table_creates_targets_and_no_label_leakage():
    times = pd.date_range("2025-07-15 14:00", periods=120, freq="1min")
    series = pd.DataFrame(
        {
            "station_id": ["A"] * 120,
            "ts": times,
            "bike_count": np.arange(120, dtype="int64"),
        }
    )
    stations = pd.DataFrame({"station_id": ["A"], "lat": [37.5], "lon": [127.0], "capacity": [200]})

    df = build_feature_table(series, stations, weather=None)

    assert "target_5min" in df.columns
    assert "target_60min" in df.columns
    # target_5min(t) should equal bike_count(t+5) — strict
    a = df[df["station_id"] == "A"].sort_values("ts").reset_index(drop=True)
    assert a["target_5min"].iloc[0] == a["bike_count"].iloc[5]
    assert a["target_60min"].iloc[0] == a["bike_count"].iloc[60]
    # Last 60 rows have NaN for target_60min — they fall off the end
    assert a["target_60min"].iloc[-1] != a["target_60min"].iloc[-1]  # NaN check


def test_build_feature_table_joins_weather_via_merge_asof():
    times = pd.date_range("2025-07-15 14:00", periods=10, freq="1min")
    series = pd.DataFrame(
        {"station_id": ["A"] * 10, "ts": times, "bike_count": [5] * 10}
    )
    stations = pd.DataFrame({"station_id": ["A"], "lat": [37.5], "lon": [127.0], "capacity": [10]})
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

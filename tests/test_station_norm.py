"""Tests for per-station normalization features."""
from __future__ import annotations

import pandas as pd

from bikeai.features.station_norm import attach_normalization, compute_station_stats


def test_compute_station_stats_basic():
    df = pd.DataFrame(
        {
            "station_id": ["A", "A", "A", "B", "B", "B"],
            "bike_count": [4, 6, 8, 20, 30, 40],
        }
    )
    stats = compute_station_stats(df).set_index("station_id")
    assert stats.loc["A", "st_mean"] == 6.0
    assert stats.loc["B", "st_mean"] == 30.0
    assert stats.loc["A", "st_max"] == 8
    assert stats.loc["B", "st_max"] == 40


def test_compute_station_stats_avoids_zero_std():
    """Stations with constant bike_count should still get a non-zero std (clip)."""
    df = pd.DataFrame({"station_id": ["A"] * 5, "bike_count": [3, 3, 3, 3, 3]})
    stats = compute_station_stats(df).set_index("station_id")
    assert stats.loc["A", "st_std"] >= 0.5


def test_attach_normalization_adds_expected_columns():
    df = pd.DataFrame(
        {
            "station_id": ["A", "A", "B", "B"],
            "ts": pd.to_datetime(
                ["2026-04-11 14:00", "2026-04-11 14:01", "2026-04-11 14:00", "2026-04-11 14:01"]
            ),
            "bike_count": [4, 8, 20, 40],
        }
    )
    out, stats = attach_normalization(df)
    for col in ("st_mean", "st_std", "st_max", "bike_count_centered", "bike_count_z", "bike_count_pct"):
        assert col in out.columns

    a = out[out["station_id"] == "A"]
    # mean of A = 6, so first row centered = -2
    assert a["bike_count_centered"].iloc[0] == -2
    # pct of first A row = 4/8 = 0.5
    assert a["bike_count_pct"].iloc[0] == 0.5

    b = out[out["station_id"] == "B"]
    # mean of B = 30, so first row centered = -10
    assert b["bike_count_centered"].iloc[0] == -10


def test_attach_normalization_uses_provided_stats():
    """When stats are provided externally (e.g., from training set), use them."""
    train_df = pd.DataFrame({"station_id": ["A", "A"], "bike_count": [10, 20]})
    train_stats = compute_station_stats(train_df)

    test_df = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "ts": pd.to_datetime(["2026-04-11 14:00", "2026-04-11 14:01"]),
            "bike_count": [25, 30],
        }
    )
    out, used = attach_normalization(test_df, stats=train_stats)
    # st_mean should come from train (15), not from test (27.5)
    assert out["st_mean"].iloc[0] == 15
    assert out["bike_count_centered"].iloc[0] == 10

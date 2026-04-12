"""Unit tests for the rental-event → bike-count time series reconstruction.

These are the most important tests in the project: if reconstruction is wrong, every
downstream model is wrong. We use small synthetic event sets where the ground truth
is computable by hand.
"""
from __future__ import annotations

import pandas as pd
import pytest

from bikeai.features.reconstruct import (
    apply_anchors,
    cumulative_relative_count,
    events_from_rentals,
    reconstruct,
    resample_to_grid,
)


def _rentals(rows):
    return pd.DataFrame(
        rows,
        columns=["rent_dt", "return_dt", "rent_station", "return_station"],
    ).assign(
        rent_dt=lambda d: pd.to_datetime(d["rent_dt"]),
        return_dt=lambda d: pd.to_datetime(d["return_dt"]),
    )


def test_events_from_rentals_explodes_into_two_deltas():
    rentals = _rentals(
        [
            ("2025-07-15 14:00", "2025-07-15 14:20", "A", "B"),
            ("2025-07-15 14:05", "2025-07-15 14:35", "A", "C"),
        ]
    )
    events = events_from_rentals(rentals)
    assert len(events) == 4
    a_events = events[events["station_id"] == "A"]
    assert (a_events["delta"] == -1).all()
    b_events = events[events["station_id"] == "B"]
    assert (b_events["delta"] == 1).all()


def test_cumulative_relative_count_handles_independent_stations():
    rentals = _rentals(
        [
            ("2025-07-15 14:00", "2025-07-15 14:20", "A", "B"),
            ("2025-07-15 14:10", "2025-07-15 14:30", "A", "B"),
        ]
    )
    events = events_from_rentals(rentals)
    rel = cumulative_relative_count(events)
    a = rel[rel["station_id"] == "A"]["rel_count"].tolist()
    b = rel[rel["station_id"] == "B"]["rel_count"].tolist()
    assert a == [-1, -2]
    assert b == [1, 2]


def test_apply_anchors_shifts_series_to_match_known_truth():
    """Anchor says A had 10 bikes at 14:00 (before any rents). Final count = 10 - 2 = 8."""
    rentals = _rentals(
        [
            ("2025-07-15 14:01", "2025-07-15 14:20", "A", "B"),
            ("2025-07-15 14:02", "2025-07-15 14:30", "A", "C"),
        ]
    )
    events = events_from_rentals(rentals)
    rel = cumulative_relative_count(events)

    anchors = pd.DataFrame(
        {
            "station_id": ["A"],
            "ts": pd.to_datetime(["2025-07-15 14:00"]),
            "bike_count": [10],
        }
    )
    abs_series = apply_anchors(rel, anchors, method="first")
    a = abs_series[abs_series["station_id"] == "A"]
    assert a.iloc[-1]["bike_count"] == 8  # 10 - 2 rents


def test_apply_anchors_clips_negative_to_zero():
    """If reconstruction drifts below zero, it's clipped — bikes can't be negative."""
    rentals = _rentals([("2025-07-15 14:01", "2025-07-15 14:20", "A", "B")])
    events = events_from_rentals(rentals)
    rel = cumulative_relative_count(events)
    anchors = pd.DataFrame(
        {"station_id": ["A"], "ts": pd.to_datetime(["2025-07-15 14:00"]), "bike_count": [0]}
    )
    abs_series = apply_anchors(rel, anchors)
    assert (abs_series["bike_count"] >= 0).all()


def test_resample_to_grid_forward_fills_between_events():
    rel = pd.DataFrame(
        {
            "station_id": ["A", "A", "A"],
            "ts": pd.to_datetime(
                ["2025-07-15 14:00", "2025-07-15 14:03", "2025-07-15 14:05"]
            ),
            "bike_count": [10, 9, 11],
        }
    )
    grid = resample_to_grid(rel, freq="1min")
    a = grid[grid["station_id"] == "A"].set_index("ts")["bike_count"]
    assert a.loc["2025-07-15 14:00"] == 10
    assert a.loc["2025-07-15 14:01"] == 10  # ffill
    assert a.loc["2025-07-15 14:02"] == 10
    assert a.loc["2025-07-15 14:03"] == 9
    assert a.loc["2025-07-15 14:04"] == 9
    assert a.loc["2025-07-15 14:05"] == 11


def test_resample_to_grid_does_not_clip_upper_bound():
    """Ttareungi has no capacity concept, so resample_to_grid never clips above."""
    rel = pd.DataFrame(
        {
            "station_id": ["A", "A"],
            "ts": pd.to_datetime(["2025-07-15 14:00", "2025-07-15 14:01"]),
            "bike_count": [10, 99],
        }
    )
    grid = resample_to_grid(rel, freq="1min")
    assert grid["bike_count"].max() == 99


def test_reconstruct_end_to_end_balances_rents_and_returns():
    """Bikes that leave A and return to B should make A go down and B go up by the same amount."""
    rentals = _rentals(
        [
            ("2025-07-15 14:00", "2025-07-15 14:10", "A", "B"),
            ("2025-07-15 14:01", "2025-07-15 14:11", "A", "B"),
            ("2025-07-15 14:02", "2025-07-15 14:12", "A", "B"),
        ]
    )
    anchors = pd.DataFrame(
        {
            "station_id": ["A", "B"],
            "ts": pd.to_datetime(["2025-07-15 13:59", "2025-07-15 13:59"]),
            "bike_count": [10, 5],
        }
    )
    result = reconstruct(rentals, anchors=anchors, freq="1min")  # no capacity arg
    final = (
        result.series.sort_values(["station_id", "ts"])
        .groupby("station_id")
        .tail(1)
        .set_index("station_id")["bike_count"]
    )
    assert final["A"] == 7
    assert final["B"] == 8
    assert result.stations == 2


def test_reconstruct_handles_empty_input():
    empty = pd.DataFrame(columns=["rent_dt", "return_dt", "rent_station", "return_station"])
    result = reconstruct(empty)
    assert result.series.empty
    assert result.stations == 0

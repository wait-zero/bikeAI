"""Tests for the collected-snapshots loader."""
from __future__ import annotations

import pandas as pd

from bikeai.ingest.snapshots import load_collected_snapshots, snapshot_horizons


def test_load_collected_snapshots_canonicalizes(tmp_path):
    p = tmp_path / "train_full.csv"
    pd.DataFrame(
        {
            "ts": [1775882340, 1775882400],
            "time_local": ["2026-04-11 13:39:00", "2026-04-11 13:40:00"],
            "station_id": ["ST-10", "ST-10"],
            "name": ["108. 서교", "108. 서교"],
            "lat": [37.55, 37.55],
            "lon": [126.92, 126.92],
            "rack_total": [None, None],
            "bike_count": [10, 11],
        }
    ).to_csv(p, index=False)

    df = load_collected_snapshots(p)
    assert list(df.columns) == ["station_id", "ts", "bike_count", "lat", "lon"]
    assert df["bike_count"].dtype == "int64"
    assert df["ts"].iloc[0] == pd.Timestamp("2026-04-11 13:39:00")
    assert df.iloc[0]["station_id"] == "ST-10"


def test_snapshot_horizons_builds_delta_labels():
    """Given 6 consecutive 1-min snapshots, snapshot_horizons should produce
    target_5min rows with the correct future absolute and delta."""
    times = pd.date_range("2026-04-11 14:00", periods=10, freq="1min")
    snaps = pd.DataFrame(
        {
            "station_id": ["ST-1"] * 10,
            "ts": times,
            "bike_count": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "lat": [37.5] * 10,
            "lon": [127.0] * 10,
        }
    )
    out = snapshot_horizons(snaps, horizons_min=(5,))
    # Rows where t+5 is present: indices 0..4
    assert len(out) == 5
    assert out.iloc[0]["bike_count"] == 10
    assert out.iloc[0]["target_5min_abs"] == 15
    assert out.iloc[0]["target_5min_delta"] == 5


def test_snapshot_horizons_drops_rows_without_full_horizon_set():
    """If only target_5 is present but not target_15, the row should be dropped."""
    times = pd.date_range("2026-04-11 14:00", periods=8, freq="1min")
    snaps = pd.DataFrame(
        {
            "station_id": ["ST-1"] * 8,
            "ts": times,
            "bike_count": list(range(8)),
            "lat": [37.5] * 8,
            "lon": [127.0] * 8,
        }
    )
    out = snapshot_horizons(snaps, horizons_min=(5, 15))
    # Need t, t+5, t+15 → only enough room if last index >= 15 ahead. With 8 rows
    # and 1-min spacing, no row has 15 future minutes available.
    assert len(out) == 0

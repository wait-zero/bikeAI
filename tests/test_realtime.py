"""Tests for the realtime API client.

Network calls are mocked — these only verify the parsing/normalization layer.
"""
from __future__ import annotations

import pandas as pd

from bikeai.ingest.realtime import _extract_items, write_snapshot


def test_extract_items_handles_list_shape():
    payload = {
        "response": {
            "body": {
                "items": {
                    "item": [
                        {"stationId": "ST-1", "parkingBikeTotCnt": "3"},
                        {"stationId": "ST-2", "parkingBikeTotCnt": "0"},
                    ]
                }
            }
        }
    }
    items = _extract_items(payload)
    assert len(items) == 2
    assert items[0]["stationId"] == "ST-1"


def test_extract_items_handles_single_dict_shape():
    payload = {
        "response": {
            "body": {"items": {"item": {"stationId": "ST-9", "parkingBikeTotCnt": "5"}}}
        }
    }
    items = _extract_items(payload)
    assert items == [{"stationId": "ST-9", "parkingBikeTotCnt": "5"}]


def test_extract_items_empty_payload():
    assert _extract_items({}) == []
    assert _extract_items({"response": {"body": {}}}) == []


def test_write_snapshot_partitions_by_date(tmp_path):
    df = pd.DataFrame(
        {
            "station_id": ["ST-1"],
            "bike_count": [3],
            "ingested_at": [pd.Timestamp("2025-07-15 14:23:00", tz="Asia/Seoul")],
        }
    )
    out = write_snapshot(df, root=tmp_path)
    assert out.exists()
    assert "date=2025-07-15" in str(out)
    roundtrip = pd.read_parquet(out)
    assert roundtrip.iloc[0]["station_id"] == "ST-1"

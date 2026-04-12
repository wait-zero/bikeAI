"""Tests for the realtime API client.

Network calls are mocked — these only verify the parsing/normalization layer.
"""
from __future__ import annotations

import pandas as pd

from unittest.mock import patch

from bikeai.ingest.realtime import _extract_items, fetch_snapshot, write_snapshot


def test_extract_items_handles_list_shape():
    payload = {
        "response": {
            "body": {
                "items": {
                    "item": [
                        {"rntstnId": "ST-1", "bcyclTpkctNocs": "3", "lat": "37.5", "lot": "127.0"},
                        {"rntstnId": "ST-2", "bcyclTpkctNocs": "0", "lat": "37.5", "lot": "127.0"},
                    ]
                }
            }
        }
    }
    items = _extract_items(payload)
    assert len(items) == 2
    assert items[0]["rntstnId"] == "ST-1"


def test_extract_items_handles_single_dict_shape():
    payload = {
        "response": {
            "body": {"items": {"item": {"rntstnId": "ST-9", "bcyclTpkctNocs": "5"}}}
        }
    }
    items = _extract_items(payload)
    assert items == [{"rntstnId": "ST-9", "bcyclTpkctNocs": "5"}]


def test_extract_items_empty_payload():
    assert _extract_items({}) == []
    assert _extract_items({"response": {"body": {}}}) == []


def test_fetch_snapshot_normalizes_to_canonical_columns(monkeypatch):
    """End-to-end normalization: API field names → canonical schema with right types."""
    monkeypatch.setenv("PUBLIC_BIKE_API_KEY", "dummy")

    fake_payload = {
        "response": {
            "body": {
                "items": {
                    "item": [
                        {
                            "rntstnId": "ST-100",
                            "rntstnNm": "테스트역",
                            "lat": "37.5527",
                            "lot": "126.9186",
                            "bcyclTpkctNocs": "12",
                            "lcgvmnInstCd": "1100000000",
                            "lcgvmnInstNm": "서울특별시",
                        }
                    ]
                }
            }
        }
    }

    with patch("bikeai.ingest.realtime._fetch_page", return_value=fake_payload):
        df = fetch_snapshot()

    assert {"station_id", "station_name", "lat", "lon", "bike_count", "region_code"}.issubset(
        df.columns
    )
    assert df.iloc[0]["station_id"] == "ST-100"
    assert df.iloc[0]["bike_count"] == 12
    assert df["bike_count"].dtype == "int64"
    assert abs(df.iloc[0]["lat"] - 37.5527) < 1e-6
    assert abs(df.iloc[0]["lon"] - 126.9186) < 1e-6
    assert "ingested_at" in df.columns


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

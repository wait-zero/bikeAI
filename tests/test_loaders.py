"""Tests for the static-data loaders. We write small synthetic CSVs and read them back."""
from __future__ import annotations

import pandas as pd

from bikeai.ingest.rental_history import load_rental_csv
from bikeai.ingest.stations import load_station_master
from bikeai.ingest.weather import load_asos_csv


def test_station_master_canonicalizes_legacy_korean_headers(tmp_path):
    p = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "대여소번호": ["ST-1", "ST-2"],
            "대여소명": ["가A", "나B"],
            "위도": [37.5, 37.51],
            "경도": [127.0, 127.01],
        }
    ).to_csv(p, index=False, encoding="utf-8-sig")

    df = load_station_master(p)
    assert list(df.columns) == ["station_id", "station_name", "lat", "lon"]
    assert df["station_id"].tolist() == ["ST-1", "ST-2"]


def test_station_master_handles_new_address_format(tmp_path):
    """The 2024+ Seoul master file has 대여소_ID + 주소1/주소2 instead of 대여소명."""
    p = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "대여소_ID": ["ST-100", "ST-200", "ST-300"],
            "주소1": ["서울특별시 마포구", "서울특별시 종로구", "서울특별시 강남구"],
            "주소2": ["홍대입구", "광화문", ""],
            "위도": [37.55, 37.57, 0.0],  # last row has bad coords
            "경도": [126.92, 126.97, 0.0],
        }
    ).to_csv(p, index=False, encoding="utf-8-sig")

    df = load_station_master(p)
    # row 3 dropped due to lat=0/lon=0 sentinel
    assert len(df) == 2
    assert df["station_id"].tolist() == ["ST-100", "ST-200"]
    assert "마포구" in df["station_name"].iloc[0]
    assert "종로구" in df["station_name"].iloc[1]


def test_rental_history_handles_korean_headers(tmp_path):
    p = tmp_path / "rents.csv"
    pd.DataFrame(
        {
            "자전거번호": ["B1", "B2"],
            "대여일시": ["2025-07-15 14:00:00", "2025-07-15 14:05:00"],
            "대여대여소번호": ["ST-1", "ST-2"],
            "반납일시": ["2025-07-15 14:20:00", "2025-07-15 14:30:00"],
            "반납대여소번호": ["ST-2", "ST-1"],
            "이용시간(분)": [20, 25],
            "이용거리(M)": [3000, 4200],
        }
    ).to_csv(p, index=False, encoding="utf-8-sig")

    df = load_rental_csv(p)
    assert {"rent_dt", "return_dt", "rent_station", "return_station"}.issubset(df.columns)
    assert len(df) == 2
    assert df["duration_sec"].iloc[0] == 20 * 60
    assert df["rent_dt"].iloc[0] == pd.Timestamp("2025-07-15 14:00:00")


def test_asos_loader_minute_format_converts_cumulative_precip(tmp_path):
    """Minute (MI) ASOS files give cumulative precip; loader should diff to per-minute."""
    p = tmp_path / "asos_min.csv"
    pd.DataFrame(
        {
            "지점": [108] * 5,
            "일시": [
                "2026-04-11 14:00",
                "2026-04-11 14:01",
                "2026-04-11 14:02",
                "2026-04-11 14:03",
                "2026-04-11 14:04",
            ],
            "기온(°C)": [12.0, 12.1, 12.1, 12.2, 12.2],
            "누적강수량(mm)": [0.0, 0.1, 0.3, 0.3, 0.4],  # cumulative
            "풍향(deg)": [45.0, 50.0, 60.0, 55.0, 50.0],
            "습도(%)": [55, 56, 57, 56, 55],
        }
    ).to_csv(p, index=False, encoding="utf-8-sig")

    import pytest
    df = load_asos_csv(p)
    assert list(df.columns) == ["ts", "temp_c", "precip_mm", "wind_ms", "humidity_pct"]
    # First row diff is NaN → 0; subsequent: 0.1, 0.2, 0.0, 0.1
    assert df["precip_mm"].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.0, 0.1])
    # wind_ms is missing in minute format → should be NA
    assert df["wind_ms"].isna().all()


def test_asos_loader_parses_korean_headers(tmp_path):
    p = tmp_path / "asos.csv"
    pd.DataFrame(
        {
            "지점": [108, 108],
            "일시": ["2025-07-15 14:00", "2025-07-15 15:00"],
            "기온(°C)": [31.2, 30.8],
            "강수량(mm)": [None, 1.5],
            "풍속(m/s)": [2.1, 2.3],
            "습도(%)": [55, 60],
        }
    ).to_csv(p, index=False, encoding="utf-8-sig")

    df = load_asos_csv(p)
    assert df["temp_c"].tolist() == [31.2, 30.8]
    assert df["precip_mm"].iloc[0] == 0  # NaN -> 0
    assert df["precip_mm"].iloc[1] == 1.5

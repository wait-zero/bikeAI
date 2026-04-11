"""Tests for the static-data loaders. We write small synthetic CSVs and read them back."""
from __future__ import annotations

import pandas as pd

from bikeai.ingest.rental_history import load_rental_csv
from bikeai.ingest.stations import load_station_master
from bikeai.ingest.weather import load_asos_csv


def test_station_master_canonicalizes_korean_headers(tmp_path):
    p = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "대여소번호": ["ST-1", "ST-2"],
            "대여소명": ["가A", "나B"],
            "위도": [37.5, 37.51],
            "경도": [127.0, 127.01],
            "거치대수": [10, 20],
        }
    ).to_csv(p, index=False, encoding="utf-8-sig")

    df = load_station_master(p)
    assert list(df.columns) == ["station_id", "station_name", "lat", "lon", "capacity"]
    assert df["capacity"].tolist() == [10, 20]


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

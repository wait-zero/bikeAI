"""Loader for the Ttareungi station master CSV (data.go.kr 15099365).

The master file lists every station with its ID, name, lat/lon, and rack capacity.
The schema has changed slightly across years; we accept both Korean and English headers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["station_id", "station_name", "lat", "lon", "capacity"]

# Map of common header variants -> canonical name. Add as needed when new files appear.
_HEADER_ALIASES = {
    "대여소번호": "station_id",
    "대여소ID": "station_id",
    "stationId": "station_id",
    "대여소명": "station_name",
    "stationName": "station_name",
    "위도": "lat",
    "stationLatitude": "lat",
    "경도": "lon",
    "stationLongitude": "lon",
    "거치대수": "capacity",
    "거치대수LCD": "capacity",
    "거치대수QR": "capacity",
    "rackTotCnt": "capacity",
}


def load_station_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=_detect_encoding(path))
    df = df.rename(columns={c: _HEADER_ALIASES.get(c.strip(), c) for c in df.columns})
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Station master {path} missing columns {missing}. Got {list(df.columns)}"
        )
    df = df[CANONICAL_COLUMNS].copy()
    df["station_id"] = df["station_id"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def _detect_encoding(path: Path) -> str:
    """Seoul open-data CSVs are usually CP949; newer ones are UTF-8 with BOM."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        path.open(encoding="utf-8").read(2048)
        return "utf-8"
    except UnicodeDecodeError:
        return "cp949"

"""Loader for the Ttareungi station master CSV.

Seoul Ttareungi has no concept of station capacity, so this loader only requires
station id + coordinates. The expected canonical schema is:
    station_id, station_name, lat, lon

Multiple header conventions are supported because Seoul publishes the master in
several formats over the years.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["station_id", "station_name", "lat", "lon"]

# Map of header variants -> canonical name. Add as needed when new formats appear.
_HEADER_ALIASES = {
    # 2024+ format published as data.seoul.go.kr
    "대여소_ID": "station_id",
    "주소1": "address1",
    "주소2": "address2",
    # Legacy formats
    "대여소번호": "station_id",
    "대여소ID": "station_id",
    "stationId": "station_id",
    "대여소명": "station_name",
    "stationName": "station_name",
    "위도": "lat",
    "stationLatitude": "lat",
    "경도": "lon",
    "stationLongitude": "lon",
}


def load_station_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=_detect_encoding(path))
    df = df.rename(columns={c: _HEADER_ALIASES.get(c.strip(), c) for c in df.columns})

    # Compose station_name from address columns when no explicit name is present
    if "station_name" not in df.columns:
        if "address1" in df.columns and "address2" in df.columns:
            df["station_name"] = (
                df["address1"].fillna("").astype(str).str.strip()
                + " "
                + df["address2"].fillna("").astype(str).str.strip()
            ).str.strip()
        elif "address1" in df.columns:
            df["station_name"] = df["address1"].astype(str).str.strip()
        else:
            df["station_name"] = ""

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Station master {path} missing columns {missing}. Got {list(df.columns)}"
        )

    df = df[CANONICAL_COLUMNS].copy()
    df["station_id"] = df["station_id"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Drop rows with missing or zero coordinates (the master file contains some
    # stations recorded as 0.000000 / 0.000000).
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"] != 0) & (df["lon"] != 0)]
    return df.reset_index(drop=True)


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

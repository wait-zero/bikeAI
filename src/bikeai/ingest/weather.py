"""Loader for KMA ASOS hourly weather CSV (data.kma.go.kr).

Expected source: 종관기상관측(ASOS) 시간자료 — Seoul station 108.
Default download is CP949 with Korean headers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["ts", "temp_c", "precip_mm", "wind_ms", "humidity_pct"]

_RENAME = {
    "지점": "station_id",
    "지점명": "station_name",
    "일시": "ts",
    "기온(°C)": "temp_c",
    "기온": "temp_c",
    "강수량(mm)": "precip_mm",
    "강수량": "precip_mm",
    "풍속(m/s)": "wind_ms",
    "풍속": "wind_ms",
    "습도(%)": "humidity_pct",
    "습도": "humidity_pct",
}


def load_asos_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=_detect_encoding(path), low_memory=False)
    df = df.rename(columns={c: _RENAME.get(c.strip(), c) for c in df.columns})

    if "ts" not in df.columns:
        raise ValueError(f"ASOS file {path} missing time column. Got {list(df.columns)}")

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    for col in ("temp_c", "precip_mm", "wind_ms", "humidity_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.NA

    df["precip_mm"] = df["precip_mm"].fillna(0)

    return (
        df[CANONICAL_COLUMNS]
        .dropna(subset=["ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )


def _detect_encoding(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(4)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        path.open(encoding="utf-8").read(4096)
        return "utf-8"
    except UnicodeDecodeError:
        return "cp949"

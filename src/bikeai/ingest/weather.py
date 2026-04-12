"""Loader for KMA ASOS weather CSVs (data.kma.go.kr).

Two formats are supported:

1. **Hourly (TIM)** — 종관기상관측(ASOS) 시간자료
   columns: 일시, 기온(°C), 강수량(mm), 풍속(m/s), 습도(%)

2. **Minute (MI)** — 종관기상관측 분자료
   columns: 일시, 기온(°C), 누적강수량(mm), 풍향(deg), 습도(%)
   • 강수량은 누적이므로 station-wise diff 로 분당값 계산
   • 풍속(m/s) 컬럼이 없어 wind_ms는 NaN으로 채움

Both are CP949 by default. The loader auto-detects format by inspecting headers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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
    "누적강수량(mm)": "precip_cum_mm",
    "누적강수량": "precip_cum_mm",
    "풍속(m/s)": "wind_ms",
    "풍속": "wind_ms",
    "풍향(deg)": "wind_dir_deg",
    "풍향": "wind_dir_deg",
    "습도(%)": "humidity_pct",
    "습도": "humidity_pct",
}


def load_asos_csv(path: Path, station_id: int | str = 108) -> pd.DataFrame:
    """Load either hourly or minute ASOS CSV and return the canonical schema.

    Some ASOS exports include multiple stations interleaved (e.g. 서울 108 +
    관악산 116). We filter to a single station first so that any per-station
    cumulative calculations work correctly.
    """
    df = pd.read_csv(path, encoding=_detect_encoding(path), low_memory=False)
    df = df.rename(columns={c: _RENAME.get(c.strip(), c) for c in df.columns})

    if "ts" not in df.columns:
        raise ValueError(f"ASOS file {path} missing time column. Got {list(df.columns)}")

    if "station_id" in df.columns:
        df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")
        df = df[df["station_id"] == int(station_id)].copy()

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    for col in ("temp_c", "precip_mm", "precip_cum_mm", "wind_ms", "humidity_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Minute-format files report cumulative precipitation that resets at midnight.
    # Convert to per-minute rainfall by differencing within each calendar day
    # (sorted ascending). Negative diffs (midnight reset) become 0.
    if "precip_cum_mm" in df.columns and "precip_mm" not in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
        date = df["ts"].dt.date
        diff = df.groupby(date)["precip_cum_mm"].diff()
        # First row of each day: take the cum value as-is (start of accumulation).
        first_of_day = diff.isna()
        df["precip_mm"] = diff.where(diff >= 0, 0).fillna(0)
        df.loc[first_of_day, "precip_mm"] = df.loc[first_of_day, "precip_cum_mm"].fillna(0)

    # Ensure all canonical columns exist with numeric dtype (LightGBM rejects object).
    for col in CANONICAL_COLUMNS:
        if col == "ts":
            continue
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

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

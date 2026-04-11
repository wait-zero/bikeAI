"""Loader for Seoul Ttareungi rental history CSVs (data.seoul.go.kr OA-15182).

Each monthly file contains one row per completed trip with start/end station and time.
We standardize to: rent_dt, return_dt, rent_station, return_station, bike_id, distance_m,
duration_sec.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

CANONICAL_COLUMNS = [
    "rent_dt",
    "return_dt",
    "rent_station",
    "return_station",
    "bike_id",
    "duration_sec",
    "distance_m",
]

# Header variants. Names differ across years (한글 / English / spacing).
_RENAME = {
    "자전거번호": "bike_id",
    "대여일시": "rent_dt",
    "대여 일시": "rent_dt",
    "대여시간": "rent_dt",
    "대여대여소번호": "rent_station",
    "대여 대여소번호": "rent_station",
    "대여스테이션번호": "rent_station",
    "반납일시": "return_dt",
    "반납 일시": "return_dt",
    "반납시간": "return_dt",
    "반납대여소번호": "return_station",
    "반납 대여소번호": "return_station",
    "이용시간": "duration_sec",
    "이용시간(분)": "duration_min",
    "사용시간": "duration_sec",
    "이용거리": "distance_m",
    "이용거리(M)": "distance_m",
}


def load_rental_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=_detect_encoding(path), low_memory=False)
    df = df.rename(columns={c: _RENAME.get(c.strip(), c) for c in df.columns})

    if "duration_sec" not in df.columns and "duration_min" in df.columns:
        df["duration_sec"] = pd.to_numeric(df["duration_min"], errors="coerce") * 60

    for col in ("rent_dt", "return_dt"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ("rent_station", "return_station", "bike_id"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for col in ("duration_sec", "distance_m"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["rent_dt", "return_dt", "rent_station", "return_station"])
    return df.reset_index(drop=True)


def load_rental_history(paths: Iterable[Path]) -> pd.DataFrame:
    """Load and concat multiple monthly files."""
    frames = [load_rental_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


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

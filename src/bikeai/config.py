"""Project paths and shared configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

HORIZONS_MIN = (5, 15, 30, 60)
RESAMPLE_FREQ = "1min"

SEOUL_ASOS_STATION_ID = 108

# Local-government code used by the realtime API to filter by city.
# Seoul Metropolitan City = 1100000000.
SEOUL_LCGVMN_CD = "1100000000"


@dataclass(frozen=True)
class RealtimeApiConfig:
    """data.go.kr 15126639 — National public bicycle realtime info.

    Endpoint 2 (`/inf_101_00010002_v2`) returns the current parked bike count per
    station for a given city (lcgvmnInstCd is required).
    """
    base_url: str = "http://apis.data.go.kr/B551982/pbdo_v2/inf_101_00010002_v2"
    service_key_env: str = "PUBLIC_BIKE_API_KEY"
    region_code: str = SEOUL_LCGVMN_CD
    page_size: int = 1000
    timeout_sec: float = 15.0

    def service_key(self) -> str:
        key = os.environ.get(self.service_key_env)
        if not key:
            raise RuntimeError(
                f"Set {self.service_key_env} env var (data.go.kr 15126639 service key)."
            )
        return key

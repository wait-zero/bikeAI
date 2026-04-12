"""Realtime public bicycle status API client (data.go.kr 15126639).

Pulls a full snapshot of all stations and appends it as a parquet file partitioned
by ingest timestamp. Designed to be invoked from cron every 1 minute.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from bikeai.config import RAW_DIR, RealtimeApiConfig

log = logging.getLogger(__name__)

SNAPSHOT_DIR = RAW_DIR / "realtime_snapshots"

# API field names from data.go.kr 15126639 endpoint 2 (`/inf_101_00010002_v2`).
# Note: the API spells longitude as "lot" — this is not a typo on our side.
# Capacity is intentionally absent here; it must be joined from the static
# Ttareungi master CSV (data.go.kr 15099365).
CANONICAL_RENAME = {
    "rntstnId": "station_id",
    "rntstnNm": "station_name",
    "lat": "lat",
    "lot": "lon",
    "bcyclTpkctNocs": "bike_count",
    "lcgvmnInstCd": "region_code",
    "lcgvmnInstNm": "region_name",
}


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
def _fetch_page(client: httpx.Client, cfg: RealtimeApiConfig, page_no: int) -> dict:
    params = {
        "serviceKey": cfg.service_key(),
        "pageNo": page_no,
        "numOfRows": cfg.page_size,
        "type": "json",
        "lcgvmnInstCd": cfg.region_code,
    }
    r = client.get(cfg.base_url, params=params, timeout=cfg.timeout_sec)
    r.raise_for_status()
    return r.json()


def fetch_snapshot(cfg: RealtimeApiConfig | None = None) -> pd.DataFrame:
    """Pull every page and return a single normalized DataFrame."""
    cfg = cfg or RealtimeApiConfig()
    rows: list[dict] = []
    with httpx.Client() as client:
        page = 1
        while True:
            payload = _fetch_page(client, cfg, page)
            items = _extract_items(payload)
            if not items:
                break
            rows.extend(items)
            if len(items) < cfg.page_size:
                break
            page += 1

    df = pd.DataFrame(rows)
    df = df.rename(columns=CANONICAL_RENAME)
    for col in ("lat", "lon"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "bike_count" in df.columns:
        df["bike_count"] = (
            pd.to_numeric(df["bike_count"], errors="coerce").fillna(0).astype("int64")
        )
    df["ingested_at"] = pd.Timestamp.now(tz="Asia/Seoul").floor("s")
    return df


def _extract_items(payload: dict) -> list[dict]:
    """API wraps items as response.body.items.item — defend against shape variants."""
    try:
        body = payload["response"]["body"]
    except (KeyError, TypeError):
        return []
    items = body.get("items") or body.get("item")
    if items is None:
        return []
    if isinstance(items, dict):
        items = items.get("item", [])
    if isinstance(items, dict):
        items = [items]
    return list(items)


def write_snapshot(df: pd.DataFrame, root: Path = SNAPSHOT_DIR) -> Path:
    ts = df["ingested_at"].iloc[0]
    out = root / f"date={ts:%Y-%m-%d}" / f"snapshot_{ts:%Y%m%dT%H%M%S}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime public bike snapshot fetcher")
    parser.add_argument("--once", action="store_true", help="Fetch a single snapshot and exit")
    parser.add_argument("--out", type=Path, default=SNAPSHOT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = fetch_snapshot()
    path = write_snapshot(df, args.out)
    log.info("wrote %d stations -> %s", len(df), path)


if __name__ == "__main__":
    main()

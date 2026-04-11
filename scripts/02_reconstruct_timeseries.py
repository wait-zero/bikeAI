"""Read raw rental history + station master + realtime snapshots, reconstruct
station × 1-min bike-count series, and write to data/interim/series.parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bikeai.config import INTERIM_DIR, RAW_DIR
from bikeai.features.reconstruct import reconstruct
from bikeai.ingest.rental_history import load_rental_history
from bikeai.ingest.stations import load_station_master


def _collect_rental_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.csv"))


def _load_anchors_from_snapshots(root: Path) -> pd.DataFrame:
    """Build anchors DataFrame [station_id, ts, bike_count] from realtime parquet shards."""
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["station_id", "ts", "bike_count"])
    frames = []
    for f in files:
        s = pd.read_parquet(f, columns=["station_id", "bike_count", "ingested_at"])
        s = s.rename(columns={"ingested_at": "ts"})
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rental-root", type=Path, default=RAW_DIR / "rental_history")
    parser.add_argument("--stations", type=Path, default=RAW_DIR / "stations.csv")
    parser.add_argument("--snapshots", type=Path, default=RAW_DIR / "realtime_snapshots")
    parser.add_argument("--out", type=Path, default=INTERIM_DIR / "series.parquet")
    args = parser.parse_args()

    print("loading station master...")
    stations = load_station_master(args.stations)

    print("loading rental history...")
    csvs = _collect_rental_csvs(args.rental_root)
    if not csvs:
        raise SystemExit(f"no rental CSVs found under {args.rental_root}")
    rentals = load_rental_history(csvs)
    print(f"  {len(rentals):,} rental events from {len(csvs)} files")

    print("loading anchors from realtime snapshots...")
    anchors = _load_anchors_from_snapshots(args.snapshots)
    print(f"  {len(anchors):,} anchor points")

    capacity = stations[["station_id", "capacity"]]
    print("reconstructing 1-min series...")
    result = reconstruct(rentals, anchors=anchors, capacity=capacity)
    print(f"  {result.stations} stations, {len(result.series):,} rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.series.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

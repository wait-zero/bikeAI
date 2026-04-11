"""Read interim/series.parquet + station master + ASOS, build the feature/target table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bikeai.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from bikeai.features.build_dataset import build_feature_table
from bikeai.ingest.stations import load_station_master
from bikeai.ingest.weather import load_asos_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=Path, default=INTERIM_DIR / "series.parquet")
    parser.add_argument("--stations", type=Path, default=RAW_DIR / "stations.csv")
    parser.add_argument("--weather", type=Path, default=RAW_DIR / "asos_seoul.csv")
    parser.add_argument("--no-lags", action="store_true", help="Skip lag/rolling features (TFT input)")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "feature_table.parquet")
    args = parser.parse_args()

    print(f"loading {args.series}")
    series = pd.read_parquet(args.series)
    stations = load_station_master(args.stations)
    weather = load_asos_csv(args.weather) if args.weather.exists() else None
    if weather is None:
        print(f"WARNING: {args.weather} not found — building features without weather")

    print("building feature table...")
    df = build_feature_table(series, stations, weather=weather, add_lags=not args.no_lags)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

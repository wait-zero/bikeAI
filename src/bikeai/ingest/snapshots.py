"""Loader for collected realtime snapshots saved as a flat CSV.

This is the format produced by the user's earlier collection runs (`train_full.csv`):
    ts (unix int), time_local (str), station_id (ST-prefixed), name, lat, lon,
    rack_total (often NaN), bike_count (int)

We canonicalize to:
    station_id, ts (datetime64), bike_count (int64), lat, lon
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["station_id", "ts", "bike_count", "lat", "lon"]


def load_collected_snapshots(path: Path) -> pd.DataFrame:
    """Read a flat snapshots CSV and return a tidy DataFrame.

    The file is expected to be UTF-8 (the user's collector wrote it that way).
    Garbled `name` bytes are tolerated by replacing undecodable characters.
    """
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")

    if "time_local" in df.columns:
        df["ts"] = pd.to_datetime(df["time_local"], errors="coerce")
    elif "ts" in df.columns and pd.api.types.is_integer_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], unit="s")
    else:
        raise ValueError(f"snapshot file {path} has no recognized time column")

    for col in ("station_id",):
        df[col] = df[col].astype(str).str.strip()
    df["bike_count"] = pd.to_numeric(df["bike_count"], errors="coerce").fillna(0).astype("int64")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(subset=["ts", "station_id"])
    df = df[CANONICAL_COLUMNS].sort_values(["station_id", "ts"]).reset_index(drop=True)
    return df


def snapshot_horizons(
    snapshots: pd.DataFrame,
    horizons_min: tuple[int, ...],
) -> pd.DataFrame:
    """Build a (station, t, target_h) table from a snapshot time series.

    For every (station, ts) row in the snapshots, look up bike_count at ts + h
    minutes (when present) and emit `target_{h}min_delta` and `target_{h}min_abs`.
    Rows where any horizon is missing are dropped.
    """
    out_rows = []
    for sid, grp in snapshots.groupby("station_id", sort=False):
        s = grp.set_index("ts")["bike_count"].sort_index()
        # snap to nearest minute so 1-min cadence with jitter still aligns
        s.index = s.index.floor("1min")
        s = s.groupby(level=0).last()
        # Build a lookup
        idx_set = set(s.index)
        for ts, current in s.items():
            row = {"station_id": sid, "ts": ts, "bike_count": int(current)}
            ok = True
            for h in horizons_min:
                target_ts = ts + pd.Timedelta(minutes=h)
                if target_ts in idx_set:
                    future = int(s.loc[target_ts])
                    row[f"target_{h}min_abs"] = future
                    row[f"target_{h}min_delta"] = future - int(current)
                else:
                    ok = False
                    break
            if ok:
                out_rows.append(row)
    return pd.DataFrame(out_rows)

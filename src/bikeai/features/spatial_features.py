"""Spatial features: station coordinates, capacity, neighbor count averages.

We compute a simple "neighbors within radius R" graph from the station master, then
expose helpers that summarize neighbor bike counts as a feature column.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def build_neighbor_index(stations: pd.DataFrame, radius_m: float = 500.0) -> dict[str, list[str]]:
    """Return {station_id: [neighbor_ids...]} excluding the station itself."""
    ids = stations["station_id"].to_numpy()
    lats = stations["lat"].to_numpy()
    lons = stations["lon"].to_numpy()

    index: dict[str, list[str]] = {}
    for i, sid in enumerate(ids):
        d = haversine_m(np.full_like(lats, lats[i]), np.full_like(lons, lons[i]), lats, lons)
        mask = (d <= radius_m) & (ids != sid)
        index[sid] = ids[mask].tolist()
    return index


def add_neighbor_mean(
    series: pd.DataFrame,
    neighbor_index: dict[str, list[str]],
) -> pd.DataFrame:
    """Add `neighbor_mean_count`: mean bike_count of neighbors at the same ts.

    Implementation: pivot to wide [ts × station_id], take per-row mean of neighbor
    columns. For sparse use cases (few stations) this is fine; for the full Seoul
    network (~2,700 stations) consider chunking by ts windows.
    """
    if series.empty:
        return series.assign(neighbor_mean_count=pd.Series(dtype="float32"))

    wide = series.pivot_table(
        index="ts", columns="station_id", values="bike_count", aggfunc="last"
    )
    means = pd.DataFrame(index=wide.index)
    for sid in wide.columns:
        neighbors = [n for n in neighbor_index.get(sid, []) if n in wide.columns]
        if neighbors:
            means[sid] = wide[neighbors].mean(axis=1)
        else:
            means[sid] = math.nan

    long = means.stack(future_stack=True).rename("neighbor_mean_count").reset_index()
    long = long.rename(columns={"level_1": "station_id"})
    out = series.merge(long, on=["ts", "station_id"], how="left")
    out["neighbor_mean_count"] = out["neighbor_mean_count"].astype("float32")
    return out


def attach_station_attrs(series: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Join lat/lon onto every (station_id, ts) row."""
    cols = ["station_id", "lat", "lon"]
    return series.merge(stations[cols], on="station_id", how="left")

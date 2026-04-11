"""Combine reconstructed bike-count series + stations + weather + time features.

Produces a tidy DataFrame ready for LightGBM (with lag/rolling features) or for
pytorch-forecasting's TimeSeriesDataSet (which prefers lag-free input and computes
its own).
"""
from __future__ import annotations

import pandas as pd

from bikeai.config import HORIZONS_MIN
from bikeai.features.spatial_features import (
    add_neighbor_mean,
    attach_station_attrs,
    build_neighbor_index,
)
from bikeai.features.time_features import add_time_features


def build_feature_table(
    series: pd.DataFrame,
    stations: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    neighbor_radius_m: float = 500.0,
    add_lags: bool = True,
) -> pd.DataFrame:
    """Build the full feature/target table.

    Parameters
    ----------
    series   : reconstructed [station_id, ts, bike_count]
    stations : station master [station_id, lat, lon, capacity]
    weather  : optional [ts, temp_c, precip_mm, wind_ms, humidity_pct]
    add_lags : if True, add lag features (LightGBM-style); set False for TFT input.
    """
    df = attach_station_attrs(series, stations)

    neighbor_idx = build_neighbor_index(stations, radius_m=neighbor_radius_m)
    df = add_neighbor_mean(df, neighbor_idx)

    if weather is not None and not weather.empty:
        df = _join_weather(df, weather)

    df = add_time_features(df, ts_col="ts")

    df = _add_targets(df, HORIZONS_MIN)

    if add_lags:
        df = _add_lag_rolling(df)

    return df.sort_values(["station_id", "ts"]).reset_index(drop=True)


def _join_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """ASOS is hourly; forward-fill to the bike series' frequency via merge_asof."""
    weather = weather.sort_values("ts")
    df = df.sort_values("ts")
    return pd.merge_asof(df, weather, on="ts", direction="backward")


def _add_targets(df: pd.DataFrame, horizons_min: tuple[int, ...]) -> pd.DataFrame:
    """For each horizon h, add target_h = bike_count(t + h) by station."""
    out = df.sort_values(["station_id", "ts"]).copy()
    grouped = out.groupby("station_id", sort=False)["bike_count"]
    for h in horizons_min:
        out[f"target_{h}min"] = grouped.shift(-h)
    return out


def _add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling mean features per station — useful for tree models."""
    out = df.sort_values(["station_id", "ts"]).copy()
    g = out.groupby("station_id", sort=False)["bike_count"]
    for lag in (1, 5, 15, 30, 60):
        out[f"lag_{lag}"] = g.shift(lag)
    for win in (5, 15, 30, 60):
        out[f"roll_mean_{win}"] = g.shift(1).rolling(win, min_periods=1).mean()
    return out

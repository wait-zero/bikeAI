"""Per-station normalization features.

Different stations operate at very different absolute bike-count levels:
small stations rarely have more than 5 bikes, large ones routinely hold 30+.
A single global model trained on the raw `bike_count` column ends up dominated
by the easy low-activity stations and learns "predict no change". Normalizing
each station against its own mean/spread lets the model learn shape patterns
that transfer across stations of all sizes.

We compute four extra columns:
    st_mean, st_std, st_max  — per-station summary stats (constants per station)
    bike_count_centered      — bike_count - st_mean
    bike_count_z             — (bike_count - st_mean) / st_std
    bike_count_pct           — bike_count / st_max  (in [0, 1+])

These are added to the feature table BEFORE lag/rolling so that the lag/rolling
columns are available on the raw scale (the model can use both raw and
normalized when it picks splits). To also produce normalized lag features call
the helper after lag construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAT_COLUMNS = ["st_mean", "st_std", "st_max"]
NORM_COLUMNS = ["bike_count_centered", "bike_count_z", "bike_count_pct"]


def compute_station_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return [station_id, st_mean, st_std, st_max] for each station in `df`."""
    if df.empty:
        return pd.DataFrame(columns=["station_id"] + STAT_COLUMNS)
    g = df.groupby("station_id")["bike_count"]
    stats = pd.DataFrame(
        {
            "st_mean": g.mean(),
            "st_std": g.std().fillna(0).clip(lower=0.5),  # avoid div-by-zero
            "st_max": g.max().clip(lower=1),
        }
    )
    return stats.reset_index()


def attach_normalization(
    df: pd.DataFrame,
    stats: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach per-station stats and add normalized bike_count columns.

    Returns
    -------
    out   : df with new columns (st_mean, st_std, st_max, bike_count_*)
    stats : the stats table actually used (computed if not provided)
    """
    if stats is None:
        stats = compute_station_stats(df)

    out = df.merge(stats, on="station_id", how="left")
    # If a station is missing in stats (shouldn't normally happen), fill with global
    if out["st_mean"].isna().any():
        global_mean = float(df["bike_count"].mean())
        global_std = float(max(df["bike_count"].std(), 0.5))
        global_max = float(max(df["bike_count"].max(), 1))
        out["st_mean"] = out["st_mean"].fillna(global_mean)
        out["st_std"] = out["st_std"].fillna(global_std)
        out["st_max"] = out["st_max"].fillna(global_max)

    out["bike_count_centered"] = out["bike_count"] - out["st_mean"]
    out["bike_count_z"] = out["bike_count_centered"] / out["st_std"]
    out["bike_count_pct"] = out["bike_count"] / out["st_max"]

    return out, stats

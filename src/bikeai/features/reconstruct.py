"""Reconstruct station-level bike count time series from rental events.

Rental history gives one row per trip: (rent_station, rent_dt, return_station, return_dt).
We turn each trip into two delta events:
    rent_dt   @ rent_station   delta = -1
    return_dt @ return_station delta = +1

Cumulative-sum per station produces a *relative* bike count series. To anchor it to
absolute values, we need a known count at some reference time. The pipeline supports
two anchor sources:
    1. Realtime API snapshots (preferred — exact ground truth)
    2. Daily nightly low (3-5 AM) anchor estimated from station capacity (fallback)

Reconstruction limits:
    - Trucks redistributing bikes are NOT in the rental log, so cumulative drift is
      possible across long horizons. Anchor every day (or more often) to bound it.

Output: tidy DataFrame with columns [station_id, ts, bike_count] resampled to a
fixed frequency (default 1 minute) with forward fill between events.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bikeai.config import RESAMPLE_FREQ


@dataclass(frozen=True)
class ReconstructionResult:
    series: pd.DataFrame  # columns: station_id, ts, bike_count
    anchors_used: int
    stations: int


def events_from_rentals(rentals: pd.DataFrame) -> pd.DataFrame:
    """Explode rental rows into (station_id, ts, delta) events."""
    if rentals.empty:
        return pd.DataFrame(columns=["station_id", "ts", "delta"])

    rents = pd.DataFrame(
        {
            "station_id": rentals["rent_station"].to_numpy(),
            "ts": rentals["rent_dt"].to_numpy(),
            "delta": -1,
        }
    )
    returns = pd.DataFrame(
        {
            "station_id": rentals["return_station"].to_numpy(),
            "ts": rentals["return_dt"].to_numpy(),
            "delta": 1,
        }
    )
    events = pd.concat([rents, returns], ignore_index=True)
    events["ts"] = pd.to_datetime(events["ts"])
    return events.sort_values(["station_id", "ts"]).reset_index(drop=True)


def cumulative_relative_count(events: pd.DataFrame) -> pd.DataFrame:
    """Per-station cumulative sum of deltas. Result is relative (starts at 0)."""
    if events.empty:
        return events.assign(rel_count=pd.Series(dtype="int64"))
    out = events.copy()
    out["rel_count"] = out.groupby("station_id")["delta"].cumsum()
    return out


def apply_anchors(
    rel_series: pd.DataFrame,
    anchors: pd.DataFrame,
    method: str = "first",
) -> pd.DataFrame:
    """Shift each station's relative series so that observed anchor points match.

    Parameters
    ----------
    rel_series : DataFrame with [station_id, ts, rel_count]
    anchors    : DataFrame with [station_id, ts, bike_count] — known truth
    method     : 'first' uses the chronologically-first anchor per station; 'mean'
                 uses the average offset across all anchors (more robust to noise but
                 also smears across drift).
    """
    if rel_series.empty:
        return rel_series.assign(bike_count=pd.Series(dtype="int64"))

    if anchors.empty:
        return rel_series.assign(bike_count=rel_series["rel_count"].clip(lower=0))

    rel = rel_series.sort_values(["station_id", "ts"]).copy()
    anc = anchors.sort_values(["station_id", "ts"]).copy()

    # For each anchor, look up the rel_count at the latest event <= anchor ts.
    # That gives us offset = bike_count - rel_count_at_anchor_time.
    merged = pd.merge_asof(
        anc,
        rel[["station_id", "ts", "rel_count"]],
        by="station_id",
        on="ts",
        direction="backward",
    )
    merged["offset"] = merged["bike_count"] - merged["rel_count"].fillna(0)

    if method == "first":
        offsets = merged.groupby("station_id")["offset"].first()
    elif method == "mean":
        offsets = merged.groupby("station_id")["offset"].mean()
    else:
        raise ValueError(f"unknown anchor method: {method}")

    rel["offset"] = rel["station_id"].map(offsets).fillna(0)
    rel["bike_count"] = (rel["rel_count"] + rel["offset"]).round().astype("int64")
    rel["bike_count"] = rel["bike_count"].clip(lower=0)
    return rel.drop(columns=["offset"])


def resample_to_grid(
    rel_series: pd.DataFrame,
    freq: str = RESAMPLE_FREQ,
    capacity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resample event-level series to a fixed grid using forward fill.

    If `capacity` is provided (columns [station_id, capacity]), the resulting
    bike_count is also clipped to [0, capacity] per station.
    """
    if rel_series.empty:
        return pd.DataFrame(columns=["station_id", "ts", "bike_count"])

    pieces = []
    for sid, grp in rel_series.groupby("station_id", sort=False):
        s = (
            grp.set_index("ts")["bike_count"]
            .groupby(level=0)
            .last()  # collapse simultaneous events
            .resample(freq)
            .ffill()
            .dropna()
            .astype("int64")
        )
        pieces.append(pd.DataFrame({"station_id": sid, "ts": s.index, "bike_count": s.values}))

    out = pd.concat(pieces, ignore_index=True)

    if capacity is not None and not capacity.empty:
        cap_map = capacity.set_index("station_id")["capacity"].to_dict()
        out["bike_count"] = np.minimum(
            out["bike_count"].to_numpy(),
            out["station_id"].map(cap_map).fillna(np.iinfo(np.int64).max).astype("int64").to_numpy(),
        )
    return out


def reconstruct(
    rentals: pd.DataFrame,
    anchors: pd.DataFrame | None = None,
    capacity: pd.DataFrame | None = None,
    freq: str = RESAMPLE_FREQ,
) -> ReconstructionResult:
    """End-to-end: rental events → resampled station × time bike_count series.

    `anchors` should be a tidy DataFrame [station_id, ts, bike_count]. Pass realtime
    snapshots in here. If None, the series is anchored to start at 0 — only useful for
    deltas / sanity checks.
    """
    events = events_from_rentals(rentals)
    rel = cumulative_relative_count(events)
    abs_series = apply_anchors(
        rel,
        anchors if anchors is not None else pd.DataFrame(columns=["station_id", "ts", "bike_count"]),
    )
    grid = resample_to_grid(abs_series, freq=freq, capacity=capacity)

    return ReconstructionResult(
        series=grid,
        anchors_used=0 if anchors is None else len(anchors),
        stations=grid["station_id"].nunique() if not grid.empty else 0,
    )

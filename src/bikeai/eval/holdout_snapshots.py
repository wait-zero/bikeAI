"""Evaluate a trained delta model against real collected snapshots.

The training data has the absolute-value drift problem (no anchors), so its own
metrics are not trustworthy. This module loads a separate `train_full.csv`-style
snapshot file, which has the *real* absolute bike counts, and asks the question
the user actually cares about:

    "Given current state at t, how many bikes will be here in 5/15/30/60 min?"

For each (station, t) snapshot:
    1. Build the same feature row that build_dataset would build (time features,
       neighbor mean from snapshots, weather join, lag/rolling).
    2. Ask the model for its delta predictions.
    3. Compare against the real observed delta from the snapshot file.
    4. Also compute predicted absolute = current + delta and compare against
       the real absolute observation.
    5. Compare against a trivial baseline (delta=0, i.e. persistence).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bikeai.config import HORIZONS_MIN
from bikeai.eval.metrics import mae, rmse, smape
from bikeai.features.spatial_features import add_neighbor_mean, build_neighbor_index
from bikeai.features.station_norm import attach_normalization
from bikeai.features.time_features import add_time_features
from bikeai.ingest.snapshots import snapshot_horizons


@dataclass
class HoldoutResult:
    per_horizon: dict[str, dict[str, float]]
    n_rows: int
    stations: int


def build_holdout_features(
    snapshots: pd.DataFrame,
    stations: pd.DataFrame,
    weather: pd.DataFrame | None,
    horizons_min: tuple[int, ...] = HORIZONS_MIN,
    station_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reproduce the training-time feature columns from snapshot data alone.

    Important: snapshots are the only source of bike_count here, so neighbor
    means and lag features are computed *within* the snapshot window. We do not
    use rental-history reconstruction at all on this side of the problem.
    """
    # Filter to stations also present in master so spatial joins succeed
    common = set(snapshots["station_id"]) & set(stations["station_id"])
    snaps = snapshots[snapshots["station_id"].isin(common)].copy()
    masters = stations[stations["station_id"].isin(common)].copy()

    # Attach lat/lon from master (snapshot already has them but we use master to
    # match training-time provenance)
    snaps = snaps.drop(columns=["lat", "lon"], errors="ignore").merge(
        masters[["station_id", "lat", "lon"]], on="station_id", how="left"
    )

    # Per-station normalization (uses provided stats or computes from this snapshot set)
    snaps, _ = attach_normalization(snaps, stats=station_stats)

    # Spatial features — neighbor mean within the snapshot window
    neighbor_idx = build_neighbor_index(masters, radius_m=500.0)
    snaps = add_neighbor_mean(snaps, neighbor_idx)

    # Weather join
    if weather is not None and not weather.empty:
        snaps = snaps.sort_values("ts")
        weather_sorted = weather.sort_values("ts")
        snaps = pd.merge_asof(snaps, weather_sorted, on="ts", direction="backward")

    # Time features
    snaps = add_time_features(snaps, ts_col="ts")

    # Lag / rolling features within the snapshot window
    snaps = snaps.sort_values(["station_id", "ts"])
    g = snaps.groupby("station_id", sort=False)["bike_count"]
    for lag in (1, 5, 15, 30, 60):
        snaps[f"lag_{lag}"] = g.shift(lag)
    for win in (5, 15, 30, 60):
        snaps[f"roll_mean_{win}"] = g.shift(1).rolling(win, min_periods=1).mean()

    # Build absolute + delta horizon labels from the snapshot file itself
    horizon_table = snapshot_horizons(snaps[["station_id", "ts", "bike_count", "lat", "lon"]], horizons_min)
    if horizon_table.empty:
        return horizon_table

    # Merge labels with the feature columns we just built
    feature_cols = [c for c in snaps.columns if c not in ("station_id", "ts", "bike_count", "lat", "lon")]
    feats = snaps[["station_id", "ts", "bike_count", "lat", "lon"] + feature_cols]
    merged = horizon_table.merge(feats, on=["station_id", "ts", "bike_count"], how="inner", suffixes=("", "_dup"))
    return merged


def evaluate_holdout(
    model,
    holdout_features: pd.DataFrame,
    horizons_min: tuple[int, ...] = HORIZONS_MIN,
) -> HoldoutResult:
    """Run the trained model on holdout features and compute per-horizon metrics."""
    if holdout_features.empty:
        return HoldoutResult(per_horizon={}, n_rows=0, stations=0)

    # Drop rows with missing lag features (start of snapshot window)
    needed = ["lag_60", "roll_mean_60"]
    df = holdout_features.dropna(subset=needed).copy()
    if df.empty:
        return HoldoutResult(per_horizon={}, n_rows=0, stations=0)

    preds = model.predict(df)
    df = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    per_horizon: dict[str, dict[str, float]] = {}
    for h in horizons_min:
        delta_col = f"target_{h}min_delta"
        abs_col = f"target_{h}min_abs"
        pred_col = f"pred_{h}min"
        if not {delta_col, abs_col, pred_col}.issubset(df.columns):
            continue
        sub = df[[delta_col, abs_col, pred_col, "bike_count"]].dropna()
        if sub.empty:
            continue
        true_delta = sub[delta_col].to_numpy(dtype=float)
        pred_delta = sub[pred_col].to_numpy(dtype=float)
        true_abs = sub[abs_col].to_numpy(dtype=float)
        current = sub["bike_count"].to_numpy(dtype=float)
        pred_abs = np.maximum(current + pred_delta, 0)

        # Persistence baseline: predict no change (delta=0, abs=current)
        baseline_delta = np.zeros_like(true_delta)
        baseline_abs = current

        per_horizon[f"{h}min"] = {
            "n": int(len(sub)),
            "delta_mae": mae(true_delta, pred_delta),
            "delta_rmse": rmse(true_delta, pred_delta),
            "abs_mae": mae(true_abs, pred_abs),
            "abs_rmse": rmse(true_abs, pred_abs),
            "abs_smape": smape(true_abs, pred_abs),
            "persistence_delta_mae": mae(true_delta, baseline_delta),
            "persistence_abs_mae": mae(true_abs, baseline_abs),
            "improvement_over_persistence": float(
                mae(true_delta, baseline_delta) - mae(true_delta, pred_delta)
            ),
        }

    return HoldoutResult(
        per_horizon=per_horizon,
        n_rows=len(df),
        stations=int(df["station_id"].nunique()),
    )

"""Train a delta model on snapshot + rental_history combined (weighted concat).

Plan ref: 2026-04 Phase 3 — `docs jaunty-wobbling-bachman.md`.

Strategy:
    - Each source is normalized with its own per-station stats (drift offset
      cancels out for both).
    - Only normalized features are used (no raw bike_count / lag_*) so the two
      sources live in a comparable feature space.
    - snapshot rows get sample_weight=30 so the optimizer keeps the snapshot
      distribution central despite rental having ~30x more rows.
    - Evaluation uses the snapshot's own time-split test window so we measure
      the metric the user actually cares about.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bikeai.eval.metrics import mae, smape
from bikeai.features.build_dataset import build_feature_table
from bikeai.features.reconstruct import reconstruct
from bikeai.features.station_norm import compute_station_stats
from bikeai.ingest.rental_history import load_rental_history
from bikeai.ingest.snapshots import load_collected_snapshots
from bikeai.ingest.stations import load_station_master
from bikeai.ingest.weather import load_asos_csv
from bikeai.models.baseline_lgbm import LgbmBaseline


def lap(t0, msg):
    print(f"  [{time.time() - t0:6.1f}s] {msg}")
    return time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-stations", type=int, default=500)
    parser.add_argument("--snapshot-weight", type=float, default=30.0)
    parser.add_argument("--rental-resample", type=str, default="5min")
    parser.add_argument("--num-boost-round", type=int, default=300)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    snap_path = project_root / "data" / "raw" / "snapshots_train_full.csv"
    master_path = project_root / "data" / "raw" / "stations.csv"
    weather_apr_path = Path.home() / "Downloads" / "OBS_ASOS_MI_20260412005900.csv"
    weather_winter_path = project_root / "data" / "raw" / "asos_seoul.csv"
    rental_paths = sorted((project_root / "data" / "raw" / "rental_history").rglob("*.csv"))

    t0 = time.time()
    print("=== STEP 1: load snapshot + master + weather + rental ===")
    snapshots = load_collected_snapshots(snap_path)
    master = load_station_master(master_path)

    # Snapshot weather (April proxy from Apr 10 shifted to Apr 11)
    weather_snap = load_asos_csv(weather_apr_path, station_id=108)
    weather_snap["ts"] = weather_snap["ts"] + pd.Timedelta(days=1)

    # Rental weather (the original Nov-Dec hourly TIM)
    weather_rental = load_asos_csv(weather_winter_path, station_id=108)

    rentals = load_rental_history(rental_paths)
    t0 = lap(t0, f"loaded snap={len(snapshots):,}, master={len(master):,}, "
                  f"weather_snap={len(weather_snap)}, weather_rental={len(weather_rental)}, "
                  f"rentals={len(rentals):,}")

    print("\n=== STEP 2: pick top-N busiest stations (intersection) ===")
    busy_snap = snapshots.groupby("station_id")["bike_count"].sum().sort_values(ascending=False)
    busy_rental = pd.concat([rentals["rent_station"], rentals["return_station"]]).value_counts()
    common = (
        set(busy_snap.head(args.top_stations * 2).index)
        & set(busy_rental.head(args.top_stations * 2).index)
        & set(master["station_id"])
    )
    # Pick the top-N busiest from snapshot perspective among the common set
    top = [s for s in busy_snap.index if s in common][: args.top_stations]
    print(f"  using {len(top)} stations (snapshot top {args.top_stations} ∩ rental top + master)")

    snapshots = snapshots[snapshots["station_id"].isin(top)].copy()
    master = master[master["station_id"].isin(top)].copy()
    rentals = rentals[
        rentals["rent_station"].isin(top) & rentals["return_station"].isin(top)
    ].copy()
    t0 = lap(t0, f"subset: snap={len(snapshots):,} rental_events={len(rentals):,}")

    print("\n=== STEP 3: build SNAPSHOT feature table (normalized only) ===")
    snap_stats = compute_station_stats(snapshots[["station_id", "bike_count"]])
    snap_series = snapshots[["station_id", "ts", "bike_count"]].copy()
    snap_feats = build_feature_table(
        snap_series,
        master,
        weather=weather_snap,
        target_mode="delta",
        lags=(1, 5, 15),
        roll_windows=(5, 15),
        normalize_per_station=True,
        station_stats=snap_stats,
    )
    target_cols = [f"target_{h}min" for h in (5, 15, 30, 60)]
    snap_feats = snap_feats.dropna(subset=target_cols + ["lag_15", "roll_mean_15"])
    snap_feats["source"] = "snapshot"
    snap_feats["sample_weight"] = args.snapshot_weight
    t0 = lap(t0, f"snap_feats: {len(snap_feats):,} rows")

    print("\n=== STEP 4: reconstruct rental → series → feature table ===")
    rec = reconstruct(rentals, anchors=None)
    rental_series = rec.series
    print(f"  rental reconstructed rows: {len(rental_series):,}")

    # Downsample rental to a coarser grid to keep memory manageable
    if args.rental_resample:
        step = pd.Timedelta(args.rental_resample) // pd.Timedelta("1min")
        rental_series = rental_series[rental_series["ts"].dt.minute % step == 0]
        rental_series = rental_series.reset_index(drop=True)
        print(f"  after {args.rental_resample} resample: {len(rental_series):,}")
    t0 = lap(t0, "rental reconstructed")

    rental_stats = compute_station_stats(rental_series[["station_id", "bike_count"]])
    print(f"  rental_stats: {len(rental_stats)} stations, "
          f"st_mean range {rental_stats['st_mean'].min():.0f}~{rental_stats['st_mean'].max():.0f}")

    rental_feats = build_feature_table(
        rental_series,
        master,
        weather=weather_rental,
        target_mode="delta",
        lags=(1, 5, 15),
        roll_windows=(5, 15),
        normalize_per_station=True,
        station_stats=rental_stats,
    )
    rental_feats = rental_feats.dropna(subset=target_cols + ["lag_15", "roll_mean_15"])
    rental_feats["source"] = "rental"
    rental_feats["sample_weight"] = 1.0
    t0 = lap(t0, f"rental_feats: {len(rental_feats):,} rows")

    print("\n=== STEP 5: time-split snapshot (70/30) BEFORE concat ===")
    snap_ts_sorted = sorted(snap_feats["ts"].unique())
    snap_split = snap_ts_sorted[int(len(snap_ts_sorted) * 0.7)]
    snap_train = snap_feats[snap_feats["ts"] < snap_split]
    snap_test = snap_feats[snap_feats["ts"] >= snap_split]
    print(f"  snap train: {len(snap_train):,}  snap test: {len(snap_test):,}")

    print("\n=== STEP 6: concat snap_train + rental_feats ===")
    combined = pd.concat([snap_train, rental_feats], ignore_index=True)
    print(f"  combined train: {len(combined):,} rows")
    print(f"    snap rows in train:   {(combined['source']=='snapshot').sum():,}")
    print(f"    rental rows in train: {(combined['source']=='rental').sum():,}")
    print(f"    effective weight ratio (snap*w / rental*w): "
          f"{(combined.loc[combined['source']=='snapshot','sample_weight'].sum()):.0f} / "
          f"{(combined.loc[combined['source']=='rental','sample_weight'].sum()):.0f}")

    print("\n=== STEP 7: train LGBM (normalized_only, weighted) ===")
    model = LgbmBaseline(
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=30,
        feature_set="normalized_only",
    ).fit(combined, valid_df=snap_test, sample_weight=combined["sample_weight"].to_numpy())
    print(f"  trained {len(model.trained_features)} features: {model.trained_features}")
    t0 = lap(t0, "trained")

    print("\n=== STEP 8: predict on snap_test + hybrid 5min ===")
    preds = model.predict(snap_test)
    preds["pred_5min"] = 0.0
    merged = pd.concat([snap_test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

    print("\n=== RESULTS — combined (snapshot + rental, normalized) ===")
    # Reference: previous best (snapshot only with normalization)
    ref = {"5min": 0.387, "15min": 0.898, "30min": 1.452, "60min": 1.446}
    persistence_ref = {"5min": 0.387, "15min": 0.894, "30min": 1.451, "60min": 2.193}

    print(f"{'Horizon':<8} {'Snap-only':<12} {'Combined':<12} {'Δ':<10} {'Persist':<10} {'sMAPE':<8}")
    for h in (5, 15, 30, 60):
        y = merged[f"target_{h}min"].to_numpy(dtype=float)
        p = merged[f"pred_{h}min"].to_numpy(dtype=float)
        current = merged["bike_count"].to_numpy(dtype=float)
        true_abs = current + y
        pred_abs = np.maximum(current + p, 0)

        new_mae = mae(y, p)
        diff = ref[f"{h}min"] - new_mae
        arrow = "✅" if diff > 0.005 else ("=" if abs(diff) < 0.005 else "❌")
        sm = smape(true_abs, pred_abs)
        print(f"{h}min     {ref[f'{h}min']:<12.3f} {new_mae:<12.3f} {diff:+.3f}    "
              f"{persistence_ref[f'{h}min']:<10.3f} {sm:.2f}%  {arrow}")


if __name__ == "__main__":
    main()

"""Train the snapshot-only normalized model and save the full serving artifact.

Output directory layout:
    models_out/v1/
    ├── lgbm_5min.txt
    ├── lgbm_15min.txt
    ├── lgbm_30min.txt
    ├── lgbm_60min.txt
    ├── station_stats.parquet      [station_id, st_mean, st_std, st_max]
    ├── master.parquet             [station_id, station_name, lat, lon]
    ├── neighbor_index.json        {station_id: [neighbor_ids...]}
    ├── feature_schema.json        {"features": [...], "categorical": [...]}
    └── model_card.json            metadata + metrics
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bikeai.features.build_dataset import build_feature_table
from bikeai.features.spatial_features import build_neighbor_index
from bikeai.features.station_norm import compute_station_stats
from bikeai.ingest.snapshots import load_collected_snapshots
from bikeai.ingest.stations import load_station_master
from bikeai.ingest.weather import load_asos_csv
from bikeai.models.baseline_lgbm import LgbmBaseline
from bikeai.eval.metrics import mae, smape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models_out/v1"))
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--neighbor-radius-m", type=float, default=500.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]

    print("=== STEP 1: load data ===")
    snapshots = load_collected_snapshots(project_root / "data/raw/snapshots_train_full.csv")
    master = load_station_master(project_root / "data/raw/stations.csv")
    weather_apr10 = load_asos_csv(
        Path.home() / "Downloads/OBS_ASOS_MI_20260412005900.csv", station_id=108
    )
    weather = weather_apr10.copy()
    weather["ts"] = weather["ts"] + pd.Timedelta(days=1)
    print(f"  snapshots: {len(snapshots):,}  master: {len(master):,}  weather: {len(weather)}")

    common = set(master["station_id"]) & set(snapshots["station_id"])
    snapshots = snapshots[snapshots["station_id"].isin(common)].copy()
    master = master[master["station_id"].isin(common)].copy().reset_index(drop=True)
    print(f"  common stations: {len(common):,}")

    print("\n=== STEP 2: compute station_stats from first 60% (no leak) ===")
    ts_sorted = sorted(snapshots["ts"].unique())
    stats_cutoff = ts_sorted[int(len(ts_sorted) * 0.6)]
    stats_period = snapshots[snapshots["ts"] < stats_cutoff]
    station_stats = compute_station_stats(stats_period[["station_id", "bike_count"]])
    # Some stations may not appear in the early window; fall back to full-period stats
    missing = set(snapshots["station_id"]) - set(station_stats["station_id"])
    if missing:
        extra = compute_station_stats(
            snapshots[snapshots["station_id"].isin(missing)][["station_id", "bike_count"]]
        )
        station_stats = pd.concat([station_stats, extra], ignore_index=True)
    print(f"  station_stats: {len(station_stats)} stations")

    print("\n=== STEP 3: build feature table (delta, normalize, lag_15) ===")
    series = snapshots[["station_id", "ts", "bike_count"]].copy()
    feats = build_feature_table(
        series, master, weather=weather,
        target_mode="delta",
        lags=(1, 5, 15),
        roll_windows=(5, 15),
        normalize_per_station=True,
        station_stats=station_stats,
        neighbor_radius_m=args.neighbor_radius_m,
    )
    target_cols = [f"target_{h}min" for h in (5, 15, 30, 60)]
    feats = feats.dropna(subset=target_cols + ["lag_15", "roll_mean_15"])
    print(f"  feature table: {len(feats):,} rows × {len(feats.columns)} cols")

    print("\n=== STEP 4: time split + train ===")
    ts2 = sorted(feats["ts"].unique())
    split_ts = ts2[int(len(ts2) * 0.7)]
    train = feats[feats["ts"] < split_ts]
    test = feats[feats["ts"] >= split_ts]
    print(f"  train: {len(train):,}  test: {len(test):,}")

    t0 = time.time()
    model = LgbmBaseline(
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=30,
    ).fit(train, valid_df=test)
    print(f"  trained in {time.time()-t0:.1f}s, {len(model.trained_features)} features")

    print("\n=== STEP 5: evaluate on test (for model card) ===")
    preds = model.predict(test)
    preds["pred_5min"] = 0.0  # hybrid persistence
    merged = pd.concat([test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    metrics = {}
    for h in (5, 15, 30, 60):
        y = merged[f"target_{h}min"].to_numpy(dtype=float)
        p = merged[f"pred_{h}min"].to_numpy(dtype=float)
        current = merged["bike_count"].to_numpy(dtype=float)
        true_abs = current + y
        pred_abs = np.maximum(current + p, 0)
        metrics[f"{h}min"] = {
            "delta_mae": float(mae(y, p)),
            "abs_mae": float(mae(true_abs, pred_abs)),
            "abs_smape": float(smape(true_abs, pred_abs)),
            "persistence_delta_mae": float(mae(y, np.zeros_like(y))),
            "n": int(len(merged)),
        }
        print(f"  {h}min: MAE={metrics[f'{h}min']['delta_mae']:.3f}  "
              f"sMAPE={metrics[f'{h}min']['abs_smape']:.2f}%")

    print("\n=== STEP 6: save artifact ===")
    # 1) LightGBM models
    model.save(args.out)

    # 2) station_stats (with name/lat/lon merged for convenience)
    stats_full = station_stats.merge(
        master[["station_id", "station_name", "lat", "lon"]], on="station_id", how="left"
    )
    stats_full.to_parquet(args.out / "station_stats.parquet", index=False)

    # 3) master
    master.to_parquet(args.out / "master.parquet", index=False)

    # 4) neighbor index
    nbr = build_neighbor_index(master, radius_m=args.neighbor_radius_m)
    (args.out / "neighbor_index.json").write_text(json.dumps(nbr, ensure_ascii=False))

    # 5) feature schema (order matters for LGBM)
    schema = {
        "features": model.trained_features,
        "categorical": [c for c in ("station_id", "hour", "dow", "month", "is_weekend", "is_holiday")
                        if c in model.trained_features],
        "horizons_min": [5, 15, 30, 60],
        "lags": [1, 5, 15],
        "roll_windows": [5, 15],
        "neighbor_radius_m": args.neighbor_radius_m,
    }
    (args.out / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2))

    # 6) model card
    card = {
        "version": "v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "snapshots_train_full.csv",
        "data_period_start": str(snapshots["ts"].min()),
        "data_period_end": str(snapshots["ts"].max()),
        "stations": int(len(master)),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "horizons": [5, 15, 30, 60],
        "metrics": metrics,
        "feature_count": len(model.trained_features),
        "limitations": [
            "Trained on a single afternoon (13:39~16:24) — rush-hour patterns NOT learned",
            "Single calendar day (2026-04-11) — weekday/weekend effects NOT learned",
            "Weather is a same-time proxy from 2026-04-10",
            "5min horizon is forced to persistence (delta=0) — natural lower bound",
        ],
        "hybrid": {"5min": "persistence"},
    }
    (args.out / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2))

    print(f"\n✅ saved to {args.out.resolve()}")
    for p in sorted(args.out.iterdir()):
        size = p.stat().st_size
        print(f"  {p.name:<30} {size:>10,} bytes")


if __name__ == "__main__":
    main()

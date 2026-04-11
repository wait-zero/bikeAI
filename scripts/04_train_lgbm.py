"""Train the LightGBM multi-horizon baseline.

Inputs:
    data/processed/feature_table.parquet
Outputs:
    models_out/lgbm/lgbm_{h}min.txt
    reports/lgbm_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bikeai.config import HORIZONS_MIN, PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from bikeai.eval.metrics import horizon_report
from bikeai.models.baseline_lgbm import LgbmBaseline


def time_split(
    df: pd.DataFrame, train_end: str, valid_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("ts")
    train = df[df["ts"] < train_end]
    valid = df[(df["ts"] >= train_end) & (df["ts"] < valid_end)]
    test = df[df["ts"] >= valid_end]
    return train, valid, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "feature_table.parquet")
    parser.add_argument("--train-end", required=True, help="ISO date, exclusive (train < this)")
    parser.add_argument("--valid-end", required=True, help="ISO date, exclusive (valid < this)")
    parser.add_argument("--out-models", type=Path, default=PROJECT_ROOT / "models_out" / "lgbm")
    parser.add_argument("--out-report", type=Path, default=REPORTS_DIR / "lgbm_metrics.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    train, valid, test = time_split(df, args.train_end, args.valid_end)
    print(f"sizes: train={len(train):,} valid={len(valid):,} test={len(test):,}")

    model = LgbmBaseline().fit(train, valid)
    model.save(args.out_models)

    preds = model.predict(test)
    test_with_preds = pd.concat([test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    report = horizon_report(test_with_preds, HORIZONS_MIN)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Train the PyTorch TFT model on the same feature table the LGBM script uses.

Note: this loads `bikeai.models.tft` which lazily imports torch / pytorch-forecasting.
Install them first:  uv pip install torch pytorch-lightning pytorch-forecasting
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bikeai.config import PROCESSED_DIR, PROJECT_ROOT
from bikeai.models.tft import TFTRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "feature_table.parquet")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--valid-end", required=True)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models_out" / "tft" / "tft.ckpt")
    args = parser.parse_args()

    df = pd.read_parquet(args.features).sort_values("ts")
    train = df[df["ts"] < args.train_end]
    valid = df[(df["ts"] >= args.train_end) & (df["ts"] < args.valid_end)]
    print(f"sizes: train={len(train):,} valid={len(valid):,}")

    runner = TFTRunner(max_epochs=args.max_epochs)
    runner.fit(train, valid)
    runner.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

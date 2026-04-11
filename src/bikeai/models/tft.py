"""Temporal Fusion Transformer wrapper around `pytorch-forecasting`.

Multi-horizon (5/15/30/60 min) per-station bike-count forecasting with quantile output
so we can show users a confidence interval ("80% chance of 3-7 bikes").

Usage (from a script):

    from bikeai.models.tft import TFTRunner
    runner = TFTRunner(max_encoder_length=120, max_prediction_length=60)
    runner.fit(train_df, valid_df)
    preds = runner.predict(test_df)

The heavy torch / pytorch-forecasting imports are lazy so importing this module is
free in environments where torch isn't installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bikeai.config import HORIZONS_MIN

# Reasonable default quantiles. The middle (0.5) is the point estimate.
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


@dataclass
class TFTRunner:
    max_encoder_length: int = 120  # 2 hours of history
    max_prediction_length: int = 60  # predict up to 60 min ahead
    batch_size: int = 256
    learning_rate: float = 3e-3
    hidden_size: int = 32
    attention_head_size: int = 2
    dropout: float = 0.1
    hidden_continuous_size: int = 16
    max_epochs: int = 20
    horizons: tuple[int, ...] = HORIZONS_MIN
    quantiles: list[float] = field(default_factory=lambda: list(QUANTILES))

    _trainer: Any = None
    _model: Any = None
    _training_dataset: Any = None

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> "TFTRunner":
        torch, pl, pf, _, _ = _imports()

        train_ds = self._make_dataset(train_df, predict=False)
        self._training_dataset = train_ds
        train_loader = train_ds.to_dataloader(train=True, batch_size=self.batch_size, num_workers=0)

        valid_loader = None
        if valid_df is not None:
            valid_ds = pf.TimeSeriesDataSet.from_dataset(train_ds, valid_df, stop_randomization=True)
            valid_loader = valid_ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

        self._model = pf.TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=len(self.quantiles),
            loss=pf.metrics.QuantileLoss(quantiles=self.quantiles),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )

        self._trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            gradient_clip_val=0.1,
            enable_model_summary=False,
        )
        self._trainer.fit(self._model, train_loader, val_dataloaders=valid_loader)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        _, _, pf, _, _ = _imports()
        if self._model is None or self._training_dataset is None:
            raise RuntimeError("call fit() before predict()")

        ds = pf.TimeSeriesDataSet.from_dataset(self._training_dataset, df, predict=True, stop_randomization=True)
        loader = ds.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)
        raw = self._model.predict(loader, mode="raw", return_x=True)

        # raw.output.prediction shape: (n_samples, prediction_length, n_quantiles)
        # Pull out the requested horizons (in minute offsets) — assume 1-min frequency.
        preds = raw.output.prediction.cpu().numpy()
        x = raw.x

        rows = []
        decoder_time_idx = x["decoder_time_idx"].cpu().numpy()
        groups = x["groups"].cpu().numpy().squeeze(-1)
        for i in range(preds.shape[0]):
            for h in self.horizons:
                step = h - 1  # horizon h min → index h-1 in 1-min grid (0-based)
                if step >= preds.shape[1]:
                    continue
                row = {
                    "group_idx": int(groups[i]),
                    "horizon": h,
                    "time_idx": int(decoder_time_idx[i, step]),
                }
                for q_i, q in enumerate(self.quantiles):
                    row[f"q{int(q*100)}"] = float(preds[i, step, q_i])
                row["pred"] = row[f"q{int(self.quantiles[len(self.quantiles)//2]*100)}"]
                rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        if self._trainer is None or self._model is None:
            raise RuntimeError("nothing to save — call fit() first")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._trainer.save_checkpoint(str(path))

    def _make_dataset(self, df: pd.DataFrame, predict: bool):
        _, _, pf, _, _ = _imports()
        df = df.sort_values(["station_id", "ts"]).copy()
        # pytorch-forecasting requires an integer time index per group
        df["time_idx"] = (
            (df["ts"] - df.groupby("station_id")["ts"].transform("min"))
            .dt.total_seconds()
            .floordiv(60)
            .astype("int64")
        )

        return pf.TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="bike_count",
            group_ids=["station_id"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["station_id"],
            time_varying_known_reals=[
                c
                for c in (
                    "hour_sin",
                    "hour_cos",
                    "dow_sin",
                    "dow_cos",
                    "is_weekend",
                    "is_holiday",
                    "temp_c",
                    "precip_mm",
                    "wind_ms",
                    "humidity_pct",
                )
                if c in df.columns
            ],
            time_varying_unknown_reals=["bike_count", "neighbor_mean_count"]
            if "neighbor_mean_count" in df.columns
            else ["bike_count"],
            add_relative_time_idx=True,
            add_target_scales=True,
            allow_missing_timesteps=True,
        )


def _imports():
    """Lazy import torch / pytorch_lightning / pytorch_forecasting."""
    import pytorch_forecasting as pf
    import pytorch_lightning as pl
    import torch
    return torch, pl, pf, None, None

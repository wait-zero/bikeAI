"""LightGBM multi-horizon baseline.

One model per horizon (5/15/30/60 min). Trained on the feature table from
`features.build_dataset.build_feature_table`. Categorical: station_id, hour, dow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from bikeai.config import HORIZONS_MIN

CATEGORICAL = ["station_id", "hour", "dow", "month", "is_weekend", "is_holiday"]

# Raw, station-specific scale features. Useful when training on a single source
# (snapshot OR rental) but harmful when combining sources because the absolute
# scale differs between rental's drifted reconstruction and snapshot's true counts.
NUMERIC_RAW = [
    "bike_count",
    "lag_1",
    "lag_5",
    "lag_15",
    "lag_30",
    "lag_60",
    "roll_mean_5",
    "roll_mean_15",
    "roll_mean_30",
    "roll_mean_60",
]

# Per-station normalized features. These are scale-invariant by design and stay
# meaningful even when one source has a per-station drift offset.
NUMERIC_NORMALIZED = [
    "bike_count_centered",
    "bike_count_z",
    "bike_count_pct",
    "st_mean",
    "st_std",
    "st_max",
]

# Source-agnostic features (geo, time, weather, neighbor average).
NUMERIC_SHARED = [
    "lat",
    "lon",
    "neighbor_mean_count",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "temp_c",
    "precip_mm",
    "wind_ms",
    "humidity_pct",
]

# Default = all numeric columns. Used by single-source training.
NUMERIC = NUMERIC_RAW + NUMERIC_NORMALIZED + NUMERIC_SHARED


@dataclass
class LgbmBaseline:
    horizons: tuple[int, ...] = HORIZONS_MIN
    params: dict = field(
        default_factory=lambda: {
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 127,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 100,
            "verbosity": -1,
        }
    )
    num_boost_round: int = 500
    early_stopping_rounds: int = 30
    feature_set: str = "all"  # 'all' | 'normalized_only'
    models: dict[int, lgb.Booster] = field(default_factory=dict)
    trained_features: list[str] = field(default_factory=list)

    def feature_columns(self, df: pd.DataFrame) -> list[str]:
        if self.feature_set == "normalized_only":
            allowed = CATEGORICAL + NUMERIC_NORMALIZED + NUMERIC_SHARED
        else:
            allowed = CATEGORICAL + NUMERIC
        return [c for c in allowed if c in df.columns]

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> "LgbmBaseline":
        """Fit one booster per horizon. `sample_weight` is per-row weight (same length as train_df)."""
        feats = self.feature_columns(train_df)
        self.trained_features = feats
        cat_feats = [c for c in CATEGORICAL if c in feats]

        X_train = _coerce_features(train_df[feats], cat_feats)
        X_valid = _coerce_features(valid_df[feats], cat_feats) if valid_df is not None else None

        if sample_weight is not None:
            sw_train = np.asarray(sample_weight, dtype="float32")
            if len(sw_train) != len(train_df):
                raise ValueError(
                    f"sample_weight length {len(sw_train)} != train_df length {len(train_df)}"
                )
        else:
            sw_train = None

        for h in self.horizons:
            target = f"target_{h}min"
            mask_t = train_df[target].notna().to_numpy()
            y_train = train_df.loc[mask_t, target].astype("float32")
            w_train = sw_train[mask_t] if sw_train is not None else None
            train_set = lgb.Dataset(
                X_train.loc[mask_t],
                label=y_train,
                weight=w_train,
                categorical_feature=cat_feats,
                free_raw_data=False,
            )
            valid_sets = [train_set]
            valid_names = ["train"]
            callbacks = [lgb.log_evaluation(0)]
            if valid_df is not None:
                mask_v = valid_df[target].notna()
                y_valid = valid_df.loc[mask_v, target].astype("float32")
                valid_set = lgb.Dataset(
                    X_valid.loc[mask_v],
                    label=y_valid,
                    categorical_feature=cat_feats,
                    reference=train_set,
                    free_raw_data=False,
                )
                valid_sets.append(valid_set)
                valid_names.append("valid")
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

            booster = lgb.train(
                self.params,
                train_set,
                num_boost_round=self.num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            self.models[h] = booster
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Use the exact feature list from training to keep column order/count stable.
        feats = self.trained_features or self.feature_columns(df)
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise ValueError(f"prediction df is missing trained features: {missing}")
        cat_feats = [c for c in CATEGORICAL if c in feats]
        X = _coerce_features(df[feats], cat_feats)
        out = pd.DataFrame(index=df.index)
        for h, booster in self.models.items():
            preds = booster.predict(X, num_iteration=booster.best_iteration or None)
            # No clipping here — predictions may be negative when target is delta.
            # Absolute clipping (max(0, current+delta)) happens at inference time
            # in the holdout/serving layer.
            out[f"pred_{h}min"] = preds
        return out

    def save(self, dir_: Path) -> None:
        dir_.mkdir(parents=True, exist_ok=True)
        for h, booster in self.models.items():
            booster.save_model(str(dir_ / f"lgbm_{h}min.txt"))

    @classmethod
    def load(cls, dir_: Path, horizons: tuple[int, ...] = HORIZONS_MIN) -> "LgbmBaseline":
        inst = cls(horizons=horizons)
        for h in horizons:
            inst.models[h] = lgb.Booster(model_file=str(dir_ / f"lgbm_{h}min.txt"))
        return inst


def _coerce_features(X: pd.DataFrame, cat_feats: list[str]) -> pd.DataFrame:
    """LightGBM accepts pandas categorical for string columns; numerics stay as float."""
    X = X.copy()
    for c in cat_feats:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X

"""Evaluation metrics for the bike-count forecasting task.

Two families:
    1. Regression error    — MAE, RMSE, sMAPE
    2. User-facing decision — "is the station empty / full" classification accuracy
       (this is what users actually care about: "can I rent here in 30 min?")
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent. Safe for zero targets (returns 0 when both are 0)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    out = np.where(denom == 0, 0.0, diff / denom)
    return float(np.mean(out) * 100)


def empty_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    empty_threshold: int = 0,
) -> dict[str, float]:
    """How often does the model agree that a station is empty?

    'Empty' = bike_count <= empty_threshold (default: literally 0). Seoul Ttareungi
    has no capacity concept, so there is no symmetric 'full' metric.
    """
    true_empty = y_true <= empty_threshold
    pred_empty = y_pred <= empty_threshold
    return {"empty_accuracy": float(np.mean(true_empty == pred_empty))}


def horizon_report(df: pd.DataFrame, horizons: Iterable[int]) -> dict:
    """For a DF with target_{h}min and pred_{h}min columns, compute per-horizon metrics."""
    rows = {}
    for h in horizons:
        y_col = f"target_{h}min"
        p_col = f"pred_{h}min"
        if y_col not in df.columns or p_col not in df.columns:
            continue
        sub = df[[y_col, p_col]].dropna(subset=[y_col, p_col])
        if sub.empty:
            continue
        y = sub[y_col].to_numpy()
        p = sub[p_col].to_numpy()
        rows[f"{h}min"] = {
            "n": int(len(sub)),
            "mae": mae(y, p),
            "rmse": rmse(y, p),
            "smape": smape(y, p),
            **empty_accuracy(y, p),
        }
    return rows

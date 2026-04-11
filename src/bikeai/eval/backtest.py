"""Walk-forward (expanding window) backtest helper.

Time series cross-validation that strictly preserves chronology: each fold trains on
[start, fold_end) and evaluates on [fold_end, fold_end + horizon).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(
    df: pd.DataFrame,
    initial_train_days: int,
    valid_days: int,
    test_days: int,
    step_days: int | None = None,
) -> Iterator[Fold]:
    """Yield folds covering the time range in `df`.

    Each fold:
        train: [start, train_end)
        valid: [train_end, valid_end)
        test : [valid_end, test_end)
    Step controls how far the window moves between folds; defaults to `test_days`
    so test windows don't overlap.
    """
    step_days = step_days or test_days
    start = df["ts"].min().normalize()
    end = df["ts"].max().normalize() + pd.Timedelta(days=1)

    train_end = start + pd.Timedelta(days=initial_train_days)
    while True:
        valid_end = train_end + pd.Timedelta(days=valid_days)
        test_end = valid_end + pd.Timedelta(days=test_days)
        if test_end > end:
            return
        yield Fold(train_end=train_end, valid_end=valid_end, test_end=test_end)
        train_end = train_end + pd.Timedelta(days=step_days)


def split_by_fold(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["ts"] < fold.train_end]
    valid = df[(df["ts"] >= fold.train_end) & (df["ts"] < fold.valid_end)]
    test = df[(df["ts"] >= fold.valid_end) & (df["ts"] < fold.test_end)]
    return train, valid, test

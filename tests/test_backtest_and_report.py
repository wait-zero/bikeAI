"""Tests for backtest folds and report rendering."""
from __future__ import annotations

import pandas as pd

from bikeai.eval.backtest import split_by_fold, walk_forward_folds
from bikeai.eval.report import render_comparison_md


def test_walk_forward_folds_are_chronological_and_non_overlapping():
    df = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=200, freq="1D")})
    folds = list(
        walk_forward_folds(df, initial_train_days=30, valid_days=10, test_days=20)
    )
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end < f.valid_end < f.test_end
    # consecutive folds: train_end advances by step (=test_days=20)
    if len(folds) >= 2:
        delta = folds[1].train_end - folds[0].train_end
        assert delta == pd.Timedelta(days=20)


def test_split_by_fold_partitions_correctly():
    df = pd.DataFrame({"ts": pd.date_range("2025-01-01", periods=100, freq="1D")})
    folds = list(walk_forward_folds(df, 30, 10, 20))
    train, valid, test = split_by_fold(df, folds[0])
    assert train["ts"].max() < folds[0].train_end
    assert valid["ts"].min() >= folds[0].train_end
    assert test["ts"].min() >= folds[0].valid_end
    # no overlap
    assert set(train["ts"]) & set(valid["ts"]) == set()
    assert set(valid["ts"]) & set(test["ts"]) == set()


def test_render_comparison_md_writes_table(tmp_path):
    lgbm = {
        "5min": {"mae": 1.2, "rmse": 2.0, "smape": 15.0, "empty_accuracy": 0.9, "full_accuracy": 0.95},
        "60min": {"mae": 3.5, "rmse": 4.0, "smape": 30.0, "empty_accuracy": 0.8, "full_accuracy": 0.85},
    }
    tft = {
        "5min": {"mae": 1.0, "rmse": 1.8, "smape": 14.0, "empty_accuracy": 0.92, "full_accuracy": 0.96},
    }
    out = render_comparison_md(lgbm, tft, tmp_path / "report.md")
    text = out.read_text()
    assert "5min" in text
    assert "60min" in text
    assert "LGBM" in text
    assert "TFT" in text

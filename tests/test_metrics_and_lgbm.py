"""Smoke tests for metrics and a tiny end-to-end LightGBM training run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bikeai.eval.metrics import empty_accuracy, horizon_report, mae, rmse, smape
from bikeai.features.build_dataset import build_feature_table
from bikeai.models.baseline_lgbm import LgbmBaseline


def test_metrics_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 4.0])
    assert mae(y, p) == pytest_approx(1 / 3)
    assert rmse(y, p) == pytest_approx(np.sqrt(1 / 3))
    assert smape(y, p) > 0


def test_empty_accuracy_only():
    """No capacity concept, so only the empty-station classification matters."""
    y = np.array([0, 0, 5, 10])
    p = np.array([0, 1, 5, 10])
    out = empty_accuracy(y, p)
    assert out["empty_accuracy"] == 0.75
    assert "full_accuracy" not in out


def test_lgbm_smoke_end_to_end():
    """Train + predict on synthetic data. Mostly checks that nothing crashes."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2025-07-15 00:00", periods=24 * 60 * 3, freq="1min")  # 3 days
    n = len(times)

    # Synthetic: bikes oscillate by hour-of-day with noise
    hours = times.hour + times.minute / 60
    base = 10 + 5 * np.sin(2 * np.pi * hours / 24)
    bikes = np.clip(np.round(base + rng.normal(0, 1, n)), 0, 20).astype("int64")

    series = pd.DataFrame({"station_id": ["A"] * n, "ts": times, "bike_count": bikes})
    stations = pd.DataFrame({"station_id": ["A"], "lat": [37.5], "lon": [127.0]})

    feats = build_feature_table(series, stations, weather=None)
    feats = feats.dropna(subset=[f"target_{h}min" for h in (5, 15, 30, 60)])
    train = feats[feats["ts"] < "2025-07-17"]
    valid = feats[feats["ts"] >= "2025-07-17"]
    assert len(train) > 100 and len(valid) > 0

    model = LgbmBaseline(num_boost_round=20).fit(train, valid)
    preds = model.predict(valid)
    assert preds.shape[0] == len(valid)
    assert all(preds[c].notna().all() for c in preds.columns)

    merged = pd.concat([valid.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    rep = horizon_report(merged, (5, 15, 30, 60))
    assert "5min" in rep and rep["5min"]["mae"] >= 0


# tiny pytest.approx shim so we don't need to import pytest in the helper
def pytest_approx(value, rel=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= max(abs(value), 1.0) * rel

    return _Approx()

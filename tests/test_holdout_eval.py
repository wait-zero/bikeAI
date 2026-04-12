"""Smoke test for holdout snapshot evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bikeai.eval.holdout_snapshots import build_holdout_features, evaluate_holdout
from bikeai.features.build_dataset import build_feature_table
from bikeai.models.baseline_lgbm import LgbmBaseline


def _synthetic_snapshots(n_minutes: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    times = pd.date_range("2026-04-11 14:00", periods=n_minutes, freq="1min")
    rows = []
    for sid, lat, lon in [("ST-1", 37.50, 127.00), ("ST-2", 37.51, 127.01)]:
        bikes = np.clip(np.round(10 + 3 * np.sin(np.linspace(0, 4 * np.pi, n_minutes)) + rng.normal(0, 1, n_minutes)), 0, None)
        for t, b in zip(times, bikes):
            rows.append({"station_id": sid, "ts": t, "bike_count": int(b), "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def test_holdout_smoke_end_to_end():
    """Train a tiny LGBM on synthetic snapshots and run the holdout eval on the same data.

    The test only checks that nothing crashes and that the output structure is correct;
    it does NOT assert specific accuracy numbers.
    """
    snaps = _synthetic_snapshots(200)
    stations = pd.DataFrame(
        {"station_id": ["ST-1", "ST-2"], "lat": [37.50, 37.51], "lon": [127.00, 127.01]}
    )

    # Build training data with delta targets from the snapshots themselves
    train_feats = build_feature_table(snaps, stations, weather=None, target_mode="delta")
    train_feats = train_feats.dropna(subset=["target_5min", "lag_60", "roll_mean_60"])
    assert len(train_feats) > 50

    model = LgbmBaseline(num_boost_round=20).fit(train_feats)

    holdout = build_holdout_features(snaps, stations, weather=None)
    assert not holdout.empty

    result = evaluate_holdout(model, holdout)
    assert result.n_rows > 0
    assert "5min" in result.per_horizon
    metrics = result.per_horizon["5min"]
    for key in (
        "delta_mae",
        "delta_rmse",
        "abs_mae",
        "abs_rmse",
        "persistence_delta_mae",
        "improvement_over_persistence",
    ):
        assert key in metrics

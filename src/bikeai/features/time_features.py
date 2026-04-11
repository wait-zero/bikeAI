"""Time-of-day, day-of-week, holiday, and cyclical encodings."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from holidayskr import is_holiday  # type: ignore
except ImportError:  # pragma: no cover
    def is_holiday(d: str) -> bool:  # noqa: D401
        """Fallback no-op when holidayskr is not installed."""
        return False


def add_time_features(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Add hour, dow, month, weekend/holiday flags, and sin/cos cyclical encodings."""
    if ts_col not in df.columns:
        raise ValueError(f"missing column {ts_col!r}")

    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["hour"] = ts.dt.hour.astype("int16")
    out["minute"] = ts.dt.minute.astype("int16")
    out["dow"] = ts.dt.dayofweek.astype("int16")
    out["month"] = ts.dt.month.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["is_holiday"] = ts.dt.date.astype(str).map(_is_holiday_cached).astype("int8")

    # sin/cos so the model sees that 23:59 is close to 00:00
    out["hour_sin"] = np.sin(2 * np.pi * (out["hour"] + out["minute"] / 60) / 24)
    out["hour_cos"] = np.cos(2 * np.pi * (out["hour"] + out["minute"] / 60) / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    return out


_holiday_cache: dict[str, bool] = {}


def _is_holiday_cached(date_str: str) -> bool:
    if date_str not in _holiday_cache:
        try:
            _holiday_cache[date_str] = bool(is_holiday(date_str))
        except Exception:
            _holiday_cache[date_str] = False
    return _holiday_cache[date_str]

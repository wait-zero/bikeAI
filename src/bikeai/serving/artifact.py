"""Load a saved model artifact directory into memory."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import pandas as pd


@dataclass
class Artifact:
    version: str = ""
    boosters: dict[int, lgb.Booster] = field(default_factory=dict)
    station_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    master: pd.DataFrame = field(default_factory=pd.DataFrame)
    neighbor_index: dict[str, list[str]] = field(default_factory=dict)
    feature_schema: dict = field(default_factory=dict)
    model_card: dict = field(default_factory=dict)
    horizons: list[int] = field(default_factory=lambda: [5, 15, 30, 60])

    @classmethod
    def load(cls, path: Path) -> "Artifact":
        path = Path(path)
        card = json.loads((path / "model_card.json").read_text())
        schema = json.loads((path / "feature_schema.json").read_text())
        horizons = card.get("horizons", [5, 15, 30, 60])

        boosters = {}
        for h in horizons:
            f = path / f"lgbm_{h}min.txt"
            if f.exists():
                boosters[h] = lgb.Booster(model_file=str(f))

        return cls(
            version=card.get("version", "unknown"),
            boosters=boosters,
            station_stats=pd.read_parquet(path / "station_stats.parquet"),
            master=pd.read_parquet(path / "master.parquet"),
            neighbor_index=json.loads((path / "neighbor_index.json").read_text()),
            feature_schema=schema,
            model_card=card,
            horizons=horizons,
        )

    def station_info(self, station_id: str) -> dict | None:
        row = self.station_stats[self.station_stats["station_id"] == station_id]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "station_id": station_id,
            "station_name": r.get("station_name", ""),
            "lat": float(r["lat"]) if pd.notna(r.get("lat")) else None,
            "lon": float(r["lon"]) if pd.notna(r.get("lon")) else None,
            "st_mean": float(r["st_mean"]),
            "st_std": float(r["st_std"]),
            "st_max": float(r["st_max"]),
        }

    def neighbors_of(self, station_id: str) -> list[str]:
        return self.neighbor_index.get(station_id, [])

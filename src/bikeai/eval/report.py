"""Render LGBM-vs-TFT comparison reports as Markdown."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping


def render_comparison_md(
    lgbm_report: Mapping[str, Mapping[str, float]],
    tft_report: Mapping[str, Mapping[str, float]] | None,
    out_path: Path,
    extra_notes: str = "",
) -> Path:
    lines: list[str] = []
    lines.append(f"# 모델 비교 리포트 ({datetime.now():%Y-%m-%d %H:%M})\n")

    horizons = sorted(set(lgbm_report) | (set(tft_report) if tft_report else set()), key=_h_key)
    metrics = ["mae", "rmse", "smape", "empty_accuracy", "full_accuracy"]

    headers = ["horizon"] + [f"{m} (LGBM)" for m in metrics]
    if tft_report:
        headers += [f"{m} (TFT)" for m in metrics]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for h in horizons:
        row = [h]
        for m in metrics:
            row.append(_fmt(lgbm_report.get(h, {}).get(m)))
        if tft_report:
            for m in metrics:
                row.append(_fmt(tft_report.get(h, {}).get(m)))
        lines.append("| " + " | ".join(row) + " |")

    if extra_notes:
        lines.append("\n## 메모\n")
        lines.append(extra_notes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _h_key(h: str) -> int:
    return int(h.replace("min", ""))


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

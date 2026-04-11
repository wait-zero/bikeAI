"""Generate the LGBM-vs-TFT comparison Markdown report.

Reads:
    reports/lgbm_metrics.json
    reports/tft_metrics.json   (optional — only if TFT was trained)
Writes:
    reports/comparison_<timestamp>.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bikeai.config import REPORTS_DIR
from bikeai.eval.report import render_comparison_md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgbm", type=Path, default=REPORTS_DIR / "lgbm_metrics.json")
    parser.add_argument("--tft", type=Path, default=REPORTS_DIR / "tft_metrics.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPORTS_DIR / f"comparison_{datetime.now():%Y%m%dT%H%M%S}.md",
    )
    args = parser.parse_args()

    lgbm = json.loads(args.lgbm.read_text()) if args.lgbm.exists() else {}
    tft = json.loads(args.tft.read_text()) if args.tft.exists() else None

    out = render_comparison_md(lgbm, tft, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

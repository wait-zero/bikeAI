"""Compare station IDs across the master CSV, realtime snapshots, and (optionally)
the rental history CSV to find format mismatches and matching rate.

Why: the master CSV (data.go.kr 15099365) was last updated 2022-01-12, while the
realtime API uses `rntstnId` like "ST-10". The two sources may use different
conventions (bare integer vs ST-prefixed, zero-padding, etc.). This tool figures
out which normalization (if any) makes them match, and reports how many stations
exist on each side.

Run:
    python -m bikeai.tools.check_id_match \
        --master data/raw/stations.csv \
        --snapshot 'data/raw/realtime_snapshots/**/*.parquet'
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from bikeai.ingest.rental_history import load_rental_csv
from bikeai.ingest.stations import load_station_master

ST_PREFIXED = re.compile(r"^ST-\d+$")
BARE_INT = re.compile(r"^\d+$")
ST_PADDED = re.compile(r"^ST-0\d+$")  # ST-0010 etc

Normalizer = Callable[[str], str]


@dataclass
class FormatStats:
    total: int
    st_prefixed: int
    bare_int: int
    other: int
    examples: list[str]

    @property
    def dominant(self) -> str:
        counts = {"st-prefixed": self.st_prefixed, "bare-int": self.bare_int, "other": self.other}
        return max(counts, key=counts.get)


def classify_format(ids: Iterable[str]) -> FormatStats:
    ids = [str(i) for i in ids if i is not None and str(i) != "nan"]
    st = sum(1 for i in ids if ST_PREFIXED.match(i))
    bi = sum(1 for i in ids if BARE_INT.match(i))
    other = len(ids) - st - bi
    examples = list({*ids[:5]})[:5]
    return FormatStats(total=len(ids), st_prefixed=st, bare_int=bi, other=other, examples=examples)


def normalize_to_st(value: str) -> str:
    """Convert bare-int and zero-padded ST-prefixed IDs to canonical 'ST-N' (no leading zeros)."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if BARE_INT.match(s):
        return f"ST-{int(s)}"
    if ST_PREFIXED.match(s):
        # Strip leading zeros from the numeric portion
        n = int(s.split("-", 1)[1])
        return f"ST-{n}"
    return s


def compare_id_sets(
    master_ids: set[str],
    realtime_ids: set[str],
    normalizer: Normalizer,
) -> dict:
    norm_master = {normalizer(i) for i in master_ids if i}
    norm_realtime = {normalizer(i) for i in realtime_ids if i}
    matched = norm_master & norm_realtime
    return {
        "master_total": len(master_ids),
        "realtime_total": len(realtime_ids),
        "normalized_master": len(norm_master),
        "normalized_realtime": len(norm_realtime),
        "matched": len(matched),
        "master_only": len(norm_master - norm_realtime),
        "realtime_only": len(norm_realtime - norm_master),
        "match_rate_of_master": len(matched) / len(norm_master) if norm_master else 0.0,
        "match_rate_of_realtime": len(matched) / len(norm_realtime) if norm_realtime else 0.0,
    }


def load_snapshot_ids(pattern: str) -> set[str]:
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet matched: {pattern}")
    ids: set[str] = set()
    for f in files:
        df = pd.read_parquet(f, columns=["station_id"])
        ids.update(df["station_id"].dropna().astype(str).tolist())
    return ids


def load_master_ids(path: Path) -> set[str]:
    df = load_station_master(path)
    return set(df["station_id"].astype(str).tolist())


def load_rental_ids(path: Path) -> set[str]:
    df = load_rental_csv(path)
    rents = set(df["rent_station"].astype(str).tolist())
    returns = set(df["return_station"].astype(str).tolist())
    return rents | returns


def format_report(
    master_fmt: FormatStats,
    realtime_fmt: FormatStats,
    raw_compare: dict,
    norm_compare: dict,
    rental_block: str | None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("Station ID Cross-Source Match Report")
    lines.append("=" * 64)
    lines.append(
        f"master    : {master_fmt.total:>6,} ids  format={master_fmt.dominant:<12}"
        f"  examples={master_fmt.examples}"
    )
    lines.append(
        f"realtime  : {realtime_fmt.total:>6,} ids  format={realtime_fmt.dominant:<12}"
        f"  examples={realtime_fmt.examples}"
    )
    lines.append("")
    lines.append("--- raw (no normalization) ---")
    _block(lines, raw_compare)
    lines.append("")
    lines.append("--- normalized (bare-int → ST-N, strip zero-padding) ---")
    _block(lines, norm_compare)
    if rental_block:
        lines.append("")
        lines.append(rental_block)
    lines.append("=" * 64)
    if norm_compare["match_rate_of_master"] < 0.5:
        lines.append("WARN: <50% of master matched. Investigate ID format manually.")
    elif norm_compare["match_rate_of_master"] >= 0.95:
        lines.append("OK: ≥95% of master matched after normalization.")
    return "\n".join(lines)


def _block(lines: list[str], c: dict) -> None:
    pct_m = c["match_rate_of_master"] * 100
    pct_r = c["match_rate_of_realtime"] * 100
    lines.append(f"  matched      : {c['matched']:>6,}  ({pct_m:5.1f}% of master, {pct_r:5.1f}% of realtime)")
    lines.append(f"  master only  : {c['master_only']:>6,}  (closed since 2022?)")
    lines.append(f"  realtime only: {c['realtime_only']:>6,}  (added since 2022?)")


def run(master_path: Path, snapshot_glob: str, rental_path: Path | None) -> tuple[str, int]:
    master_ids = load_master_ids(master_path)
    realtime_ids = load_snapshot_ids(snapshot_glob)

    master_fmt = classify_format(master_ids)
    realtime_fmt = classify_format(realtime_ids)

    raw_cmp = compare_id_sets(master_ids, realtime_ids, normalizer=lambda s: str(s).strip())
    norm_cmp = compare_id_sets(master_ids, realtime_ids, normalizer=normalize_to_st)

    rental_block = None
    if rental_path is not None:
        rental_ids = load_rental_ids(rental_path)
        rental_fmt = classify_format(rental_ids)
        r_master = compare_id_sets(rental_ids, master_ids, normalize_to_st)
        r_realtime = compare_id_sets(rental_ids, realtime_ids, normalize_to_st)
        rental_block = (
            f"--- rental history vs others (file: {rental_path.name}) ---\n"
            f"  rental ids   : {rental_fmt.total:>6,} ids  format={rental_fmt.dominant}"
            f"  examples={rental_fmt.examples}\n"
            f"  vs master    : matched {r_master['matched']:,} "
            f"({r_master['match_rate_of_master']*100:.1f}% of rental)\n"
            f"  vs realtime  : matched {r_realtime['matched']:,} "
            f"({r_realtime['match_rate_of_master']*100:.1f}% of rental)"
        )

    report = format_report(master_fmt, realtime_fmt, raw_cmp, norm_cmp, rental_block)
    exit_code = 0 if norm_cmp["match_rate_of_master"] >= 0.5 else 1
    return report, exit_code


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--snapshot", type=str, required=True, help="parquet path or glob")
    p.add_argument("--rental", type=Path, default=None, help="optional rental history CSV")
    args = p.parse_args()

    report, code = run(args.master, args.snapshot, args.rental)
    print(report)
    sys.exit(code)


if __name__ == "__main__":
    main()

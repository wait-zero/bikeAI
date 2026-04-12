"""Tests for the station-ID cross-source matching tool."""
from __future__ import annotations

import pandas as pd

from bikeai.tools.check_id_match import (
    classify_format,
    compare_id_sets,
    normalize_to_st,
    run,
)


def test_classify_format_st_prefixed_dominant():
    fmt = classify_format(["ST-1", "ST-2", "ST-100"])
    assert fmt.dominant == "st-prefixed"
    assert fmt.st_prefixed == 3
    assert fmt.bare_int == 0


def test_classify_format_bare_int_dominant():
    fmt = classify_format(["1", "2", "100", "1000"])
    assert fmt.dominant == "bare-int"
    assert fmt.bare_int == 4
    assert fmt.st_prefixed == 0


def test_classify_format_handles_other_and_nan():
    fmt = classify_format(["ABC-1", "ST-2", "5", None, "nan"])
    assert fmt.total == 3  # None and "nan" filtered
    assert fmt.other == 1


def test_normalize_to_st_handles_three_formats():
    assert normalize_to_st("10") == "ST-10"
    assert normalize_to_st("ST-10") == "ST-10"
    assert normalize_to_st("ST-0010") == "ST-10"  # zero-padding stripped
    assert normalize_to_st("ABC") == "ABC"  # passthrough for unknown
    assert normalize_to_st("  10  ") == "ST-10"  # whitespace stripped


def test_compare_id_sets_perfect_match_no_normalization():
    a = {"ST-1", "ST-2", "ST-3"}
    b = {"ST-1", "ST-2", "ST-3"}
    out = compare_id_sets(a, b, normalizer=lambda s: s)
    assert out["matched"] == 3
    assert out["master_only"] == 0
    assert out["realtime_only"] == 0
    assert out["match_rate_of_master"] == 1.0


def test_compare_id_sets_format_mismatch_fixed_by_normalizer():
    """Master uses bare ints, realtime uses ST-prefixed. Normalizer should fix it."""
    master = {"1", "2", "3"}
    realtime = {"ST-1", "ST-2", "ST-3", "ST-4"}

    raw = compare_id_sets(master, realtime, normalizer=lambda s: s)
    assert raw["matched"] == 0  # raw match fails

    norm = compare_id_sets(master, realtime, normalizer=normalize_to_st)
    assert norm["matched"] == 3
    assert norm["master_only"] == 0
    assert norm["realtime_only"] == 1  # ST-4 is new


def test_compare_id_sets_partial_overlap():
    master = {"ST-1", "ST-2", "ST-3", "ST-99"}  # ST-99 closed
    realtime = {"ST-1", "ST-2", "ST-3", "ST-100", "ST-101"}  # 100/101 new
    out = compare_id_sets(master, realtime, normalizer=normalize_to_st)
    assert out["matched"] == 3
    assert out["master_only"] == 1
    assert out["realtime_only"] == 2
    assert out["match_rate_of_master"] == 0.75


def test_run_end_to_end_with_synthetic_files(tmp_path):
    """Write a synthetic master CSV + snapshot parquet, run the tool, check output."""
    # Master CSV (Korean headers, bare-int IDs — simulates 2022 master)
    master_csv = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "대여소번호": ["1", "2", "3", "999"],  # 999 = closed station
            "대여소명": ["A", "B", "C", "D"],
            "위도": [37.5, 37.51, 37.52, 37.53],
            "경도": [127.0, 127.01, 127.02, 127.03],
        }
    ).to_csv(master_csv, index=False, encoding="utf-8-sig")

    # Realtime snapshot (ST-prefixed IDs, includes new stations)
    snapshot = tmp_path / "snapshot.parquet"
    pd.DataFrame(
        {
            "station_id": ["ST-1", "ST-2", "ST-3", "ST-1000", "ST-1001"],  # 1000/1001 new
            "bike_count": [3, 5, 7, 2, 0],
            "ingested_at": pd.to_datetime(["2026-04-11 14:00"] * 5),
        }
    ).to_parquet(snapshot, index=False)

    report, code = run(master_csv, str(snapshot), rental_path=None)

    assert code == 0  # ≥50% match
    assert "matched" in report
    # 3 of 4 master matched (75%)
    assert "75.0% of master" in report
    # ST-prefixed format detected
    assert "st-prefixed" in report
    assert "bare-int" in report

#!/usr/bin/env python3
"""Full-window Bybit ETHUSDC order-book reconstruction and strict-prior
block join. Native exchange timestamp `cts` is the primary clock (documented
as the matching-engine production time; see
../../jfqa_public_reference_audit/ONE_DAY_REFERENCE_QA.md).

Streams all 31 daily archives in chronological order, maintaining running
book state across day boundaries (day files are not treated as independent
resets). Never loads the canonical AMM analysis panel - block timestamps only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from array import array
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

# Mechanical extension: START/END are CLI-provided (see main()) instead of
# hardcoded. No other line in this file differs from the discovery-sample
# original reconstruct_bybit_bbo.py.
START: date
END: date


def dates_between(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def load_blocks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    numbers, ts_ms = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            numbers.append(int(row["block_number"]))
            ts_ms.append(int(row["timestamp_unix"]) * 1000)
    numbers = np.asarray(numbers, dtype=np.int64)
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    order = np.argsort(ts_ms, kind="stable")
    numbers, ts_ms = numbers[order], ts_ms[order]
    if np.any(ts_ms[1:] < ts_ms[:-1]):
        raise ValueError("block timestamps not sortable into nondecreasing order")
    return numbers, ts_ms


def update_side(book: dict[float, float], updates: list[list[str]]) -> None:
    for level in updates:
        price = float(level[0])
        size = float(level[1])
        if size == 0.0:
            book.pop(price, None)
        else:
            book[price] = size


def open_one_member(path: Path):
    archive = zipfile.ZipFile(path)
    names = [x for x in archive.namelist() if not x.endswith("/")]
    if len(names) != 1:
        raise ValueError(f"expected one member in {path}, got {names}")
    return archive, archive.open(names[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                         help="RAW_REFERENCE_MANIFEST.json from download_full_reference.py")
    parser.add_argument("--block-headers", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-qa-json", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()

    global START, END
    START, END = args.start, args.end

    manifest = json.loads(args.manifest.read_text())
    path_by_date = {row["date"]: Path(row["path"]) for row in manifest
                    if row["venue"] == "Bybit" and row["symbol"] == "ETHUSDC"}

    block_numbers, blocks = load_blocks(args.block_headers)
    n = blocks.size
    block_cursor = 0
    block_quote_ts = np.full(n, -1, dtype=np.int64)
    block_bid = np.full(n, np.nan, dtype=np.float64)
    block_ask = np.full(n, np.nan, dtype=np.float64)

    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    last_ts: int | None = None
    last_bid = math.nan
    last_ask = math.nan
    min_ts = max_ts = None
    duplicate_ts = reversals = crossed = missing_bbo = rows = 0
    actions: dict[str, int] = {}
    diffs = array("q")
    previous_u: int | None = None
    previous_seq: int | None = None
    u_gap_count = seq_reversals = 0
    per_day_counts: dict[str, int] = {}

    days = dates_between(START, END)
    for day in days:
        path = path_by_date.get(day.isoformat())
        if path is None:
            raise FileNotFoundError(f"no manifest entry for Bybit ETHUSDC {day}")
        if not path.exists():
            raise FileNotFoundError(f"missing raw file for {day}: {path}")
        archive, member = open_one_member(path)
        day_rows = 0
        try:
            for raw in member:
                if not raw.strip():
                    continue
                record = orjson.loads(raw)
                ts = int(record["cts"])

                while block_cursor < n and blocks[block_cursor] <= ts:
                    if last_ts is not None:
                        block_quote_ts[block_cursor] = last_ts
                        block_bid[block_cursor] = last_bid
                        block_ask[block_cursor] = last_ask
                    block_cursor += 1

                if last_ts is not None:
                    diff = ts - last_ts
                    diffs.append(diff)
                    if diff == 0:
                        duplicate_ts += 1
                    elif diff < 0:
                        reversals += 1
                last_ts = ts
                min_ts = ts if min_ts is None else min(min_ts, ts)
                max_ts = ts if max_ts is None else max(max_ts, ts)

                action = record["type"]
                data = record["data"]
                u = int(data["u"])
                seq = int(data["seq"])
                if action == "snapshot":
                    bids.clear()
                    asks.clear()
                elif previous_u is not None and u != previous_u + 1:
                    u_gap_count += 1
                if previous_seq is not None and seq < previous_seq:
                    seq_reversals += 1
                previous_u, previous_seq = u, seq

                actions[action] = actions.get(action, 0) + 1
                update_side(bids, data["b"])
                update_side(asks, data["a"])
                if bids and asks:
                    last_bid, last_ask = max(bids), min(asks)
                    if last_bid >= last_ask:
                        crossed += 1
                else:
                    last_bid = last_ask = math.nan
                    missing_bbo += 1
                rows += 1
                day_rows += 1
        finally:
            member.close()
            archive.close()
        per_day_counts[day.isoformat()] = day_rows
        print(f"Bybit {day.isoformat()}: {day_rows:,} rows, cumulative blocks joined "
              f"{block_cursor:,}/{n:,}", flush=True)

    while block_cursor < n:
        if last_ts is not None:
            block_quote_ts[block_cursor] = last_ts
            block_bid[block_cursor] = last_bid
            block_ask[block_cursor] = last_ask
        block_cursor += 1

    valid = (block_quote_ts >= 0) & np.isfinite(block_bid) & np.isfinite(block_ask)
    ages_s = (blocks - block_quote_ts) / 1000.0
    diff_np = np.frombuffer(diffs, dtype=np.int64)
    nonneg_diff_s = diff_np[diff_np >= 0] / 1000.0

    table = pa.table({
        "block_number": block_numbers,
        "block_ts_ms": blocks,
        "bybit_quote_ts_ms": block_quote_ts,
        "bybit_bid": block_bid,
        "bybit_ask": block_ask,
        "bybit_mid": (block_bid + block_ask) / 2.0,
        "bybit_age_s": ages_s,
        "bybit_valid": valid,
    })
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_parquet)

    def q(x: np.ndarray, p: float):
        return float(np.quantile(x, p)) if x.size else None

    qa = {
        "venue": "Bybit", "symbol": "ETHUSDC", "window": {"start": START.isoformat(), "end": END.isoformat()},
        "timestamp_field": "cts",
        "total_rows": rows,
        "per_day_rows": per_day_counts,
        "actions": actions,
        "min_exchange_ts_ms": min_ts, "max_exchange_ts_ms": max_ts,
        "duplicate_timestamp_count": duplicate_ts,
        "timestamp_reversal_count": reversals,
        "crossed_or_locked_state_count": crossed,
        "missing_bbo_state_count": missing_bbo,
        "sequence_fields_present": True,
        "update_id_gap_count": u_gap_count,
        "sequence_reversal_count": seq_reversals,
        "event_spacing_seconds": {f"p{int(p*100):02d}": q(nonneg_diff_s, p) for p in (0.5, 0.9, 0.95, 0.99)},
        "block_count": int(n),
        "strict_prior_matched_blocks": int(valid.sum()),
        "reference_age_seconds": {f"p{int(p*100):02d}": q(ages_s[valid], p) for p in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "age_share": {f"le_{lim:g}s": float(np.mean(ages_s[valid] <= lim)) if valid.any() else None
                      for lim in (0.5, 1.0, 2.0, 3.0, 10.0, 30.0)},
        "coverage_gaps_over_3s_blocks": int(np.sum(ages_s[valid] > 3.0)),
        "coverage_gaps_over_10s_blocks": int(np.sum(ages_s[valid] > 10.0)),
        "coverage_gaps_over_30s_blocks": int(np.sum(ages_s[valid] > 30.0)),
    }
    args.output_qa_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_qa_json.write_text(json.dumps(qa, indent=2) + "\n")
    print(f"done: {rows:,} total rows, {int(valid.sum()):,}/{n:,} blocks matched")


if __name__ == "__main__":
    main()

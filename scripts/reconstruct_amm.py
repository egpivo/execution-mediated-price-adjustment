#!/usr/bin/env python3
"""Replay Tier A AMM state on deepfrk from raw events (recompute, not copy)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Run from repo copy on deepfrk; imports local amm_replay_engine
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_core.reconstruction.amm_replay_engine import (  # noqa: E402
    POOLS, load_events, merge_events, replay_block_range, PoolState,
)

SEEDS = {
    "ETH5": {"min_block": 25276142, "sqrt": "1926691449993816710131074074852089", "tick": 201989, "liq": 2673349088786403075},
    "BASE5": {"min_block": 47086927, "sqrt": "3257538177591802156302112", "tick": -201993, "liq": 761209445547306260},
    "BASE1": {"min_block": 47086927, "sqrt": "3257659542131727594436432", "tick": -201992, "liq": 37674331452060790},
}


def load_headers(path: Path) -> tuple[list[int], dict[int, int]]:
    blocks, ts_map = [], {}
    with path.open() as f:
        for row in csv.DictReader(f):
            b = int(row["block_number"])
            t = int(row["block_timestamp"])
            blocks.append(b)
            ts_map[b] = t
    blocks.sort()
    return blocks, ts_map


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool-key", required=True, choices=SEEDS)
    p.add_argument("--headers", type=Path, required=True)
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=50000)
    args = p.parse_args()

    import duckdb
    con = duckdb.connect()
    meta = POOLS[args.pool_key]
    seed_cfg = SEEDS[args.pool_key]
    blocks, ts_map = load_headers(args.headers)
    min_b, max_b = blocks[0], blocks[-1]
    assert min_b == seed_cfg["min_block"], f"expected seed block {seed_cfg['min_block']}, got {min_b}"

    swaps, mints, burns = [], [], []
    for month in ("06", "07", "08"):
        s, m, b = load_events(con, meta["address"], meta["chain"], "2026", month)
        swaps.extend(s for s in s if min_b <= s["block_number"] <= max_b)
        mints.extend(x for x in m if min_b <= x["block_number"] <= max_b)
        burns.extend(x for x in b if min_b <= x["block_number"] <= max_b)
    events_by_block = merge_events(swaps, mints, burns)

    block_ts = {b: ts_map.get(b) for b in range(min_b, max_b + 1)}
    seed = PoolState(seed_cfg["sqrt"], seed_cfg["tick"], seed_cfg["liq"])

    records = list(replay_block_range(events_by_block, block_ts, min_b, max_b, seed))
    # filter to header blocks only
    header_set = set(blocks)
    records = [r for r in records if r.block_number in header_set]

    table = pa.table({
        "block_number": pa.array([r.block_number for r in records], type=pa.int64()),
        "block_timestamp": pa.array([r.block_timestamp if r.block_timestamp is not None else -1 for r in records], type=pa.int64()),
        "has_real_timestamp": [r.has_real_timestamp for r in records],
        "pre_sqrtPriceX96": [r.pre_sqrtPriceX96 for r in records],
        "pre_tick": pa.array([r.pre_tick for r in records], type=pa.int32()),
        "pre_active_liquidity": [str(r.pre_active_liquidity) for r in records],
        "post_sqrtPriceX96": [r.post_sqrtPriceX96 for r in records],
        "post_tick": pa.array([r.post_tick for r in records], type=pa.int32()),
        "post_active_liquidity": [str(r.post_active_liquidity) for r in records],
        "n_swaps": pa.array([r.n_swaps for r in records], type=pa.int32()),
        "n_mints": pa.array([r.n_mints for r in records], type=pa.int32()),
        "n_burns": pa.array([r.n_burns for r in records], type=pa.int32()),
    })
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_parquet)
    print(json.dumps({"pool": args.pool_key, "rows": len(records), "min_block": min_b, "max_block": max_b,
                      "events": {"swap": len(swaps), "mint": len(mints), "burn": len(burns)}}, indent=2))


if __name__ == "__main__":
    main()

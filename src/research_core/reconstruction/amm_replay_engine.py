#!/usr/bin/env python3
"""Production AMM state replay from raw Swap/Mint/Burn events.

Validated lineage: EXACT_ENGINE_DISCOVERY_PARITY_PASS (374,300 rows, zero mismatches).
Uses on-chain event arguments directly — no counterfactual swap simulation.

Canonical ordering: block_number → transaction_index → log_index.
Deduplication: keep latest extraction_run per (block, tx_index, log_index).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from research_core.paths import raw_amm_event_lake_root

POOLS = {
    "ETH5": {"address": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", "chain": "ethereum"},
    "BASE5": {"address": "0xd0b53d9277642d899df5c87a3966a349a798f224", "chain": "base"},
    "BASE1": {"address": "0xb4cb800910b228ed3d0834cf79d697127bbb00e5", "chain": "base"},
}


@dataclass
class PoolState:
    sqrt_price_x96: str
    tick: int
    active_liquidity: int


@dataclass
class BlockRecord:
    block_number: int
    block_timestamp: int | None
    has_real_timestamp: bool
    pre_sqrtPriceX96: str
    pre_tick: int
    pre_active_liquidity: int
    post_sqrtPriceX96: str
    post_tick: int
    post_active_liquidity: int
    n_swaps: int
    n_mints: int
    n_burns: int


def partition_glob(chain: str, event: str, year: str, month: str) -> str:
    root = raw_amm_event_lake_root()
    if chain == "ethereum":
        return str(root / f"event={event}/year={year}/month={month}/part-*.parquet")
    return str(root / f"base/chain_id=8453/event={event}/year={year}/month={month}/part-*.parquet")


def load_events(con: duckdb.DuckDBPyConnection, pool_addr: str, chain: str,
                year: str, month: str) -> tuple[list[dict], list[dict], list[dict]]:
    swaps, mints, burns = [], [], []
    for ev, out in [("Swap", swaps), ("Mint", mints), ("Burn", burns)]:
        glob = partition_glob(chain, ev, year, month)
        q = f"""
        SELECT block_number, block_timestamp, transaction_index, log_index,
               extraction_run, decoded_json
        FROM read_parquet('{glob}')
        WHERE lower(pool_address) = '{pool_addr.lower()}'
        ORDER BY block_number, transaction_index, log_index, extraction_run
        """
        try:
            df = con.execute(q).fetchdf()
        except Exception:
            continue
        if df.empty:
            continue
        df = df.drop_duplicates(subset=["block_number", "transaction_index", "log_index"], keep="last")
        for _, r in df.iterrows():
            d = json.loads(r["decoded_json"])
            rec = {
                "block_number": int(r["block_number"]),
                "block_timestamp": int(r["block_timestamp"]) if r["block_timestamp"] is not None else None,
                "transaction_index": int(r["transaction_index"]),
                "log_index": int(r["log_index"]),
            }
            if ev == "Swap":
                rec.update(sqrtPriceX96=str(d["sqrtPriceX96"]), tick=int(d["tick"]), liquidity=str(d["liquidity"]))
                rec["etype"] = "swap"
            else:
                rec.update(tickLower=int(d["tickLower"]), tickUpper=int(d["tickUpper"]), amount=int(d["amount"]))
                rec["etype"] = ev.lower()
            out.append(rec)
    return swaps, mints, burns


def merge_events(swaps: list[dict], mints: list[dict], burns: list[dict]) -> dict[int, list[dict]]:
    all_ev = swaps + mints + burns
    all_ev.sort(key=lambda e: (e["block_number"], e["transaction_index"], e["log_index"]))
    by_block: dict[int, list[dict]] = {}
    for e in all_ev:
        by_block.setdefault(e["block_number"], []).append(e)
    return by_block


def apply_event(state: PoolState, ev: dict) -> PoolState:
    if ev["etype"] == "swap":
        return PoolState(str(ev["sqrtPriceX96"]), int(ev["tick"]), int(ev["liquidity"]))
    lo, hi, amt = ev["tickLower"], ev["tickUpper"], ev["amount"]
    liq = state.active_liquidity
    if lo <= state.tick < hi:
        liq = liq + amt if ev["etype"] == "mint" else liq - amt
    return PoolState(state.sqrt_price_x96, state.tick, liq)


def replay_block_range(
    events_by_block: dict[int, list[dict]],
    block_ts: dict[int, int | None],
    min_block: int,
    max_block: int,
    seed: PoolState,
) -> Iterator[BlockRecord]:
    state = seed
    for b in range(min_block, max_block + 1):
        pre = state
        n_swaps = n_mints = n_burns = 0
        ts = block_ts.get(b)
        for ev in events_by_block.get(b, []):
            if ev["etype"] == "swap":
                n_swaps += 1
            elif ev["etype"] == "mint":
                n_mints += 1
            else:
                n_burns += 1
            state = apply_event(state, ev)
        yield BlockRecord(
            block_number=b,
            block_timestamp=ts,
            has_real_timestamp=ts is not None,
            pre_sqrtPriceX96=pre.sqrt_price_x96,
            pre_tick=pre.tick,
            pre_active_liquidity=pre.active_liquidity,
            post_sqrtPriceX96=state.sqrt_price_x96,
            post_tick=state.tick,
            post_active_liquidity=state.active_liquidity,
            n_swaps=n_swaps,
            n_mints=n_mints,
            n_burns=n_burns,
        )


def write_panel_chunk(records: list[BlockRecord], out_path: Path) -> None:
    table = pa.table({
        "block_number": [r.block_number for r in records],
        "block_timestamp": [r.block_timestamp if r.block_timestamp is not None else -1 for r in records],
        "has_real_timestamp": [r.has_real_timestamp for r in records],
        "pre_sqrtPriceX96": [r.pre_sqrtPriceX96 for r in records],
        "pre_tick": [r.pre_tick for r in records],
        "pre_active_liquidity": [str(r.pre_active_liquidity) for r in records],
        "post_sqrtPriceX96": [r.post_sqrtPriceX96 for r in records],
        "post_tick": [r.post_tick for r in records],
        "post_active_liquidity": [str(r.post_active_liquidity) for r in records],
        "n_swaps": [r.n_swaps for r in records],
        "n_mints": [r.n_mints for r in records],
        "n_burns": [r.n_burns for r in records],
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)

#!/usr/bin/env python3
"""Combine the two frozen per-venue block-joined quote streams into the
composite HF reference, per sprint spec section 8:

    log X*_n = 0.5 * (log mid_Bybit,n + log mid_OKX,n)

No estimated weights, no outcome-dependent weighting. A block's composite is
valid only if BOTH venues independently satisfy the strict-prior join and the
3-second freshness gate (section 7: no one-venue fallback).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FRESHNESS_S = 3.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bybit-parquet", type=Path, required=True)
    parser.add_argument("--okx-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    b = pq.read_table(args.bybit_parquet).to_pydict()
    o = pq.read_table(args.okx_parquet).to_pydict()
    assert b["block_number"] == o["block_number"], "block ordering mismatch between venue tables"

    block_number = np.asarray(b["block_number"], dtype=np.int64)
    block_ts_ms = np.asarray(b["block_ts_ms"], dtype=np.int64)
    n = block_number.size

    by_bid = np.asarray(b["bybit_bid"]); by_ask = np.asarray(b["bybit_ask"])
    by_age = np.asarray(b["bybit_age_s"]); by_valid = np.asarray(b["bybit_valid"], dtype=bool)
    ox_bid = np.asarray(o["okx_bid"]); ox_ask = np.asarray(o["okx_ask"])
    ox_age = np.asarray(o["okx_age_s"]); ox_valid = np.asarray(o["okx_valid"], dtype=bool)

    by_crossed = by_valid & (by_bid >= by_ask)
    ox_crossed = ox_valid & (ox_bid >= ox_ask)
    by_nonpositive = by_valid & ((by_bid <= 0) | (by_ask <= 0))
    ox_nonpositive = ox_valid & ((ox_bid <= 0) | (ox_ask <= 0))

    by_usable = by_valid & ~by_crossed & ~by_nonpositive
    ox_usable = ox_valid & ~ox_crossed & ~ox_nonpositive

    log_mid_bybit = np.where(by_usable, np.log((by_bid + by_ask) / 2.0), np.nan)
    log_mid_okx = np.where(ox_usable, np.log((ox_bid + ox_ask) / 2.0), np.nan)

    both_quoted = by_usable & ox_usable
    both_fresh = both_quoted & (by_age <= FRESHNESS_S) & (ox_age <= FRESHNESS_S)

    xstar_hf_log = np.where(both_fresh, 0.5 * (log_mid_bybit + log_mid_okx), np.nan)
    dispersion = np.where(both_quoted, np.abs(log_mid_bybit - log_mid_okx), np.nan)
    max_age_s = np.where(both_quoted, np.maximum(by_age, ox_age), np.nan)

    impossible = (~by_valid | by_crossed | by_nonpositive) | (~ox_valid | ox_crossed | ox_nonpositive)

    table = pa.table({
        "block_number": block_number,
        "block_ts_ms": block_ts_ms,
        "bybit_quote_ts_ms": b["bybit_quote_ts_ms"],
        "bybit_bid": by_bid, "bybit_ask": by_ask, "bybit_mid": b["bybit_mid"],
        "bybit_age_s": by_age, "bybit_valid": by_usable, "bybit_crossed": by_crossed,
        "okx_quote_ts_ms": o["okx_quote_ts_ms"],
        "okx_bid": ox_bid, "okx_ask": ox_ask, "okx_mid": o["okx_mid"],
        "okx_age_s": ox_age, "okx_valid": ox_usable, "okx_crossed": ox_crossed,
        "log_mid_bybit": log_mid_bybit, "log_mid_okx": log_mid_okx,
        "cross_venue_log_dispersion": dispersion,
        "max_component_age_s": max_age_s,
        "xstar_hf_log": xstar_hf_log,
        "xstar_hf_valid": both_fresh,
    })
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_parquet)

    def q(x: np.ndarray, p: float):
        x = x[np.isfinite(x)]
        return float(np.quantile(x, p)) if x.size else None

    manifest = {
        "freshness_threshold_s": FRESHNESS_S,
        "composite_formula": "log X*_n = 0.5*(log mid_Bybit,n + log mid_OKX,n), no estimated weights",
        "canonical_block_count": int(n),
        "both_venue_quoted_count": int(both_quoted.sum()),
        "both_venue_le_3s_count": int(both_fresh.sum()),
        "retained_share": float(both_fresh.mean()),
        "share_impossible_bookstate": float(impossible.mean()),
        "max_component_age_seconds": {f"p{int(p*100):02d}": q(max_age_s[both_quoted], p)
                                       for p in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "bybit_age_seconds_conditional_on_quoted": {f"p{int(p*100):02d}": q(by_age[by_usable], p)
                                                     for p in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "okx_age_seconds_conditional_on_quoted": {f"p{int(p*100):02d}": q(ox_age[ox_usable], p)
                                                   for p in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "cross_venue_log_dispersion": {
            **{f"p{int(p*100):02d}": q(dispersion, p) for p in (0.5, 0.9, 0.95, 0.99)},
            "max": float(np.nanmax(dispersion)) if np.isfinite(dispersion).any() else None,
        },
        "bybit_share_crossed_or_locked": float(by_crossed.mean()),
        "okx_share_crossed_or_locked": float(ox_crossed.mean()),
        "bybit_share_nonpositive": float(by_nonpositive.mean()),
        "okx_share_nonpositive": float(ox_nonpositive.mean()),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in
                       ("max_component_age_seconds", "bybit_age_seconds_conditional_on_quoted",
                        "okx_age_seconds_conditional_on_quoted")}, indent=2))


if __name__ == "__main__":
    main()

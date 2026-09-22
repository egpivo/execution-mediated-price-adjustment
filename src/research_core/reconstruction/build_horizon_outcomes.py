#!/usr/bin/env python3
"""Multi-horizon outcome construction with explicit naming.

Never aliases same_block to 12s. X* held fixed at pre-state for all horizons.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HORIZONS = {
    "12s": 12.0,
    "30s": 30.0,
    "60s": 60.0,
}


def build_horizon(
    block_number: np.ndarray,
    block_timestamp: np.ndarray,
    xA_post: np.ndarray,
    abs_d_pre: np.ndarray,
    xstar_log_fixed: np.ndarray,
    retained: np.ndarray,
    horizon_s: float,
) -> dict[str, np.ndarray]:
    order = np.argsort(block_number, kind="stable")
    bn_s = block_number[order]
    ts_s = block_timestamp[order].astype(np.float64)
    xA_post_s = xA_post[order]
    n = bn_s.size

    # Per-row forward scan in block order. Timestamps need not be monotone
    # (merged headers can regress); horizon = last block j>=i with ts[j]<=ts[i]+h.
    end_idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        target = ts_s[i] + horizon_s
        j = i
        while j + 1 < n and ts_s[j + 1] <= target:
            j += 1
        end_idx[i] = j
    horizon_bn_s = bn_s[end_idx]
    horizon_ts_s = ts_s[end_idx]
    xA_end_s = xA_post_s[end_idx]
    n_blocks_s = end_idx - np.arange(n) + 1

    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)

    xA_end = xA_end_s[inv]
    horizon_bn = horizon_bn_s[inv]
    horizon_ts = horizon_ts_s[inv]
    n_blocks = n_blocks_s[inv]

    d_post_h = np.where(retained & np.isfinite(xA_end), np.abs(xstar_log_fixed - xA_end), np.nan)
    correction = np.where(retained, abs_d_pre - d_post_h, np.nan)
    corrective = retained & (d_post_h < abs_d_pre)

    return {
        "horizon_post_block": horizon_bn.astype(np.int64),
        "horizon_post_timestamp": horizon_ts.astype(np.int64),
        "n_blocks_in_window": n_blocks.astype(np.int32),
        "d_post_horizon": d_post_h,
        "correction": correction,
        "corrective": corrective,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-parquet", type=Path, required=True,
                   help="Silver/Gold partial with block_number, timestamps, xA_post, abs_d_pre, xstar columns")
    p.add_argument("--lineage", choices=["legacy", "hf"], required=True)
    p.add_argument("--output-parquet", type=Path, required=True)
    args = p.parse_args()

    t = pq.read_table(args.input_parquet).to_pydict()
    bn = np.asarray(t["block_number"], dtype=np.int64)
    ts = np.asarray(t["block_timestamp"], dtype=np.int64)
    xA_post = np.asarray(t["xA_post"], dtype=np.float64)

    if args.lineage == "legacy":
        retained = np.asarray(t["retained_legacy"], dtype=bool)
        abs_d_pre = np.asarray(t["abs_d_pre_legacy"], dtype=np.float64)
        xstar = np.asarray(t["xstar_legacy_log"], dtype=np.float64)
        prefix = ""
    else:
        retained = np.asarray(t["retained_hf"], dtype=bool)
        abs_d_pre = np.asarray(t["d_pre_hf"], dtype=np.float64)
        xstar = np.asarray(t["xstar_hf_log"], dtype=np.float64)
        prefix = "hf_"

    out_cols: dict[str, np.ndarray] = {"block_number": bn}
    for label, seconds in HORIZONS.items():
        h = build_horizon(bn, ts, xA_post, abs_d_pre, xstar, retained, seconds)
        sfx = label.replace("s", "s")
        out_cols[f"correction_{prefix}{sfx}"] = h["correction"]
        out_cols[f"corrective_{prefix}{sfx}"] = h["corrective"]
        out_cols[f"horizon_post_block_{prefix}{sfx}"] = h["horizon_post_block"]
        out_cols[f"horizon_post_timestamp_{prefix}{sfx}"] = h["horizon_post_timestamp"]
        out_cols[f"n_blocks_in_{prefix}{sfx}_window"] = h["n_blocks_in_window"]

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out_cols), args.output_parquet)
    print(f"Wrote horizon outcomes ({args.lineage}) -> {args.output_parquet}")


if __name__ == "__main__":
    main()

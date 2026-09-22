#!/usr/bin/env python3
"""Assemble Tier A Gold panel from recomputed Silver (dual lineage)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from research_core.reconstruction.build_horizon_outcomes import build_horizon
from research_core.methods.estimand import (
    POOL_META,
    amm_usdc_per_eth,
    build_x_support_flags,
    gamma_fee,
    hf_same_block_outcomes,
    legacy_same_block_outcomes,
)
from research_core.paths import legacy_bridge_eth5_parquet

DISCOVERY_START = datetime(2026, 7, 6, 0, 0, 11, tzinfo=timezone.utc).timestamp()
CONFIRMATION_END = datetime(2026, 7, 5, 23, 59, 59, tzinfo=timezone.utc).timestamp()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_role(ts: np.ndarray) -> np.ndarray:
    roles = np.full(ts.shape, "LEGACY_DISCOVERY", dtype=object)
    roles[ts <= CONFIRMATION_END] = "LEGACY_CONFIRMATION"
    return roles


def calendar_fields(ts: np.ndarray) -> dict[str, np.ndarray]:
    dt = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts]
    return {
        "utc_hour": np.array([d.hour for d in dt], dtype=np.int16),
        "utc_dow": np.array([d.weekday() for d in dt], dtype=np.int16),
        "weekend": np.array([d.weekday() >= 5 for d in dt]),
        "month_id": np.array([f"{d.year:04d}-{d.month:02d}" for d in dt]),
        "quarter_id": np.array([f"{d.year:04d}-Q{(d.month-1)//3+1}" for d in dt]),
        "year": np.array([d.year for d in dt], dtype=np.int16),
        "hour_of_week": np.array([d.weekday() * 24 + d.hour for d in dt], dtype=np.int16),
    }


def join_xstar_hf(panel_bn: np.ndarray, xstar_path: Path) -> dict[str, np.ndarray]:
    xs = pq.read_table(xstar_path).to_pydict()
    xbn = np.asarray(xs["block_number"], dtype=np.int64)
    order = np.argsort(xbn)
    xbn_s = xbn[order]
    lookup = np.searchsorted(xbn_s, panel_bn)
    ok = (lookup < xbn_s.size) & (xbn_s[np.clip(lookup, 0, xbn_s.size - 1)] == panel_bn)
    idx = order[np.clip(lookup, 0, xbn_s.size - 1)]

    def al(col, default=np.nan):
        arr = np.asarray(xs[col])
        out = np.full(panel_bn.size, default, dtype=arr.dtype if np.isscalar(default) else float)
        if arr.dtype == bool:
            out = np.zeros(panel_bn.size, dtype=bool)
        out[ok] = arr[idx[ok]]
        return out

    return {
        "xstar_hf_log": al("xstar_hf_log", np.nan).astype(np.float64),
        "xstar_hf_valid": al("xstar_hf_valid", False),
        "bybit_age_s": al("bybit_age_s", np.nan).astype(np.float64),
        "okx_age_s": al("okx_age_s", np.nan).astype(np.float64),
        "max_component_age_s": al("max_component_age_s", np.nan).astype(np.float64),
        "cross_venue_log_dispersion": al("cross_venue_log_dispersion", np.nan).astype(np.float64),
    }


def join_legacy_recomputed(panel_bn: np.ndarray, legacy_path: Path) -> dict[str, np.ndarray]:
    n = panel_bn.size
    defaults = {
        "xstar_legacy_log": np.full(n, np.nan),
        "abs_d_pre_legacy": np.full(n, np.nan),
        "gross_arb_usd": np.full(n, np.nan),
        "net_usd": np.full(n, np.nan),
        "net_positive": np.zeros(n, dtype=bool),
        "correction_same_block": np.full(n, np.nan),
        "corrective_same_block": np.zeros(n, dtype=bool),
        "correction_12s": np.full(n, np.nan),
        "corrective_12s": np.zeros(n, dtype=bool),
        "retained_legacy": np.zeros(n, dtype=bool),
    }
    if not legacy_path.exists():
        return defaults
    leg = pq.read_table(legacy_path).to_pydict()
    lbn = np.asarray(leg["block_number"], dtype=np.int64)
    order = np.argsort(lbn)
    lbn_s = lbn[order]
    lookup = np.searchsorted(lbn_s, panel_bn)
    ok = (lookup < lbn_s.size) & (lbn_s[np.clip(lookup, 0, lbn_s.size - 1)] == panel_bn)
    idx = order[np.clip(lookup, 0, lbn_s.size - 1)]
    for col in defaults:
        if col == "retained_legacy":
            continue
        if col in leg:
            arr = np.asarray(leg[col])
            if arr.dtype == bool:
                defaults[col][ok] = arr[idx[ok]]
            else:
                defaults[col][ok] = arr[idx[ok]].astype(float) if col != "net_positive" else arr[idx[ok]]
    defaults["retained_legacy"] = ok & np.asarray(leg.get("eligible_legacy", [False]*len(lbn)))[idx] if ok.any() else defaults["retained_legacy"]
    if "eligible_legacy" in leg:
        el = np.asarray(leg["eligible_legacy"], dtype=bool)
        defaults["retained_legacy"] = np.zeros(n, dtype=bool)
        defaults["retained_legacy"][ok] = el[idx[ok]]
    return defaults


def build_pool(pool_key: str, amm_path: Path, xstar_path: Path, chain_id: int, legacy_path: Path | None = None) -> pa.Table:
    t = pq.read_table(amm_path).to_pydict()
    bn = np.asarray(t["block_number"], dtype=np.int64)
    ts = np.asarray(t["block_timestamp"], dtype=np.int64)
    has_ts = np.asarray(t["has_real_timestamp"], dtype=bool)
    sqrt_pre = t["pre_sqrtPriceX96"]
    sqrt_post = t["post_sqrtPriceX96"]
    p_pre = amm_usdc_per_eth(sqrt_pre, pool_key)
    p_post = amm_usdc_per_eth(sqrt_post, pool_key)
    xA_pre = np.log(p_pre)
    xA_post = np.log(p_post)
    n_swaps = np.asarray(t["n_swaps"], dtype=np.int64)

    hf_ref = join_xstar_hf(bn, xstar_path)
    panel_eligible = has_ts & np.isfinite(xA_pre) & np.isfinite(xA_post)
    retained_hf = panel_eligible & hf_ref["xstar_hf_valid"]
    hf_out = hf_same_block_outcomes(hf_ref["xstar_hf_log"], xA_pre, xA_post, retained_hf, pool_key)
    h12_hf = build_horizon(bn, ts, xA_post, hf_out["d_pre_hf"], hf_ref["xstar_hf_log"], retained_hf, 12.0)
    flags = build_x_support_flags(hf_out["x"])

    cols = {
        "pool_key": np.full(bn.size, pool_key),
        "chain_id": np.full(bn.size, chain_id, dtype=np.int32),
        "block_number": bn,
        "block_timestamp": ts,
        "has_real_timestamp": has_ts,
        "sample_role": sample_role(ts),
        "panel_eligible": panel_eligible,
        "retained_hf": retained_hf,
        "n_swaps": n_swaps,
        "has_focal_swap": n_swaps > 0,
        "pre_sqrtPriceX96": sqrt_pre,
        "post_sqrtPriceX96": sqrt_post,
        "pre_active_liquidity": t["pre_active_liquidity"],
        "post_active_liquidity": t["post_active_liquidity"],
        "p_amm_pre": p_pre,
        "p_amm_post": p_post,
        "xA_pre": xA_pre,
        "xA_post": xA_post,
        "reference_era_id_hf": np.full(bn.size, "HF_BYBIT_OKX_USDC"),
        **hf_ref,
        **hf_out,
        "correction_hf_12s": h12_hf["correction"],
        "corrective_hf_12s": h12_hf["corrective"],
        **flags,
        **calendar_fields(ts),
    }
    if pool_key == "ETH5" and legacy_path:
        cols.update(join_legacy_recomputed(bn, legacy_path))
    return pa.table(cols)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--silver-amm-dir", type=Path, required=True)
    p.add_argument("--silver-ref-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for pool_key, chain_id, chain_tag in [("ETH5", 1, "eth"), ("BASE5", 8453, "base"), ("BASE1", 8453, "base")]:
        amm = args.silver_amm_dir / f"{pool_key.lower()}_tier_a_amm.parquet"
        xs = args.silver_ref_dir / f"xstar_hf_{chain_tag}_tier_a.parquet"
        if not amm.exists() or not xs.exists():
            print(f"SKIP {pool_key}: missing {amm} or {xs}")
            continue
        legacy = args.silver_amm_dir.parent / "legacy" / "eth5_legacy_discovery.parquet" if pool_key == "ETH5" else None
        out = args.output_dir / f"{pool_key.lower()}_tier_a_gold.parquet"
        table = build_pool(pool_key, amm, xs, chain_id, legacy)
        pq.write_table(table, out)
        manifest[pool_key] = {"rows": table.num_rows, "path": str(out), "sha256": sha256_file(out)}
        print(pool_key, manifest[pool_key])
    (args.output_dir / "TIER_A_GOLD_BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

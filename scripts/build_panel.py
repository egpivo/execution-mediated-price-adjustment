#!/usr/bin/env python3
"""Builds the confirmatory-long PRE-window response panel directly from the
raw-event block panel produced by build_pre_block_panel.py (validated in
EXACT_ENGINE_PARITY_AUDIT.md) plus the frozen xstar_hf composite.

Same join/definition logic as
jfqa_service_primitives/code/build_response_panel_generic.py, minus the
`legacy_*` columns: this window was never part of the original ad-hoc bridge
build, so there is no legitimate legacy comparison object to carry through
(fabricating placeholder legacy values would be worse than omitting them).
x, correction_hf, corrective_hf, eligible, retained_hf are computed
identically to the frozen script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# CODE_REVIEW.md repair #11: this script previously reimplemented
# amm_usdc_per_eth/gamma_fee/d_pre/d_post/x/C inline, independently of
# research_core.methods.estimand -- a duplicate that could silently diverge.
# Both now call the same functions; this is the single source of truth.
from research_core.methods.estimand import (  # noqa: E402
    POOL_META,
    amm_usdc_per_eth,
    gamma_fee,
    hf_same_block_outcomes,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-panel", type=Path, required=True)
    parser.add_argument("--pool-key", required=True, choices=list(POOL_META))
    parser.add_argument("--xstar-hf-parquet", type=Path, required=True)
    parser.add_argument("--fee-fraction", type=float, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    # --fee-fraction is a fee-tier FACT (which of ETH5's 5bp / BASE1's 1bp
    # tier this pool is), not a tuning knob -- validate it against the
    # single source of truth in estimand.POOL_META rather than silently
    # trusting a CLI value that could drift out of sync.
    expected_fee = POOL_META[args.pool_key]["fee_fraction"]
    if not np.isclose(args.fee_fraction, expected_fee):
        raise SystemExit(
            f"--fee-fraction {args.fee_fraction} does not match the known fee "
            f"fraction {expected_fee} for {args.pool_key} in "
            f"research_core.methods.estimand.POOL_META"
        )
    gfee = gamma_fee(args.pool_key)

    panel = pq.read_table(args.block_panel).to_pydict()
    xstar = pq.read_table(args.xstar_hf_parquet,
                           columns=["block_number", "xstar_hf_log", "xstar_hf_valid",
                                    "max_component_age_s", "cross_venue_log_dispersion",
                                    "bybit_age_s", "okx_age_s"]).to_pydict()

    panel_bn = np.asarray(panel["block_number"], dtype=np.int64)
    has_real_ts = np.asarray(panel["has_real_timestamp"], dtype=bool)
    xstar_bn = np.asarray(xstar["block_number"], dtype=np.int64)

    xstar_order = np.argsort(xstar_bn, kind="stable")
    xstar_bn_sorted = xstar_bn[xstar_order]
    lookup = np.searchsorted(xstar_bn_sorted, panel_bn)
    in_xstar = (lookup < xstar_bn_sorted.size) & (xstar_bn_sorted[np.clip(lookup, 0, xstar_bn_sorted.size - 1)] == panel_bn)
    xstar_idx = xstar_order[np.clip(lookup, 0, xstar_bn_sorted.size - 1)]

    def aligned(col: str, dtype) -> np.ndarray:
        arr = np.asarray(xstar[col])
        out = np.full(panel_bn.size, np.nan, dtype=dtype)
        out[in_xstar] = arr[xstar_idx[in_xstar]]
        return out

    xstar_hf_log = aligned("xstar_hf_log", np.float64)
    xstar_hf_valid = np.zeros(panel_bn.size, dtype=bool)
    xstar_hf_valid[in_xstar] = np.asarray(xstar["xstar_hf_valid"])[xstar_idx[in_xstar]].astype(bool)

    xA_pre = np.log(amm_usdc_per_eth(panel["pre_sqrtPriceX96"], args.pool_key))
    xA_post = np.log(amm_usdc_per_eth(panel["post_sqrtPriceX96"], args.pool_key))
    n_swaps = np.asarray(panel["n_swaps"], dtype=np.int64)

    eligible = has_real_ts & np.isfinite(xA_pre) & np.isfinite(xA_post)
    state_valid = eligible
    retained = state_valid & xstar_hf_valid

    hf_out = hf_same_block_outcomes(xstar_hf_log, xA_pre, xA_post, retained, args.pool_key)
    d_pre_hf = hf_out["d_pre_hf"]
    d_post_hf = hf_out["d_post_hf"]
    x = hf_out["x"]
    correction_hf = hf_out["correction_hf_same_block"]
    corrective_bool = hf_out["corrective_hf_same_block"]
    corrective_hf = np.where(retained, corrective_bool.astype(np.float64), np.nan)
    excess_pre_hf = np.where(retained, np.maximum(d_pre_hf - gfee, 0.0), np.nan)

    table = pa.table({
        "block_number": panel_bn,
        "block_timestamp": panel["block_timestamp"],
        "has_real_timestamp": has_real_ts,
        "retained_hf": retained,
        "panel_eligible": eligible,
        "xstar_hf_valid": xstar_hf_valid,
        "n_swaps": n_swaps,
        "has_focal_swap": n_swaps > 0,
        "xA_pre": xA_pre, "xA_post": xA_post,
        "xstar_hf_log": xstar_hf_log,
        "d_pre_hf": d_pre_hf, "d_post_hf": d_post_hf,
        "x": x,
        "correction_hf": correction_hf,
        "corrective_hf": corrective_hf,
        "excess_pre_hf": excess_pre_hf,
        "max_component_age_s": aligned("max_component_age_s", np.float64),
        "cross_venue_log_dispersion": aligned("cross_venue_log_dispersion", np.float64),
        "bybit_age_s": aligned("bybit_age_s", np.float64),
        "okx_age_s": aligned("okx_age_s", np.float64),
    })
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output_parquet)

    manifest = {
        "pool_key": args.pool_key,
        "fee_fraction": args.fee_fraction,
        "gamma_fee": gfee,
        "block_panel_path": str(args.block_panel),
        "block_panel_sha256": sha256_file(args.block_panel),
        "xstar_hf_path": str(args.xstar_hf_parquet),
        "xstar_hf_sha256": sha256_file(args.xstar_hf_parquet),
        "complete_block_n": int(panel_bn.size),
        "has_real_timestamp_n": int(has_real_ts.sum()),
        "panel_eligible_n": int(eligible.sum()),
        "xstar_hf_valid_n": int(xstar_hf_valid.sum()),
        "retained_hf_n": int(retained.sum()),
        "retained_share_of_complete": float(retained.mean()),
        "swap_rows_retained": int((retained & (n_swaps > 0)).sum()),
        "no_swap_rows_retained": int((retained & (n_swaps == 0)).sum()),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

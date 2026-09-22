#!/usr/bin/env python3
"""Build Gold panel row with dual lineage: legacy bridge + HF service objects.

Preserves manuscript executable-opportunity chain AND fee-normalized service
semantics. Never aliases response horizons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

Q96 = 2 ** 96

POOL_META = {
    "ETH5": {"token0": "USDC", "token0_decimals": 6, "token1_decimals": 18, "fee_fraction": 0.0005},
    "BASE5": {"token0": "WETH", "token0_decimals": 18, "token1_decimals": 6, "fee_fraction": 0.0005},
    "BASE1": {"token0": "WETH", "token0_decimals": 18, "token1_decimals": 6, "fee_fraction": 0.0001},
}


def amm_usdc_per_eth(sqrt_str: str | np.ndarray, pool_key: str) -> np.ndarray:
    meta = POOL_META[pool_key]
    if isinstance(sqrt_str, str):
        s = np.array([int(sqrt_str)], dtype=np.float64)
    else:
        s = np.array([int(x) for x in sqrt_str], dtype=np.float64)
    if meta["token0"] == "USDC":
        px = (float(Q96) / s) ** 2 * (10 ** (meta["token1_decimals"] - meta["token0_decimals"]))
    else:
        px = (s / float(Q96)) ** 2 * (10 ** (meta["token0_decimals"] - meta["token1_decimals"]))
    return px


def gamma_fee(pool_key: str) -> float:
    f = POOL_META[pool_key]["fee_fraction"]
    return -math.log(1 - f)


def legacy_same_block_outcomes(
    xstar_log: np.ndarray,
    xA_pre: np.ndarray,
    xA_post: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    d_pre_signed = np.where(valid, xstar_log - xA_pre, np.nan)
    abs_d_pre = np.where(valid, np.abs(d_pre_signed), np.nan)
    d_post_signed = np.where(valid, xstar_log - xA_post, np.nan)
    abs_d_post = np.where(valid, np.abs(d_post_signed), np.nan)
    correction_same_block = np.where(valid, abs_d_pre - abs_d_post, np.nan)
    corrective_same_block = valid & (abs_d_post < abs_d_pre)
    return {
        "d_pre_signed_legacy": d_pre_signed,
        "abs_d_pre_legacy": abs_d_pre,
        "d_post_signed_legacy": d_post_signed,
        "correction_same_block": correction_same_block,
        "corrective_same_block": corrective_same_block,
    }


def hf_same_block_outcomes(
    xstar_hf_log: np.ndarray,
    xA_pre: np.ndarray,
    xA_post: np.ndarray,
    retained_hf: np.ndarray,
    pool_key: str,
) -> dict[str, np.ndarray]:
    g = gamma_fee(pool_key)
    d_pre_hf = np.where(retained_hf, np.abs(xstar_hf_log - xA_pre), np.nan)
    d_post_hf = np.where(retained_hf, np.abs(xstar_hf_log - xA_post), np.nan)
    x = d_pre_hf / g
    correction_hf = d_pre_hf - d_post_hf
    corrective_hf = retained_hf & (d_post_hf < d_pre_hf)
    return {
        "gamma_fee": np.full_like(d_pre_hf, g),
        "d_pre_hf": d_pre_hf,
        "d_post_hf": d_post_hf,
        "x": x,
        "correction_hf_same_block": correction_hf,
        "corrective_hf_same_block": corrective_hf,
    }


# Frozen 10-bin schedule (CODE_REVIEW.md repair #11): this previously
# merged [0,.25) and [.25,.5) into a single "x_lt_0_5" bin, disagreeing with
# the 10-bin schedule actually used everywhere the estimand is analyzed
# (src/research_core/methods/rmt.py::BIN_EDGES/BIN_LABELS,
# scripts/qa_decomposition_identity.py::BINS). No stored Gold-panel output
# using the old 9-bin schema is checked into this repo or depended on by
# any frozen number -- assemble_gold_panel.py could not even run before
# this repair pass (broken imports, fixed separately) -- so correcting the
# schema here changes no reported scientific value.
X_SUPPORT_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, float("inf"))
X_SUPPORT_BIN_FLAG_NAMES = (
    "x_0_0_25", "x_0_25_0_5", "x_0_5_0_75", "x_0_75_1",
    "x_1_1_25", "x_1_25_1_5", "x_1_5_2", "x_2_3", "x_3_5", "x_gt_5",
)


def build_x_support_flags(x: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean membership flags for the frozen 10-bin schedule:
    [0,.25) [.25,.5) [.5,.75) [.75,1) [1,1.25) [1.25,1.5) [1.5,2) [2,3)
    [3,5) [5,inf). Half-open, lower-inclusive; NaN/inf/negative x are
    excluded from every flag (no silent clipping)."""
    finite = np.isfinite(x) & (x >= 0)
    edges = X_SUPPORT_BIN_EDGES
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(X_SUPPORT_BIN_FLAG_NAMES):
        lo, hi = edges[i], edges[i + 1]
        if np.isinf(hi):
            out[name] = finite & (x >= lo)
        else:
            out[name] = finite & (x >= lo) & (x < hi)
    return out



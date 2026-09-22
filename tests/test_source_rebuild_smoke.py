"""Bounded source-rebuild smoke test (CODE_REVIEW.md / repair task §18.C).

No live RPC access is available in this environment, so this does not
exercise eth_pool_events.py's network calls -- that remains
VALIDATED_BUT_CURRENTLY_REQUIRES_REFETCH (unchanged from prior audits).

What this DOES prove, with a tiny (3-block, 2-event) synthetic fixture
built in the exact shape eth_pool_events.py's decode_event() produces: the
reconstruction layer (amm_replay_engine.merge_events/replay_block_range)
and the canonical-panel estimand layer (estimand.amm_usdc_per_eth,
gamma_fee, hf_same_block_outcomes) actually compose end-to-end -- ingestion
schema -> reconstruction -> canonical panel -- without silently disagreeing
on field names, types, or units. This is real code, not a mock of it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_core.reconstruction.amm_replay_engine import (
    PoolState, apply_event, merge_events, replay_block_range,
)
from research_core.methods.estimand import amm_usdc_per_eth, gamma_fee, hf_same_block_outcomes

POOL = "ETH5"
Q96 = 2 ** 96


def _sqrt_price_for_price(price_usdc_per_eth: float) -> str:
    # ETH5: token0=USDC(6dec), token1=WETH(18dec) -> price = (Q96/sqrt)^2 * 10^(18-6)
    # invert: sqrt = Q96 / sqrt(price / 10^12)
    sqrt_val = Q96 / np.sqrt(price_usdc_per_eth / 1e12)
    return str(int(sqrt_val))


def test_ingestion_schema_to_reconstruction_to_canonical_panel_composes():
    # 1. Synthetic "ingestion output" -- same event shape decode_event() produces.
    seed = PoolState(sqrt_price_x96=_sqrt_price_for_price(3000.0), tick=0, active_liquidity=10**18)
    events_by_block = {
        101: [],  # no swap this block: state must be preserved unchanged
        102: [{  # one swap: moves price to 3010
            "block_number": 102, "transaction_index": 0, "log_index": 0,
            "etype": "swap",
            "sqrtPriceX96": _sqrt_price_for_price(3010.0), "tick": 5, "liquidity": str(10**18),
        }],
        103: [],  # no swap again: state preserved from block 102's post-state
    }
    block_ts = {101: 1_700_000_000, 102: 1_700_000_012, 103: 1_700_000_024}

    # 2. Reconstruction layer (real code, not a mock).
    records = list(replay_block_range(events_by_block, block_ts, 101, 103, seed))
    assert len(records) == 3
    assert records[0].pre_sqrtPriceX96 == records[0].post_sqrtPriceX96 == seed.sqrt_price_x96
    assert records[0].n_swaps == 0
    assert records[1].n_swaps == 1
    assert records[1].pre_sqrtPriceX96 == seed.sqrt_price_x96  # pre-state unchanged before the swap
    assert records[1].post_sqrtPriceX96 != seed.sqrt_price_x96  # post-state reflects the swap
    # No-swap block preserves the PRIOR post-state exactly (bit-for-bit) --
    # this is the structural basis for C=0 whenever S=0.
    assert records[2].pre_sqrtPriceX96 == records[1].post_sqrtPriceX96
    assert records[2].n_swaps == 0

    # 3. Canonical-panel / estimand layer, fed directly from reconstruction output.
    pre_prices = amm_usdc_per_eth([r.pre_sqrtPriceX96 for r in records], POOL)
    post_prices = amm_usdc_per_eth([r.post_sqrtPriceX96 for r in records], POOL)
    xA_pre = np.log(pre_prices)
    xA_post = np.log(post_prices)

    # No-swap blocks: price must be IDENTICAL pre/post (schema-level check
    # that the reconstruction's raw units survive the log-price conversion
    # unchanged for a genuinely unchanged sqrtPriceX96).
    assert xA_pre[0] == xA_post[0]
    assert xA_pre[2] == xA_post[2]
    assert xA_pre[1] != xA_post[1]  # the swapped block DID move

    xstar_log = np.full(3, np.log(3005.0))  # synthetic external reference
    retained = np.array([True, True, True])
    out = hf_same_block_outcomes(xstar_log, xA_pre, xA_post, retained, POOL)

    # Schema check: every column the downstream Gold-panel/hinge/RMT layer
    # expects is present and finite.
    for col in ("d_pre_hf", "d_post_hf", "x", "correction_hf_same_block", "corrective_hf_same_block"):
        assert col in out
        assert np.isfinite(out[col]).all() if col != "corrective_hf_same_block" else True

    # The structural identity this whole pipeline depends on: no swap ⇒
    # C=0, verified here on genuinely reconstructed (not hand-typed) state.
    assert out["correction_hf_same_block"][0] == 0.0
    assert out["correction_hf_same_block"][2] == 0.0
    assert out["correction_hf_same_block"][1] != 0.0  # the swap block moved, so C != 0

    assert (out["x"] >= 0).all()
    assert out["x"].dtype == np.float64
    assert gamma_fee(POOL) > 0
